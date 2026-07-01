import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(precision=4, sci_mode=False)

class SpatioTemporalNetwork(nn.Module):


    def __init__(self, obs_dim, hidden_size, output_size, device,
                 recurrent_N=1, topk=5, num_heads=1):
        super(SpatioTemporalNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.recurrent_N = recurrent_N
        self.topk = topk
        self.num_heads = num_heads


        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=recurrent_N,
            batch_first=False
        )


        self.adjacency_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)


        self.gat_layer = MultiHeadGATLayer(
            in_features=hidden_size,
            out_features=output_size,
            num_heads=num_heads,
            concat=True
        )

        self.final_proj = nn.Linear(num_heads * output_size, output_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(output_size)

        self.to(device)

    def compute_adjacency_matrix_batch_optimized(self, hidden_states, n_agents, active_masks_batch):
        batch_total = hidden_states.size(0)
        batch_size = batch_total // n_agents


        hidden_reshaped = hidden_states.view(batch_size, n_agents, self.hidden_size)
        active_masks_reshaped = active_masks_batch.view(batch_size, n_agents)



        adj_features = self.adjacency_net(hidden_reshaped)


        queries = self.query_proj(adj_features)
        keys = self.key_proj(adj_features)

        attention_scores = torch.bmm(queries, keys.transpose(-2, -1)) / (self.hidden_size ** 0.5)



        mask_source = active_masks_reshaped.unsqueeze(1).float()
        mask_target = active_masks_reshaped.unsqueeze(2).float()
        attention_scores = attention_scores * mask_source * mask_target + \
                          (1 - mask_source * mask_target) * (-1e9)


        topk = min(self.topk, n_agents)
        topk_values, topk_indices = torch.topk(attention_scores, topk, dim=2)
        topk_weights = F.softmax(topk_values, dim=2)


        batch_indices = torch.arange(batch_size, device=self.device).view(-1, 1, 1).expand(-1, n_agents, topk)
        target_indices = torch.arange(n_agents, device=self.device).view(1, -1, 1).expand(batch_size, -1, topk)
        source_indices = topk_indices


        sources = source_indices + batch_indices * n_agents
        targets = target_indices + batch_indices * n_agents


        valid_edges = active_masks_reshaped.unsqueeze(2).expand(-1, -1, topk)


        sources_flat = sources.reshape(-1)[valid_edges.reshape(-1)]
        targets_flat = targets.reshape(-1)[valid_edges.reshape(-1)]
        weights_flat = topk_weights.reshape(-1)[valid_edges.reshape(-1)]

        edge_index = torch.stack([sources_flat, targets_flat], dim=0)

        return edge_index, weights_flat

    def forward(self, obs, rnn_states, active_masks, data_chunk_length):
        if rnn_states.dim() == 3:
            return self._forward_single_step(obs, rnn_states, active_masks)
        elif rnn_states.dim() == 4:


            return self._forward_batch_sequence_optimized_v2(obs, rnn_states, active_masks, data_chunk_length)


        else:
            raise ValueError(f"不支持的rnn_states维度: {rnn_states.dim()}")

    def _forward_single_step(self, obs, rnn_states, active_masks):
        n_rollouts, n_agents, obs_dim = obs.size()

        obs_reshaped = obs.reshape(1, n_rollouts * n_agents, obs_dim)

        if not rnn_states.is_contiguous():
            rnn_states = rnn_states.contiguous()


        obs_reshaped = obs_reshaped * active_masks.unsqueeze(0).unsqueeze(-1).float()
        rnn_states = rnn_states * active_masks.unsqueeze(0).unsqueeze(-1).float()



        gru_out, output_hidden_state = self.gru(obs_reshaped, rnn_states)




        agg_obs = gru_out.squeeze(0)
        agg_obs = agg_obs * active_masks.unsqueeze(1).float()
        agg_obs = self.layer_norm1(agg_obs)


        last_hidden = output_hidden_state[-1]
        last_hidden = last_hidden * active_masks.unsqueeze(1).float()


        edge_index, edge_weights = self.compute_adjacency_matrix_batch_optimized(
            last_hidden, n_agents, active_masks
        )



        gat_out = self.gat_layer(agg_obs, edge_index, edge_weights, active_masks)

        obs_features_flat = self.final_proj(gat_out)
        obs_features_flat = self.layer_norm2(obs_features_flat)


        if self.hidden_size == self.output_size:
            obs_features_flat = obs_features_flat + agg_obs


        obs_features_flat = obs_features_flat * active_masks.unsqueeze(1).float()
        obs_features = obs_features_flat.reshape(n_rollouts, n_agents, self.output_size)

        return obs_features, output_hidden_state

    def _forward_batch_sequence_optimized_v2(self, obs, rnn_states, active_masks, data_chunk_length):

        total_timesteps, n_agents, obs_dim = obs.size()
        num_chunks, n_agents_check, recurrent_N, rnn_dim = rnn_states.shape

        assert n_agents == n_agents_check, f"n_agents mismatch: {n_agents} vs {n_agents_check}"
        assert total_timesteps == num_chunks * data_chunk_length, "total_timesteps!=num_chunks*data_chunk_length"

        active_masks_reshaped = active_masks.view(total_timesteps, n_agents)


        obs_chunked = obs.view(num_chunks, data_chunk_length, n_agents, obs_dim)
        masks_chunked = active_masks_reshaped.view(num_chunks, data_chunk_length, n_agents)


        current_hidden = rnn_states.permute(2, 0, 1, 3).contiguous().view(recurrent_N, num_chunks * n_agents, rnn_dim)


        gru_outputs = []
        gru_hidden_states = []


        for t in range(data_chunk_length):

            step_obs = obs_chunked[:, t, :, :].unsqueeze(0).reshape(1, num_chunks * n_agents, obs_dim)
            step_mask = masks_chunked[:, t, :].reshape(num_chunks * n_agents)


            step_obs = step_obs * step_mask.unsqueeze(0).unsqueeze(-1).float()
            current_hidden = current_hidden * step_mask.unsqueeze(0).unsqueeze(-1).float()










            step_out, current_hidden = self.gru(step_obs, current_hidden)



            gru_outputs.append(step_out)

            gru_hidden_states.append(current_hidden.clone())


        gru_out_chunked = torch.cat(gru_outputs, dim=0)

        gru_out_chunked = gru_out_chunked.view(data_chunk_length, num_chunks, n_agents, self.hidden_size)
        gru_out = gru_out_chunked.permute(1, 0, 2, 3).contiguous().view(total_timesteps, n_agents, self.hidden_size)




        gru_hidden_stacked = torch.stack(gru_hidden_states, dim=0)
        gru_hidden_stacked = gru_hidden_stacked.view(data_chunk_length, recurrent_N, num_chunks, n_agents, rnn_dim)
        gru_hidden_stacked = gru_hidden_stacked.permute(2, 0, 3, 1, 4).contiguous()
        gru_hidden_state = gru_hidden_stacked.view(total_timesteps, n_agents, recurrent_N, rnn_dim)


        agg_obs = gru_out.reshape(total_timesteps * n_agents, self.hidden_size)
        agg_obs = agg_obs * active_masks.unsqueeze(1).float()
        agg_obs = self.layer_norm1(agg_obs)


        last_hidden = gru_hidden_state[:, :, -1, :]
        last_hidden = last_hidden.reshape(total_timesteps * n_agents, self.hidden_size)
        last_hidden = last_hidden * active_masks.unsqueeze(1).float()


        edge_index, edge_weights = self.compute_adjacency_matrix_batch_optimized(
            last_hidden, n_agents, active_masks
        )


        gat_out = self.gat_layer(agg_obs, edge_index, edge_weights, active_masks)


        obs_features_flat = self.final_proj(gat_out)
        obs_features_flat = self.layer_norm2(obs_features_flat)

        if self.hidden_size == self.output_size:
            obs_features_flat = obs_features_flat + agg_obs

        obs_features_flat = obs_features_flat * active_masks.unsqueeze(1).float()
        obs_features = obs_features_flat.reshape(total_timesteps, n_agents, self.output_size)


        output_hidden_state = gru_hidden_state[data_chunk_length-1::data_chunk_length]





        return obs_features, output_hidden_state



    def _forward_batch_sequence_optimized(self, obs, rnn_states, active_masks, data_chunk_length):

        total_timesteps, n_agents, obs_dim = obs.size()
        num_chunks, n_agents_check, recurrent_N, rnn_dim = rnn_states.shape

        assert n_agents == n_agents_check, f"n_agents mismatch: {n_agents} vs {n_agents_check}"
        assert total_timesteps == num_chunks * data_chunk_length, "total_timesteps!=num_chunks*data_chunk_length"


        gru_out = torch.zeros((total_timesteps, n_agents, self.hidden_size), device=self.device)
        gru_hidden_state = torch.zeros((total_timesteps, n_agents, recurrent_N, rnn_dim), device=self.device)


        active_masks_reshaped = active_masks.view(total_timesteps, n_agents)


        for n_chunk in range(num_chunks):

            sigle_rnn = rnn_states[n_chunk:n_chunk+1]
            sigle_rnn = sigle_rnn.squeeze(0).permute(1, 0, 2).contiguous()

            for time_step in range(data_chunk_length):
                total_index = n_chunk * data_chunk_length + time_step
                sigle_obs = obs[total_index:total_index+1]
                sigle_active_mask = active_masks_reshaped[total_index:total_index+1]


                sigle_obs_masked = sigle_obs * sigle_active_mask.unsqueeze(-1).float()
                sigle_rnn_masked = sigle_rnn * sigle_active_mask.view(1, n_agents, 1).float()






                sigle_gru_out, sigle_gru_hidden_state = self.gru(sigle_obs_masked, sigle_rnn_masked)







                sigle_rnn = sigle_gru_hidden_state

                gru_out[total_index] = sigle_gru_out.squeeze(0)
                gru_hidden_state[total_index] = sigle_gru_hidden_state.permute(1, 0, 2)






        agg_obs = gru_out.reshape(total_timesteps * n_agents, self.hidden_size)

        agg_obs = agg_obs * active_masks.unsqueeze(1).float()
        agg_obs = self.layer_norm1(agg_obs)



        last_hidden = gru_hidden_state[:, :, -1, :]
        last_hidden = last_hidden.reshape(total_timesteps * n_agents, self.hidden_size)
        last_hidden = last_hidden * active_masks.unsqueeze(1).float()


        edge_index, edge_weights = self.compute_adjacency_matrix_batch_optimized(
            last_hidden, n_agents, active_masks
        )





        gat_out = self.gat_layer(agg_obs, edge_index, edge_weights, active_masks)


        obs_features_flat = self.final_proj(gat_out)
        obs_features_flat = self.layer_norm2(obs_features_flat)


        if self.hidden_size == self.output_size:
            obs_features_flat = obs_features_flat + agg_obs


        obs_features_flat = obs_features_flat * active_masks.unsqueeze(1).float()


        obs_features = obs_features_flat.reshape(total_timesteps, n_agents, self.output_size)



        output_hidden_state = torch.zeros((num_chunks, n_agents, recurrent_N, rnn_dim), device=self.device)
        for n_chunk in range(num_chunks):
            last_step_in_chunk = (n_chunk + 1) * data_chunk_length - 1
            output_hidden_state[n_chunk] = gru_hidden_state[last_step_in_chunk]
        return obs_features, output_hidden_state






































    def init_hidden(self, batch_size, n_agents):
        return torch.zeros(
            self.recurrent_N,
            batch_size * n_agents,
            self.hidden_size
        ).to(self.device)


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, concat=True):
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.concat = concat

        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, concat=True)
            for _ in range(num_heads)
        ])

    def forward(self, h, edge_index, edge_weights=None, mask=None):
        head_outputs = [att(h, edge_index, edge_weights) for att in self.attentions]

        if self.concat:
            return torch.cat(head_outputs, dim=1)
        else:
            return torch.mean(torch.stack(head_outputs), dim=0)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_index, edge_weights=None):
        Wh = torch.mm(h, self.W)
        N = Wh.size(0)

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        Wh_source = Wh[source_nodes]
        Wh_target = Wh[target_nodes]

        attention_input = torch.cat([Wh_source, Wh_target], dim=1)
        e = self.leakyrelu(torch.matmul(attention_input, self.a).squeeze(1))

        if edge_weights is not None:
            e = e * edge_weights

        attention = torch.zeros(N, device=h.device).scatter_add_(
            0, target_nodes, torch.exp(e)
        )
        attention = attention + 1e-10
        alpha = torch.exp(e) / attention[target_nodes]

        h_prime = torch.zeros((N, self.out_features), device=h.device)
        h_prime.scatter_add_(
            0,
            target_nodes.unsqueeze(1).expand(-1, self.out_features),
            alpha.unsqueeze(1) * Wh_source
        )

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
