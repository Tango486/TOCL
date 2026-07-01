import torch
import torch.nn as nn

"""RNN modules."""


class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, active_masks=None):
        """
        Args:
            x: 输入特征
            hxs: GRU hidden states
            masks: reset masks for GRU
            active_masks: (batch_size,) bool tensor，标识有效的agents
        """

        if x.size(0) == hxs.size(0):









            if active_masks is not None:
                active_masks_expanded = active_masks.unsqueeze(-1).unsqueeze(-1).float()



            x_rnn = x.unsqueeze(0)
            hxs_rnn = (hxs * active_masks_expanded).transpose(0, 1).contiguous()

            x, hxs = self.rnn(x_rnn, hxs_rnn)

            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)


            if active_masks is not None:
                x = x * active_masks.unsqueeze(-1).float()
                hxs = hxs * active_masks.unsqueeze(-1).unsqueeze(-1).float()

        else:


            N = hxs.size(0)
            n_agents = hxs.size(1)
            T = int(x.size(0) / (N * n_agents))


            x = x.view(T, N * n_agents, x.size(1))
            hxs = hxs.reshape(-1, *hxs.shape[2:]).permute(1, 0, 2)


            if active_masks is not None:

                active_masks_seq = active_masks.view(T, N * n_agents)



                first_step_mask = active_masks_seq[0]
                hxs = hxs * first_step_mask.unsqueeze(0).unsqueeze(-1).float()


            x, hxs = self.rnn(x, hxs)




            x = x.reshape(-1, x.shape[-1])


            if active_masks is not None:
                x = x * active_masks.unsqueeze(-1).float()

        x = self.norm(x)
        return x, hxs
