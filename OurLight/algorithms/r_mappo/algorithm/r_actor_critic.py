import torch
import torch.nn as nn
from OurLight.algorithms.utils.util import init, check
from OurLight.algorithms.utils.cnn import CNNBase
from OurLight.algorithms.utils.mlp import MLPBase
from OurLight.algorithms.utils.rnn import RNNLayer
from OurLight.algorithms.utils.act import ACTLayer, ACTLayer_TopK
from OurLight.algorithms.utils.popart import PopArt
from OurLight.utils.util import get_shape_from_obs_space

from OurLight.algorithms.r_mappo.algorithm.trans_net import *
from gym import spaces
import numpy as np
import time


from OurLight.algorithms.r_mappo.algorithm.sp import SpatioTemporalNetwork

class R_Actor_Trans(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor_Trans, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.data_chunk_length = args.data_chunk_length

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        obs_shape = get_shape_from_obs_space(obs_space)
















        self.model = SpatioTemporalNetwork(obs_dim=obs_shape[0], hidden_size=args.trans_hidden, output_size=args.trans_hidden, device=device)

        self.hidden_size = args.trans_hidden

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain).to(device)









        self.to(device)

    def forward(self, obs, rnn_states, available_actions=None, deterministic=False, active_masks=None, act_f=None):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        if act_f:
            roll_outs = act_f
        else:
            roll_outs = self.args.n_rollout_threads
        obs = np.array(np.split(obs, roll_outs))

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        active_masks = check(active_masks).to(self.device)
        available_actions = check(available_actions).to(**self.tpdv)


        rnn_states = rnn_states.permute(1, 0, 2)

        actor_features_, rnn_states = self.model(obs, rnn_states=rnn_states, active_masks=active_masks, data_chunk_length=self.data_chunk_length)



        rnn_states = rnn_states.permute(1, 0, 2)
        actor_features = actor_features_.reshape(-1, actor_features_.shape[-1])














        actions, action_log_probs = self.act(actor_features, available_actions, deterministic, active_masks)



        return actions, action_log_probs, rnn_states, actor_features

    def evaluate_actions(self, obs, rnn_states, action, available_actions=None, active_masks=None):











        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        active_masks = check(active_masks).to(self.device)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv).long()







        actor_features_, rnn_states = self.model(obs, rnn_states=rnn_states, active_masks=active_masks, data_chunk_length=self.data_chunk_length)





        if len(actor_features_.shape) == 3:
            T_, B_ = actor_features_.shape[0], actor_features_.shape[1]
            actor_features = actor_features_.reshape(T_*B_, -1)
            action = action.reshape(T_*B_, -1)
            available_actions = available_actions.reshape(T_*B_, -1)

            actor_features = actor_features.reshape(T_*B_, -1)

            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=active_masks)
            action_log_probs = action_log_probs.reshape(T_, B_, 1)































        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=active_masks)


        return action_log_probs, dist_entropy, actor_features_

























































































































































class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        base = MLPBase

        self.base = base(args, (args.trans_hidden,))

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.device = device
        self.to(device)

    def forward(self, cent_obs, rnn_states, backward=False, active_masks=None):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        active_masks = check(active_masks).to(self.device)


        if len(cent_obs.shape) == 3:
            T_, B_ = cent_obs.shape[0], cent_obs.shape[1]
            cent_obs = cent_obs.reshape(T_*B_, -1)

            critic_features = self.base(cent_obs, active_masks)
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                critic_features, rnn_states = self.rnn(critic_features, rnn_states, active_masks)
            values = self.v_out(critic_features)

            if active_masks is not None:
                values = values * active_masks.unsqueeze(-1).float()
            values = values.reshape(T_, B_, 1)

        else:
            critic_features = self.base(cent_obs, active_masks)
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                critic_features, rnn_states = self.rnn(critic_features, rnn_states, active_masks)
            values = self.v_out(critic_features)

            if active_masks is not None:
                values = values * active_masks.unsqueeze(-1).float()
        return values, rnn_states
