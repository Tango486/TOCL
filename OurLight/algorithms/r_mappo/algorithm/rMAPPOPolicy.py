import torch
from OurLight.algorithms.r_mappo.algorithm.r_actor_critic import R_Critic, R_Actor_Trans
from OurLight.utils.util import update_linear_schedule
from OurLight.algorithms.utils.util import init, check
import time

import numpy as np

class R_MAPPOPolicy_Trans:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):

        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space

        self.act_space = act_space
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.actor = R_Actor_Trans(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                    lr=self.critic_lr,
                                                    eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)

























    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, obs, rnn_states_actor, rnn_states_critic, available_actions=None, active_masks=None):

        """
        Compute actions and value function predictions for the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.  # (rollout*max_n_agent, obs_dim)
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.   # (rollout*max_n_agent, N_recurrent, feat_dim)
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.  # (rollout*max_n_agent, N_recurrent, feat_dim)
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.    # (rollout*max_n_agent, 1)
        :param available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available)   #  (rollout*max_n_agent, 8)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        :param active_masks: bool, (rollout*max_n_agent, )

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        deterministic = False
        if np.random.uniform() > self.args.epsilon:
            deterministic = True

        actions, action_log_probs, rnn_states_actor, actor_features = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 available_actions,
                                                                 deterministic,
                                                                 active_masks=active_masks
                                                                 )

        values, rnn_states_critic = self.critic(actor_features, rnn_states_critic, active_masks=active_masks)



        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, actor_features, active_masks






























    def get_values_three(self, obs, rnn_states_actor, rnn_states_critic, available_actions=None,
                    deterministic=False, active_masks=None):

        actions, action_log_probs, rnn_states_actor, actor_features = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 available_actions,
                                                                 deterministic,
                                                                 active_masks)
        values, _ = self.critic(actor_features, rnn_states_critic, active_masks=active_masks)

        return values




    def evaluate_actions(self, obs, rnn_states_actor, rnn_states_critic, action, available_actions=None, actor_features_batch=None, active_masks_batch=None):

        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        active_masks_batch = active_masks_batch.squeeze(axis=-1)
        active_masks_batch = active_masks_batch.reshape(-1)







        action_log_probs, dist_entropy, actor_features_= self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     available_actions,
                                                                     active_masks_batch
                                                                     )




        actor_features_detached = actor_features_.detach()
        values, _ = self.critic(actor_features_detached, rnn_states_critic, backward=True, active_masks=active_masks_batch)

        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, available_actions=None, deterministic=False, active_masks=None, act_f=None):

        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.  (rollout*max_n_agents, feat_dim)
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.  # (rollout*max_n_agents, N_recurrent, feat_dim)
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.  # (rollout*max_n_agents, 1)
        :param available_actions: (np.ndarray) denotes which actions are available to agent   # (rollout*max_n_agents, 8)
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        :active_masks: (np.array), bool, (rollout*max_n_agents,)
        """
        actions, action_log_probs, rnn_states_actor, actor_features = self.actor(obs, rnn_states_actor, available_actions, deterministic, active_masks=active_masks, act_f=act_f)
        return actions, rnn_states_actor, actor_features
