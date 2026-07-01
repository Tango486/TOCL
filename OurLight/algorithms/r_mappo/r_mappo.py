
import numpy as np
import torch
import torch.nn as nn
from OurLight.utils.util import get_gard_norm, huber_loss, mse_loss
from OurLight.utils.valuenorm import ValueNorm
from OurLight.algorithms.utils.util import check

import torch.nn.functional as F

class R_MAPPO_Trans():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.args =args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:

            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None

        self.n_rollout_threads, self.episode_length =  args.n_rollout_threads, args.episode_length
        self.td_error = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original


        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()

        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['loss/value_loss'] = 0
        train_info['loss/policy_loss'] = 0
        train_info['loss/dist_entropy'] = 0
        train_info['loss/actor_grad_norm'] = 0
        train_info['loss/critic_grad_norm'] = 0
        train_info['loss/ratio'] = 0





        self.num_max_agents = advantages.shape[-2]
        self.td_error = torch.zeros((self.ppo_epoch, self.n_rollout_threads*self.episode_length, self.num_max_agents, 1)).to(self.device)
        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:

                if self.args.use_ours:

                    data_generator = buffer.recurrent_generator_graph(advantages, self.num_mini_batch, self.data_chunk_length)
                else:
                    data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)

            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                if self.args.use_ours:
                    data_generator = buffer.feed_forward_generator_graph(advantages, self.num_mini_batch)
                else:
                    data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)



            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, active_masks_batch \
                        = self.ppo_update(sample, _)

                train_info['loss/value_loss'] += value_loss.item()
                train_info['loss/policy_loss'] += policy_loss.item()
                train_info['loss/dist_entropy'] += dist_entropy.item()
                train_info['loss/actor_grad_norm'] += actor_grad_norm
                train_info['loss/critic_grad_norm'] += critic_grad_norm
                train_info['loss/ratio'] += torch.sum(imp_weights)/torch.sum(active_masks_batch.float())
        self.td_error = torch.mean(self.td_error.reshape((self.ppo_epoch, self.n_rollout_threads, self.episode_length, self.num_max_agents, 1)), dim=0)
        ac_ag = buffer.active_masks[0]
        self.td_error = self.td_error.detach().cpu().numpy() * np.expand_dims(np.expand_dims(ac_ag, axis=1), axis=-1)
        self.td_error = self.td_error.sum(1).sum(1).squeeze(-1)
        self.td_error = self.td_error / (ac_ag.sum(-1) * self.episode_length)
        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
        train_info["td_error"] = self.td_error
        return train_info


    def ppo_update(self, sample, up_index):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, actor_features_batch, indices = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(self.device)





















        obs_batch = check(obs_batch).to(**self.tpdv)


        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(obs_batch,
                                                                              rnn_states_batch,
                                                                              rnn_states_critic_batch,
                                                                              actions_batch,
                                                                              available_actions_batch,
                                                                              actor_features_batch,
                                                                              active_masks_batch
                                                                              )

        if self._use_valuenorm:
            values = self.value_normalizer.normalize(values)





        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)


        imp_weights = imp_weights * active_masks_batch.float()





        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss


        start = indices * self.data_chunk_length
        end = start + self.data_chunk_length
        td_error = (values - return_batch).abs()


        for i, (sat, ed) in enumerate(zip(start, end)):
            self.td_error[up_index, sat: ed] = td_error[i*self.data_chunk_length: (i+1)*self.data_chunk_length]

        if len(values.shape) == 3:
            T_, B_ = values.shape[0], values.shape[1]
            values = values.reshape(T_*B_, -1)
            value_preds_batch = value_preds_batch.reshape(T_*B_, -1)
            return_batch = return_batch.reshape(T_*B_, -1)
            active_masks_batch = active_masks_batch.reshape(T_*B_, -1)
        value_losses = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)



        self.policy.actor_optimizer.zero_grad()
        (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()



        self.policy.critic_optimizer.zero_grad()
        (value_losses * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()
        return value_losses, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, active_masks_batch


    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
