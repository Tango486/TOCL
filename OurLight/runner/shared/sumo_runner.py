import csv
import json
import os
import random
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from gym import spaces

from GenRandomGrids.gen_random_grid import GEN_RANDOM_GRID
from OurLight.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy_Trans as Policy
from OurLight.algorithms.r_mappo.r_mappo import R_MAPPO_Trans as TrainAlgo
from OurLight.curriculum.tocl_sampler import TOCLSampler
from OurLight.envs.env_wrappers import SubprocVecEnv
from OurLight.envs.sumo_files_marl.SUMO_env import SUMOEnv
from OurLight.utils.shared_buffer import SharedReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


def make_train_env(all_args, env_configs):
    def get_env_fn(rank, sumo_cfg):
        def init_env():
            env = SUMOEnv(all_args, rank, sumo_cfg)
            env.set_seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return SubprocVecEnv([
        get_env_fn(rank, env_config["sumo_cfg"])
        for rank, env_config in enumerate(env_configs)
    ])


class CSVLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row):
        row = dict(row)
        write_header = not self.path.exists()
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def close(self):
        return


class SUMORunner:
    """Source only TOCL training loop for generated SUMO contexts."""

    def __init__(self, config):
        self.all_args = config["all_args"]
        self.device = config["device"]
        self.run_dir = Path(config["run_dir"])
        self.log_dir = self.run_dir / "logs"
        self.save_dir = self.run_dir / "models"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.num_env_steps = int(self.all_args.num_env_steps)
        self.episode_length = int(self.all_args.episode_length)
        self.n_rollout_threads = int(self.all_args.n_rollout_threads)
        self.use_linear_lr_decay = bool(self.all_args.use_linear_lr_decay)
        self.use_centralized_V = bool(self.all_args.use_centralized_V)
        self.recurrent_N = int(self.all_args.recurrent_N)
        self.hidden_size = int(self.all_args.hidden_size)
        self.save_interval = int(self.all_args.save_interval)
        self.model_dir = self.all_args.model_dir

        repo_root = Path(__file__).resolve().parents[3]
        self.envs_root = repo_root / "OurLight" / "envs"
        generated_root = getattr(self.all_args, "generated_output_dir", None)
        if generated_root in (None, "", "None"):
            generated_root = (
                self.envs_root
                / "sumo_files_marl"
                / "generated_source_runs"
                / self.run_dir.name
            )
        else:
            generated_root = Path(generated_root)
            if not generated_root.is_absolute():
                generated_root = repo_root / generated_root
        self.generated_root = Path(generated_root)
        self.generated_root.mkdir(parents=True, exist_ok=True)

        obs_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.all_args.obs_shape,),
            dtype=np.float32,
        )
        act_space = spaces.Discrete(self.all_args.num_actions)
        self.policy = Policy(self.all_args, obs_space, act_space, device=self.device)
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
        if self.all_args.load_model:
            self.restore()

        self.param_env_bounds = self._default_param_bounds()
        self._apply_generator_bounds_override()
        self.norm_param_env_bounds = OrderedDict(
            (key, [0.0, 1.0]) for key in self.param_env_bounds
        )
        self.sampler = TOCLSampler(
            self.norm_param_env_bounds,
            seed=self.all_args.seed,
            args=self.all_args,
            run_dir=str(self.run_dir),
        )
        self.gen_random_grid = GEN_RANDOM_GRID(
            whole_output_path=str(self.generated_root)
        )

        self.phase_episodes = max(1, int(self.all_args.tocl_phase_episodes))
        self.phase_id = 0
        self.phase_episode_count = 0
        self.phase_feedback = []
        self.phase_env_configs = None
        self.phase_sampler_info = None
        self.phase_norm_params = None
        self.phase_raw_params = None

        self.train_logger = CSVLogger(self.log_dir / "train_metrics.csv")
        self.context_logger = CSVLogger(self.log_dir / "tocl_context_records.csv")
        self.envs = None

    def _default_param_bounds(self):
        return OrderedDict([
            ("x_nodes", [2.0, 6.0]),
            ("y_nodes", [2.0, 6.0]),
            ("x_len", [300.0, 600.0]),
            ("y_len", [300.0, 600.0]),
            ("del_factor", [0.0, 0.4]),
            ("flow_scale", [0.2, 5.0]),
            ("turn_left_ratio", [0.1, 0.4]),
            ("spatial_imbalance", [0.0, 0.8]),
            ("temporal_variation", [0.0, 0.8]),
        ])

    def _apply_generator_bounds_override(self):
        bounds_path = getattr(self.all_args, "generator_bounds_json", None)
        if bounds_path in (None, "", "None"):
            return
        path = Path(bounds_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        with path.open() as f:
            payload = json.load(f)
        bounds = payload.get("bounds", payload)
        for key, value in bounds.items():
            if key not in self.param_env_bounds:
                continue
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"Invalid bound for {key}: {value}")
            lo, hi = float(value[0]), float(value[1])
            if hi < lo:
                raise ValueError(f"Invalid bound for {key}: {value}")
            self.param_env_bounds[key] = [lo, hi]

    def denormalize(self, param_name, normalized_value):
        lo, hi = self.param_env_bounds[param_name]
        return float(normalized_value) * (hi - lo) + lo

    def build_raw_env_params(self, norm_params, sampler_info):
        denorm = {
            key: self.denormalize(key, value)
            for key, value in norm_params.items()
        }
        x_nodes = int(denorm["x_nodes"] + 0.5)
        y_nodes = int(denorm["y_nodes"] + 0.5)
        del_num_nodes = int(denorm["del_factor"] * x_nodes * y_nodes)
        generation_seed = int(sampler_info.get("generation_seed", self.all_args.seed))
        return {
            "x_nodes": x_nodes,
            "y_nodes": y_nodes,
            "x_len": denorm["x_len"],
            "y_len": denorm["y_len"],
            "attach_len": 300,
            "del_num_nodes": del_num_nodes,
            "sim_time": 3600,
            "base_mean_flow": 600,
            "flow_scale": denorm["flow_scale"],
            "turn_left_ratio": denorm["turn_left_ratio"],
            "spatial_imbalance": denorm["spatial_imbalance"],
            "temporal_variation": denorm["temporal_variation"],
            "random_seed": generation_seed % 1000000,
        }

    def _relative_sumo_cfg(self, sumo_cfg_abs):
        path = Path(sumo_cfg_abs).resolve()
        return str(path.relative_to(self.envs_root.resolve()))

    def _make_phase_configs(self, total_num_steps):
        params, sampler_info = self.sampler.sample(total_num_steps)
        norm_params = OrderedDict(
            (key, float(value))
            for key, value in zip(self.sampler.param_keys, params)
        )
        raw_params = self.build_raw_env_params(norm_params, sampler_info)
        sumo_cfg_abs = self.gen_random_grid.gen_random_sumocfg(
            rollout_id=self.phase_id,
            **raw_params,
        )
        sumo_cfg = self._relative_sumo_cfg(sumo_cfg_abs)
        configs = []
        for rollout_id in range(self.n_rollout_threads):
            configs.append({
                "type": "source",
                "sumo_cfg": sumo_cfg,
                "rollout_id": rollout_id,
                "norm_env_params": OrderedDict(norm_params),
                "env_params": dict(raw_params),
                "tocl_info": dict(sampler_info),
            })
        self.phase_id += 1
        self.phase_episode_count = 0
        self.phase_feedback = []
        self.phase_env_configs = configs
        self.phase_sampler_info = dict(sampler_info)
        self.phase_norm_params = OrderedDict(norm_params)
        self.phase_raw_params = dict(raw_params)
        self._append_context_record(
            total_num_steps,
            configs[0],
            event="sample",
            feedback=None,
            update_info=None,
        )
        return configs

    def set_source_phase(self, total_num_steps):
        if (
            self.phase_env_configs is not None
            and self.phase_episode_count < self.phase_episodes
        ):
            configs = self.phase_env_configs
        else:
            configs = self._make_phase_configs(total_num_steps)
        self.envs = make_train_env(self.all_args, configs)
        self.num_agents_list = [
            len(action_space) for action_space in self.envs.action_space
        ]
        self.max_n_agents = max(self.num_agents_list)
        share_observation_space = (
            self.envs.share_observation_space[0]
            if self.use_centralized_V
            else self.envs.observation_space[0]
        )
        observation_space = self.envs.observation_space[0]
        action_space = self.envs.action_space[0]
        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.max_n_agents,
            observation_space[0],
            share_observation_space[0],
            action_space[0],
        )

    def run(self):
        start = time.time()
        episodes = self.num_env_steps // self.episode_length // self.n_rollout_threads
        if episodes <= 0:
            raise ValueError("num_env_steps is too small for the configured rollout size.")
        self.all_args.anneal_steps = max(1, int(0.8 * episodes * self.episode_length))
        anneal_epsilon = (
            float(self.all_args.epsilon) - float(self.all_args.min_epsilon)
        ) / self.all_args.anneal_steps

        total_num_steps = 0
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            self.set_source_phase(total_num_steps)
            self.warmup()
            final_infos = None
            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actor_features, active_masks = self.collect(step)
                actions = actions.squeeze(axis=-1)
                env_actions = [
                    actions[i][:self.num_agents_list[i]]
                    for i in range(actions.shape[0])
                ]
                obs_list, rewards_list, dones_list, infos = self.envs.step(env_actions)
                final_infos = infos
                data = (
                    obs_list,
                    rewards_list,
                    dones_list,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actor_features,
                    active_masks,
                )
                self.insert(data)
                if self.all_args.epsilon > self.all_args.min_epsilon:
                    self.all_args.epsilon = max(
                        self.all_args.min_epsilon,
                        self.all_args.epsilon - anneal_epsilon,
                    )

            self.compute()
            train_infos = self.train()
            td_error = train_infos.pop("td_error", None)
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            average_episode_reward = self._average_episode_reward()
            op_feedback_by_rollout = self._operation_feedback(final_infos)
            td_by_rollout = self._td_error_by_rollout(td_error)
            episode_feedback = self._episode_feedback(
                op_feedback_by_rollout,
                td_by_rollout,
                average_episode_reward,
            )
            self.phase_feedback.append(episode_feedback)
            self.phase_episode_count += 1

            train_infos["average_episode_reward"] = average_episode_reward
            self._append_train_record(
                episode,
                total_num_steps,
                train_infos,
                episode_feedback,
                time.time() - start,
            )

            save_final_episode_interval = getattr(
                self.all_args,
                "save_final_episode_interval",
                None,
            )
            save_by_completed_episode = (
                save_final_episode_interval is not None
                and save_final_episode_interval > 0
                and (episode + 1) % save_final_episode_interval == 0
            )
            if (
                episode % self.save_interval == 0
                or save_by_completed_episode
                or episode == episodes - 1
            ):
                self.save(episode)

            should_update = (
                self.phase_episode_count >= self.phase_episodes
                or episode == episodes - 1
            )
            if should_update:
                self._update_sampler(total_num_steps)

            self.envs_close()

    def _average_episode_reward(self):
        active = np.expand_dims(self.buffer.active_masks[:-1], axis=-1)
        active_count = np.maximum(active.sum(), 1.0)
        return float((self.buffer.rewards * active).sum() / active_count)

    def _operation_feedback(self, final_infos):
        if final_infos is None:
            return []
        feedback_rows = []
        for info in final_infos:
            values = defaultdict(list)
            for agent_id in info.keys():
                for key, value in info[agent_id].items():
                    if value is None:
                        continue
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(numeric):
                        continue
                    if key in {"queue_len", "wait_time", "delay_time", "pressure"}:
                        numeric = max(-numeric, 0.0)
                    values[key].append(numeric)
            row = {}
            for key in ["queue_len", "wait_time", "delay_time", "pressure"]:
                row[key] = float(np.mean(values[key])) if values[key] else None
            feedback_rows.append(row)
        return feedback_rows

    def _td_error_by_rollout(self, td_error):
        if td_error is None:
            return [None] * self.n_rollout_threads
        values = np.asarray(td_error).reshape(-1)
        rows = []
        for rollout_id in range(self.n_rollout_threads):
            if rollout_id < len(values) and np.isfinite(values[rollout_id]):
                rows.append(float(values[rollout_id]))
            else:
                rows.append(None)
        return rows

    def _episode_feedback(self, op_rows, td_rows, episode_reward):
        rows = []
        for rollout_id in range(self.n_rollout_threads):
            row = dict(op_rows[rollout_id]) if rollout_id < len(op_rows) else {}
            row["td_error"] = td_rows[rollout_id] if rollout_id < len(td_rows) else None
            row["reward"] = episode_reward
            rows.append(row)
        merged = {"reward": episode_reward}
        for key in ["queue_len", "wait_time", "delay_time", "pressure", "td_error"]:
            values = [row.get(key) for row in rows if row.get(key) is not None]
            merged[key] = float(np.mean(values)) if values else None
        merged["by_rollout"] = rows
        return merged

    def _update_sampler(self, total_num_steps):
        if not self.phase_feedback or self.phase_sampler_info is None:
            self.phase_env_configs = None
            return
        feedback = {
            "info": self.phase_sampler_info,
            "rollout_id": "phase",
        }
        for key in ["reward", "td_error", "queue_len", "wait_time", "delay_time", "pressure"]:
            values = [
                row.get(key)
                for row in self.phase_feedback
                if row.get(key) is not None
            ]
            feedback[key] = float(np.mean(values)) if values else None

        update_infos = self.sampler.update([feedback], total_num_steps)
        update_info = update_infos[0] if update_infos else None
        self.sampler.dump(str(self.run_dir / "tocl_sampler_state" / "record.pkl"))
        representative = self.phase_env_configs[0] if self.phase_env_configs else {}
        self._append_context_record(
            total_num_steps,
            representative,
            event="update",
            feedback=feedback,
            update_info=update_info,
        )
        self.phase_env_configs = None
        self.phase_feedback = []
        self.phase_episode_count = 0

    def _append_train_record(self, episode, total_num_steps, train_infos, feedback, elapsed):
        row = {
            "episode": int(episode),
            "total_num_steps": int(total_num_steps),
            "elapsed_sec": float(elapsed),
            "phase_id": int(self.phase_id - 1),
            "phase_episode": int(self.phase_episode_count),
            "epsilon": float(self.all_args.epsilon),
            "reward": feedback.get("reward"),
            "td_error": feedback.get("td_error"),
            "queue_len": feedback.get("queue_len"),
            "wait_time": feedback.get("wait_time"),
            "delay_time": feedback.get("delay_time"),
            "pressure": feedback.get("pressure"),
        }
        for key, value in sorted(train_infos.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = value
        self.train_logger.append(row)

    def _append_context_record(self, total_num_steps, env_config, event, feedback, update_info):
        info = env_config.get("tocl_info") or self.phase_sampler_info or {}
        norm_params = env_config.get("norm_env_params") or self.phase_norm_params or {}
        raw_params = env_config.get("env_params") or self.phase_raw_params or {}
        row = {
            "total_num_steps": int(total_num_steps),
            "event": event,
            "phase_id": int(self.phase_id - 1),
            "phase_episodes": int(self.phase_episode_count),
            "source": info.get("source"),
            "level_id": info.get("level_id"),
            "score": info.get("score"),
            "raw_score": info.get("raw_score"),
            "n_seen": info.get("n_seen"),
            "n_sampled": info.get("n_sampled"),
            "staleness": info.get("staleness"),
            "sumo_cfg": env_config.get("sumo_cfg"),
            "reward": None if feedback is None else feedback.get("reward"),
            "td_error": None if feedback is None else feedback.get("td_error"),
            "queue_len": None if feedback is None else feedback.get("queue_len"),
            "wait_time": None if feedback is None else feedback.get("wait_time"),
            "delay_time": None if feedback is None else feedback.get("delay_time"),
            "pressure": None if feedback is None else feedback.get("pressure"),
            "updated_score": None if update_info is None else update_info.get("updated_score"),
            "tocl_priority": None if update_info is None else update_info.get("tocl_priority"),
            "tocl_operational_score": None if update_info is None else update_info.get("tocl_operational_score"),
            "tocl_static_score": info.get("tocl_static_score") if update_info is None else update_info.get("tocl_static_score"),
        }
        for key in self.sampler.param_keys:
            row[f"norm_{key}"] = norm_params.get(key)
        for key in [
            "x_nodes",
            "y_nodes",
            "x_len",
            "y_len",
            "del_num_nodes",
            "flow_scale",
            "turn_left_ratio",
            "spatial_imbalance",
            "temporal_variation",
            "random_seed",
        ]:
            row[key] = raw_params.get(key)
        self.context_logger.append(row)

    def warmup(self):
        obs_list = self.envs.reset()
        self.ava_list = self.envs.get_unava_phase_index()
        for rollout_id in range(self.n_rollout_threads):
            n_agents = obs_list[rollout_id].shape[1]
            self.buffer.obs[0, rollout_id, :n_agents] = obs_list[rollout_id][0]
            self.buffer.available_actions[0, rollout_id, :n_agents] = self.get_ava_actions(
                self.ava_list[rollout_id],
                rollout_id,
            )[0]
            self.buffer.active_masks[0, rollout_id, :n_agents] = True
        self.buffer.available_actions[1:] = self.buffer.available_actions[0]

    def get_ava_actions(self, unavailable, rollout_id):
        available_actions = np.ones(
            (1, self.num_agents_list[rollout_id], self.all_args.num_actions),
            dtype=np.float32,
        )
        if unavailable is None:
            return available_actions
        arr = np.asarray(unavailable)
        if arr.size == 0:
            return available_actions
        if arr.ndim == 2:
            for agent_id in range(self.num_agents_list[rollout_id]):
                action_id = int(arr[0][agent_id])
                if 0 <= action_id < self.all_args.num_actions:
                    available_actions[0, agent_id, action_id] = 0.0
        elif arr.shape[-1] != 0:
            for agent_id in range(self.num_agents_list[rollout_id]):
                action_id = int(arr[0][agent_id][0])
                if 0 <= action_id < self.all_args.num_actions:
                    available_actions[0, agent_id, action_id] = 0.0
        return available_actions

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic, actor_features, active_masks = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            available_actions=np.concatenate(self.buffer.available_actions[step]),
            active_masks=np.concatenate(self.buffer.active_masks[step]),
        )
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        actor_features = np.array(np.split(_t2n(actor_features), self.n_rollout_threads))
        active_masks = np.array(np.split(active_masks, self.n_rollout_threads))
        if self.trainer._use_valuenorm:
            values = self.trainer.value_normalizer.normalize(values).cpu().numpy()
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actor_features, active_masks

    def insert(self, data):
        obs_list, rewards_list, dones_list, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, actor_features, active_masks = data
        obs = np.zeros((self.n_rollout_threads, self.max_n_agents, obs_list[0].shape[-1]), dtype=np.float32)
        rewards = np.zeros((self.n_rollout_threads, self.max_n_agents, 1), dtype=np.float32)
        actions = np.expand_dims(actions, axis=-1)
        for rollout_id in range(self.n_rollout_threads):
            n_agents = self.num_agents_list[rollout_id]
            obs[rollout_id, :n_agents] = obs_list[rollout_id]
            rewards[rollout_id, :n_agents] = rewards_list[rollout_id]
        self.buffer.insert(
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            score=None,
            score_log_probs=None,
            actor_features=actor_features,
            active_masks=active_masks,
        )

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values_three(
            np.concatenate(self.buffer.obs[-1]),
            np.concatenate(self.buffer.rnn_states[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            available_actions=np.concatenate(self.buffer.available_actions[-1]),
            active_masks=np.concatenate(self.buffer.active_masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        if self.trainer._use_valuenorm:
            next_values_t = torch.from_numpy(next_values).to(self.device)
            next_values = self.trainer.value_normalizer.normalize(next_values_t).cpu().numpy()
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        self.trainer.prep_training()
        return self.trainer.train(self.buffer)

    def save(self, episode):
        torch.save(
            self.trainer.policy.actor.state_dict(),
            self.save_dir / f"actor_{episode}.pt",
        )
        torch.save(
            self.trainer.policy.critic.state_dict(),
            self.save_dir / f"critic_{episode}.pt",
        )
        if self.trainer._use_valuenorm:
            torch.save(
                self.trainer.value_normalizer.state_dict(),
                self.save_dir / f"vnorm_{episode}.pt",
            )

    def restore(self):
        actor_state_dict = torch.load(
            str(Path(self.model_dir) / f"actor_{self.all_args.model_index}.pt"),
            map_location=self.device,
        )
        self.policy.actor.load_state_dict(actor_state_dict)
        critic_state_dict = torch.load(
            str(Path(self.model_dir) / f"critic_{self.all_args.model_index}.pt"),
            map_location=self.device,
        )
        self.policy.critic.load_state_dict(critic_state_dict)
        if self.trainer._use_valuenorm:
            vnorm_state_dict = torch.load(
                str(Path(self.model_dir) / f"vnorm_{self.all_args.model_index}.pt"),
                map_location=self.device,
            )
            self.trainer.value_normalizer.load_state_dict(vnorm_state_dict)

    def envs_close(self):
        if getattr(self, "envs", None) is not None:
            self.envs.close()
            self.envs = None

    def cleanup(self):
        self.envs_close()
        self.train_logger.close()
        self.context_logger.close()
