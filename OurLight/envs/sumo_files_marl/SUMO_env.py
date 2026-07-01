import random


from OurLight.envs.sumo_files_marl.env.sim_env import TSCSimulator
from OurLight.envs.sumo_files_marl.config import config

from gym import spaces
import numpy as np

import torch
import copy
import os
import re




class SUMOEnv(object):
    '''Wrapper to make Google Research Football environment compatible'''

    def __init__(self, args, rank, sumo_cfg, is_record=False, thread_id=None):
        self.args = args
        id = args.seed + np.random.randint(0, 2023) + rank
        self.set_seed(id)


        env_config = config['environment']


        sumo_cfg = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/' + sumo_cfg

        env_config = copy.deepcopy(env_config)
        env_config['sumocfg_file'] = sumo_cfg
        target_match = re.search(r"rollouttarget_(\d+)_", sumo_cfg)
        eval_seed_offset = int(target_match.group(1)) if is_record and target_match else rank
        env_config['seed'] = int(args.seed * 1000 + eval_seed_offset)

        port = args.port_start + id

        print('----port--', port, '----sumo_cfg--', sumo_cfg)

        if is_record:
            env_config["is_record"] = True



            if "my_random_grids" in sumo_cfg:
                env_config["output_path"] = os.path.dirname(sumo_cfg)
            else:
                env_config["output_path"] = os.path.join(str(args.run_dir), "tripinfo")
                os.makedirs(env_config["output_path"], exist_ok=True)
            env_config["thread_id"] = thread_id
        else:
            env_config["is_record"] = False
            env_config["thread_id"] = None
        self.env = TSCSimulator(env_config, port)

        self.unava_phase_index = []
        for i in self.env.all_tls:
            self.unava_phase_index.append(self.env._crosses[i].unava_index)

        self.num_agents = len(self.unava_phase_index)
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        for idx in range(self.num_agents):
            self.action_space.append(spaces.Discrete(n=env_config['num_actions']))
            self.share_observation_space.append(spaces.Box(-float('inf'), float('inf'), [env_config['obs_shape']*self.num_agents], dtype=np.float32))
            self.observation_space.append(spaces.Box(-float('inf'), float('inf'), [env_config['obs_shape']], dtype=np.float32))


    def get_unava_phase_index(self):
        return np.array(self.unava_phase_index)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    def get_reward(self, reward, all_tls):
        ans = []
        for i in all_tls:
            ans.append(sum(reward[i].values()))
        return np.array(ans)

    def batch(self, env_output, use_keys, all_tl):
        """Transform agent-wise env_output to batch format."""
        if all_tl == ['gym_test']:
            return torch.tensor([env_output])
        obs_batch = {}
        for i in use_keys+['mask', 'neighbor_index', 'neighbor_dis']:
            obs_batch[i] = []
        for agent_id in all_tl:
            out = env_output[agent_id]
            tmp_dict = {k: np.zeros(8) for k in use_keys}
            state, mask, neight_msg = out
            for i in range(len(state)):
                for s in use_keys:
                    tmp_dict[s][i] = state[i].get(s, 0)
            for k in use_keys:
                obs_batch[k].append(tmp_dict[k])
            obs_batch['mask'].append(mask)
            obs_batch['neighbor_index'].append(neight_msg[0][0])
            obs_batch['neighbor_dis'].append(neight_msg[0][1])

        for key, val in obs_batch.items():
            if key not in ['current_phase', 'mask', 'neighbor_index']:
                obs_batch[key] = torch.FloatTensor(np.array(val))
            else:
                obs_batch[key] = torch.LongTensor(np.array(val))



        if self.args.use_pressure:
            obs_batch['pressure'] = obs_batch.pop('pressure')
        else:
            obs_batch.pop('pressure')

        if self.args.use_gat:
            obs_batch['neighbor_index'] = obs_batch.pop('neighbor_index')
            obs_batch['neighbor_dis'] = obs_batch.pop('neighbor_dis')
        else:
            obs_batch.pop('neighbor_index')
            obs_batch.pop('neighbor_dis')

        self.obs_keys = list(obs_batch.keys())
        obs_values = np.hstack(list(obs_batch.values()))


        return obs_values
    def reset(self):
        obs = self.env.reset()
        obs_values = self.batch(obs, config['environment']['state_key'], self.env.all_tls)
        obs_values = self._obs_wrapper(obs_values)
        return obs_values

    def step(self, action):
        tl_action_select = {}
        for tl_index in range(len(self.env.all_tls)):
            tl_action_select[self.env.all_tls[tl_index]] = (self.env._crosses[self.env.all_tls[tl_index]].green_phases)[action[tl_index]]
        obs, reward, done, info = self.env.step(tl_action_select)
        obs = self.batch(obs, config['environment']['state_key'], self.env.all_tls)
        obs = self._obs_wrapper(obs)
        reward = self.get_reward(reward, self.env.all_tls)
        reward = reward.reshape(self.num_agents, 1)






        done = np.array([done] * self.num_agents)

        return obs, reward, done, info

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def close(self):
        self.env.terminate()

    def _obs_wrapper(self, obs):
        return obs
        if self.num_agents == 1:
            return obs[np.newaxis, :]
        else:
            return obs
