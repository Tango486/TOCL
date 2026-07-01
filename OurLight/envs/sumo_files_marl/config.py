#!/usr/bin/env python3

name = 'agent'
config = {
    "episode": {
        "num_train_rollouts": 100000,
        "rollout_length": 240,

        "warmup_ep_steps": 0,
        "test_num_eps": 50
    },
    "agent": {
        "agent_type": "ppo",
        "single_agent": True
    },
    "ppo": {
        "gae_tau": 0.85,
        "entropy_weight": 0.01,
        "minibatch_size": 128,
        "optimization_epochs": 4,
        "ppo_ratio_clip": 0.2,
        "discount": 0.99,
        "learning_rate": 1e-4,
        "clip_grads": True,
        "gradient_clip": 2.0,
        "value_loss_coef": 1.0,
        'target_kl': None
    },
    "environment": {
        "num_actions": 8,
        "obs_shape": 56,



        "action_type": "select_phase",

        "gui": False,
        "yellow_duration": 5,
        "iter_duration": 10,
        "episode_length_time": 3600,
        "is_record": False,

        'output_path': None,
        "name": name,






















        'port_start': 19900,
        "sumocfg_files": [






            'sumo_files_marl/scenarios/resco_envs/grid4x4/grid4x4.sumocfg'





        ],

        "state_key": ['current_phase', 'car_num', 'queue_length', "occupancy", 'flow', 'stop_car_num', 'pressure'],

        'reward_type': ['queue_len', 'wait_time', 'delay_time', 'pressure', ]
    },
    "model_save": {
        "frequency": 200,
        "path": "envs/sumo_files_marl/tsc/{}".format(name)
    },
    "parallel": {
        "num_workers": 1
    }
}
