import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def get_config():
    parser = argparse.ArgumentParser(
        description="TOCL source-only MARL-TSC training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--algorithm_name", type=str, default="rmappo", choices=["rmappo"])
    parser.add_argument("--experiment_name", type=str, default="tocl")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--cuda_deterministic", type=str2bool, default=True)
    parser.add_argument("--n_training_threads", type=int, default=1)
    parser.add_argument("--n_rollout_threads", type=int, default=8)
    parser.add_argument("--num_env_steps", type=int, default=120000)
    parser.add_argument("--user_name", type=str, default="marl")
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--env_name", type=str, default="sumo")
    parser.add_argument("--scenario_name", type=str, default="generated_source")

    parser.add_argument("--episode_length", type=int, default=240)
    parser.add_argument("--num_actions", type=int, default=8)
    parser.add_argument("--obs_shape", type=int, default=56)
    parser.add_argument("--port_start", type=int, default=15900)
    parser.add_argument("--run_dir", type=str, default="runs")
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--generator_bounds_json", type=str, default=None)
    parser.add_argument("--generated_output_dir", type=str, default=None)

    parser.add_argument("--share_policy", type=str2bool, default=True)
    parser.add_argument("--use_centralized_V", type=str2bool, default=True)
    parser.add_argument("--use_obs_instead_of_state", type=str2bool, default=False)
    parser.add_argument("--stacked_frames", type=int, default=1)
    parser.add_argument("--use_stacked_frames", type=str2bool, default=False)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--layer_N", type=int, default=1)
    parser.add_argument("--use_ReLU", type=str2bool, default=True)
    parser.add_argument("--use_popart", type=str2bool, default=False)
    parser.add_argument("--use_valuenorm", type=str2bool, default=True)
    parser.add_argument("--use_feature_normalization", type=str2bool, default=True)
    parser.add_argument("--use_orthogonal", type=str2bool, default=True)
    parser.add_argument("--gain", type=float, default=0.01)
    parser.add_argument("--use_naive_recurrent_policy", type=str2bool, default=False)
    parser.add_argument("--use_recurrent_policy", type=str2bool, default=True)
    parser.add_argument("--recurrent_N", type=int, default=1)
    parser.add_argument("--data_chunk_length", type=int, default=10)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--critic_lr", type=float, default=5e-4)
    parser.add_argument("--opti_eps", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ppo_epoch", type=int, default=10)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--num_mini_batch", type=int, default=1)
    parser.add_argument("--entropy_coef", type=float, default=0.02)
    parser.add_argument("--value_loss_coef", type=float, default=1.0)
    parser.add_argument("--use_clipped_value_loss", type=str2bool, default=True)
    parser.add_argument("--use_max_grad_norm", type=str2bool, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--use_gae", type=str2bool, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--use_proper_time_limits", type=str2bool, default=False)
    parser.add_argument("--use_huber_loss", type=str2bool, default=True)
    parser.add_argument("--use_value_active_masks", type=str2bool, default=True)
    parser.add_argument("--use_policy_active_masks", type=str2bool, default=True)
    parser.add_argument("--huber_delta", type=float, default=10.0)
    parser.add_argument("--use_linear_lr_decay", type=str2bool, default=False)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--save_final_episode_interval", type=int, default=None)

    parser.add_argument("--use_pressure", type=str2bool, default=False)
    parser.add_argument("--use_gat", type=str2bool, default=False)
    parser.add_argument("--use_ours", type=str2bool, default=True)
    parser.add_argument("--use_trans_critic", type=str2bool, default=False)
    parser.add_argument("--use_K", type=int, default=0)
    parser.add_argument("--use_kl", type=str2bool, default=False)
    parser.add_argument("--use_frap_in_trans", type=str2bool, default=False)
    parser.add_argument("--use_3cons", type=str2bool, default=False)
    parser.add_argument("--use_sym_loss", type=str2bool, default=True)
    parser.add_argument("--part_mask", type=str2bool, default=True)
    parser.add_argument("--trans_hidden", type=int, default=64)
    parser.add_argument("--use_trans_hidden", type=str2bool, default=True)
    parser.add_argument("--trans_heads", type=int, default=8)
    parser.add_argument("--trans_layers", type=int, default=2)
    parser.add_argument(
        "--state_key",
        nargs="+",
        default=["current_phase", "car_num", "queue_length", "occupancy", "flow", "stop_car_num", "pressure"],
    )
    parser.add_argument("--epsilon_decay", type=str2bool, default=True)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--min_epsilon", type=float, default=1.0)
    parser.add_argument("--anneal_steps", type=int, default=500000)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--model_index", type=str, default=None)
    parser.add_argument("--load_model", type=str2bool, default=False)
    parser.add_argument("--use_render", type=str2bool, default=False)
    parser.add_argument("--n_render_rollout_threads", type=int, default=1)

    parser.add_argument("--tocl_phase_episodes", type=int, default=5)
    parser.add_argument("--tocl_buffer_size", type=int, default=4000)
    parser.add_argument("--tocl_replay_prob", type=float, default=0.65)
    parser.add_argument("--tocl_min_seen_before_replay", type=int, default=10)
    parser.add_argument("--tocl_score_transform", type=str, default="rank", choices=["constant", "max", "rank", "softmax", "power"])
    parser.add_argument("--tocl_temperature", type=float, default=0.5)
    parser.add_argument("--tocl_staleness_coef", type=float, default=0.25)
    parser.add_argument("--tocl_staleness_temperature", type=float, default=1.0)
    parser.add_argument("--tocl_ema_alpha", type=float, default=0.5)
    parser.add_argument("--tocl_score_clip", type=float, default=5.0)
    parser.add_argument("--tocl_new_candidate_count", type=int, default=8)
    parser.add_argument("--tocl_static_select_prob", type=float, default=0.25)
    parser.add_argument("--tocl_static_temperature", type=float, default=0.35)
    parser.add_argument("--tocl_scale_window", type=int, default=128)

    parser.add_argument("--tocl_static_flow_weight", type=float, default=0.30)
    parser.add_argument("--tocl_static_spatial_weight", type=float, default=0.20)
    parser.add_argument("--tocl_static_temporal_weight", type=float, default=0.20)
    parser.add_argument("--tocl_static_left_weight", type=float, default=0.15)
    parser.add_argument("--tocl_static_del_weight", type=float, default=0.10)
    parser.add_argument("--tocl_static_size_weight", type=float, default=0.05)
    parser.add_argument("--tocl_delay_weight", type=float, default=0.26)
    parser.add_argument("--tocl_wait_weight", type=float, default=0.22)
    parser.add_argument("--tocl_queue_weight", type=float, default=0.22)
    parser.add_argument("--tocl_pressure_weight", type=float, default=0.16)
    parser.add_argument("--tocl_td_weight", type=float, default=0.04)
    parser.add_argument("--tocl_static_weight", type=float, default=0.10)

    return parser
