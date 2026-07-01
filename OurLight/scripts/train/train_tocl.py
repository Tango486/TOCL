#!/usr/bin/env python3
"""Run source only TOCL training on generated SUMO contexts."""

from pathlib import Path
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from OurLight.config import get_config
from OurLight.envs.sumo_files_marl.config import config as sumo_config
from OurLight.runner.shared.sumo_runner import SUMORunner


def make_run_dir(root, experiment_name, seed, suffix=None):
    root = Path(root)
    base = root / experiment_name / f"seed_{seed}"
    if suffix:
        base = Path(f"{base}_{suffix}")
    run_dir = base
    index = 1
    while run_dir.exists():
        run_dir = Path(f"{base}_run{index}")
        index += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def configure_sumo_env(args):
    env_cfg = sumo_config["environment"]
    env_cfg["num_actions"] = int(args.num_actions)
    env_cfg["obs_shape"] = int(args.obs_shape)
    env_cfg["state_key"] = list(args.state_key)
    env_cfg["gui"] = bool(args.use_render)
    env_cfg["episode_length_time"] = int(args.episode_length) * (
        int(env_cfg.get("yellow_duration", 5)) + int(env_cfg.get("iter_duration", 10))
    )
    env_cfg["is_record"] = False
    env_cfg["output_path"] = None


def set_seeds(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main():
    parser = get_config()
    args = parser.parse_args()

    args.algorithm_name = "rmappo"
    args.env_name = "sumo"
    args.use_centralized_V = True
    args.use_recurrent_policy = True
    args.use_naive_recurrent_policy = False
    args.use_ours = True
    args.use_trans_hidden = True

    set_seeds(args.seed, args.cuda_deterministic)
    configure_sumo_env(args)

    use_cuda = bool(args.cuda) and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.set_num_threads(int(args.n_training_threads))

    run_dir = make_run_dir(PROJECT_ROOT / args.run_dir, args.experiment_name, args.seed, args.suffix)
    args.run_dir = str(run_dir)
    with (run_dir / "args.txt").open("w") as f:
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")

    runner = SUMORunner({
        "all_args": args,
        "device": device,
        "run_dir": run_dir,
    })
    try:
        runner.run()
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
