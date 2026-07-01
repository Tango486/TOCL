# TOCL

Traffic Operation-aware Curriculum Learning (TOCL) for source only MARL based traffic signal control training with generated SUMO scenarios.

## Installation

Install SUMO and make sure the following commands are available on `PATH`:

```bash
sumo
netconvert
netgenerate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The released target SUMO scenarios are packaged as compressed archives. Extract them before using the released datasets:

```bash
tar -xzf data/archives/gentsc_stress400_sumo.tar.gz -C data
tar -xzf data/archives/gentsc_broad240_sumo.tar.gz -C data
```

This creates `data/gentsc_stress400/` and `data/gentsc_broad240/`.

## Usage

Run a default TOCL training job:

```bash
bash scripts/train_tocl.sh
```

Common training options:

```bash
bash scripts/train_tocl.sh --seed 11
bash scripts/train_tocl.sh --num_env_steps 240000 --n_rollout_threads 8
bash scripts/train_tocl.sh --generated_output_dir generated/source_run_seed11
```

Generated source SUMO files are written to:

```text
OurLight/envs/sumo_files_marl/generated_source_runs/
```

Training outputs are written to:

```text
runs/
```
