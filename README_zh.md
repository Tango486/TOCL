# TOCL

[English](README.md) | 中文

Traffic Operation-aware Curriculum Learning (TOCL) 用于基于生成式 SUMO 场景的 MARL 交通信号控制 source only 训练。

## 环境安装

请先安装 SUMO，并确保以下命令位于 `PATH` 中：

```bash
sumo
netconvert
netgenerate
```

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

## 数据集准备

发布的 target SUMO 场景已压缩打包。使用发布数据集前，请先解压：

```bash
tar -xzf data/archives/gentsc_stress400_sumo.tar.gz -C data
tar -xzf data/archives/gentsc_broad240_sumo.tar.gz -C data
```

解压后会得到 `data/gentsc_stress400/` 和 `data/gentsc_broad240/`。

## 使用方法

运行默认 TOCL 训练：

```bash
bash scripts/train_tocl.sh
```

常用训练参数：

```bash
bash scripts/train_tocl.sh --seed 11
bash scripts/train_tocl.sh --num_env_steps 240000 --n_rollout_threads 8
bash scripts/train_tocl.sh --generated_output_dir generated/source_run_seed11
```

训练过程中生成的 source SUMO 文件会写入：

```text
OurLight/envs/sumo_files_marl/generated_source_runs/
```

训练输出会写入：

```text
runs/
```
