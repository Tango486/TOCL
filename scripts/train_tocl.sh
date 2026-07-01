#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python OurLight/scripts/train/train_tocl.py \
  --experiment_name tocl_source_only \
  --seed 10 \
  "$@"
