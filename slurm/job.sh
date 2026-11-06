#!/usr/bin/env bash
set -euo pipefail
stage="$1"
shift
: "${SLURM_JOB_ID:?Run this script in a Slurm GPU allocation}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi
bash "scripts/run_$stage.sh" "$@"
