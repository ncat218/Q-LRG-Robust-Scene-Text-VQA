#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
exec bash slurm/job.sh d0 "$@"
