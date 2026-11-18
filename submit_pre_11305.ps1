$ErrorActionPreference = 'Stop'
$worker = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'scripts/pre_11305.py')).TrimEnd()

$prefix = @'
set -eo pipefail
. /etc/profile.d/modules.sh
module load slurm/slurm/25.05
set -u
base=/datastore/dungmt/textvqa/Tuan
main="$base/Q-LRG-Robust-Scene-Text-VQA"
crop="$base/Q-LRG-crop"
source="$main/runs/s1-20261107T051836Z-wymhdtwk/s1"
env_dir="$main/.venv"
exec 9>"$crop/runs/.pre-11305-submit.lock"
flock -n 9
active=$(squeue -h -u "$(id -un)" -n qlrg-pre-11305,qlrg-merger-replay -o '%i %j %T %R')
if [[ -n "$active" ]]; then
    printf '%s\n' "$active"
    exit 0
fi
out=$(mktemp -d "$crop/runs/pre-11305-XXXXXXXX")
printf 'Run directory: %s\n' "$out"
mkdir "$out/logs" "$out/code"
git -C "$main" archive 5230c5a | tar -x -C "$out/code"
cd "$out/code"
cat > "$out/pre_11305.py" <<'PYTHON_WORKER'
'@

$suffix = @'
PYTHON_WORKER
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1
OMP_NUM_THREADS=1 "$env_dir/bin/python" -B "$out/pre_11305.py" "$out" "$source" --cpu-check
cat > "$out/pre-11305.sbatch" <<'SLURM_WORKER'
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export QLRG_ENV="$3"
bash scripts/check_environment.sh
exec "$QLRG_ENV/bin/python" -B "$1/pre_11305.py" "$1" "$2"
SLURM_WORKER
bash -n "$out/pre-11305.sbatch"
job=$(sbatch --parsable --chdir="$out/code" --job-name=qlrg-pre-11305 --account=uit --partition=normal --qos=mig2-35 --gres=gpu:2g.35gb:1 --cpus-per-task=8 --mem=48G --time=16:00:00 --output="$out/logs/%x-%j.out" --error="$out/logs/%x-%j.err" "$out/pre-11305.sbatch" "$out" "$source" "$env_dir")
printf '%s\n' "$job" | tee "$out/job-id.txt"
squeue -h -j "${job%%;*}" -o '%i %j %T %R' || true
'@

($prefix + [char]10 + $worker + [char]10 + $suffix + [char]10) |
    ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=8 -o ServerAliveCountMax=2 UIT-HPC 'tr -d "\r" | bash -s'
exit $LASTEXITCODE
