# UIT-HPC runbook

Run these commands from the UIT-HPC login node.

```bash
cd /datastore/dungmt/textvqa/Tuan
git clone https://github.com/ncat218/Q-LRG-Robust-Scene-Text-VQA.git
cd Q-LRG-Robust-Scene-Text-VQA
python3 run_all.py
```

The launcher loads Slurm if needed, saves the generated `smoke.sbatch`, and submits one job with one H200 35 GB MIG slice, four CPUs, 32 GB RAM and a 30-minute limit. After allocation, the job checks or creates the pinned environment, builds a 12-image manifest, runs D0, and verifies its artifacts before running S0. If D0 fails, the job stops at D0.

The command prints the run directory and the Slurm Job ID. Each invocation creates a new run under `runs/smoke12-*/`, with logs in its `logs/` directory. Use `python3 run_all.py --dry-run` to generate the script without setup or submission.

Setup uses `QLRG_UV` when set, then checks PATH and the shared UIT-HPC uv executable. The first setup downloads the pinned dependencies inside the job and records progress in the Slurm logs. If compute-node downloads fail, prepare the environment once from the login node with `bash scripts/setup_hpc.sh` and submit again. For an existing verified Q-LRG environment, export `QLRG_ENV` with its absolute path. Slurm inherits it from the launching shell.

Dataset and offline model cache paths are in `configs/paths_hpc.yaml`. Override them with `QLRG_TEXTVQA`, `QLRG_TEXTOCR` and `QLRG_HF_HUB`, or set `QLRG_PATHS_CONFIG` for the smoke pipeline. The model snapshot must exist in the Hugging Face hub cache at the pinned revision.

After smoke passes, prepare the initial 500-image diagnostic manifest and grounding review.

```bash
P="${QLRG_ENV:-$PWD/.venv}/bin/python"
"$P" -B -m qlrg.cli build-manifest --images 500 --output local/diag500.json
"$P" -B -m qlrg.cli validate-dataset --manifest local/diag500.json
"$P" -B -m qlrg.cli prepare-review --manifest local/diag500.json --limit 150 --output local/o3-review
```

Open `local/o3-review/review.html` in a local browser and inspect the candidate regions. Edit `review.jsonl` in a copy, updating each status to `accepted` or `rejected`. Correct the OCR word IDs where needed and give reasons for rejections. Finalized rows carry the image hash. Treat these files as evaluation data and keep them outside Git.

Size the full diagnostic jobs from smoke latency and peak VRAM, then set the approved `QLRG_FULL_TIME`, `QLRG_FULL_CPUS` and `QLRG_FULL_MEM`. The launcher prints the full commands for review. Run and verify D0 before S0.

```bash
. /etc/profile.d/modules.sh
module load slurm/slurm/25.05
: "${QLRG_FULL_TIME:?Set the approved full walltime}"
: "${QLRG_FULL_CPUS:?Set the approved full CPU count}"
: "${QLRG_FULL_MEM:?Set the approved full RAM request}"

"$P" scripts/submit.py d0 --mode full --manifest local/diag500.json --run-dir runs/diag500-v1 \
  --cpus "$QLRG_FULL_CPUS" --memory "$QLRG_FULL_MEM" --time "$QLRG_FULL_TIME"
"$P" -B -m qlrg.cli verify-run --stage d0 --run-dir runs/diag500-v1
"$P" scripts/submit.py s0 --mode full --manifest local/diag500.json --run-dir runs/diag500-v1 \
  --review local/o3-review/review.approved.jsonl \
  --cpus "$QLRG_FULL_CPUS" --memory "$QLRG_FULL_MEM" --time "$QLRG_FULL_TIME"
"$P" -B -m qlrg.cli verify-run --stage s0 --run-dir runs/diag500-v1
cat runs/diag500-v1/s0/decision.md
```

Slurm stdout and stderr go under the run's `logs/` directory with the job ID. Use the same manifest, options and run directory to resume. A changed experiment gets a new directory. S0 writes scored predictions, the paired summaries, coverage, PNG overlays and a decision memo. Set `QLRG_FEATURE_CACHE` to a dedicated writable cache directory to reuse validated feature pairs across compatible runs.
