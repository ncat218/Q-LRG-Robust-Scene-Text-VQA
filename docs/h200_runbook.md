# UIT-HPC runbook

Use the project checkout on UIT-HPC and activate the Slurm module on the login node.

```bash
ssh UIT-HPC
cd /datastore/dungmt/textvqa/Tuan
git clone https://github.com/ncat218/Q-LRG-Robust-Scene-Text-VQA.git
cd Q-LRG-Robust-Scene-Text-VQA
. /etc/profile.d/modules.sh
module load slurm/slurm/25.05
```

Set `QLRG_UV` to an installed uv executable if uv is outside PATH. The following path was verified on UIT-HPC. Setup creates a separate Python 3.10.22 environment constrained by `requirements-hpc.lock` and records its resolved packages.

```bash
export QLRG_UV=/datastore/dungmt/textvqa/Tuan/monkeyocr_workspace/environment_and_models__phase1_verified_stack/uv_bootstrap/bin/uv
bash scripts/setup_hpc.sh
bash scripts/check_environment.sh
P="$PWD/.venv/bin/python"
"$P" -B -m unittest discover -s tests -v
```

For an existing verified Q-LRG environment, export `QLRG_ENV` with its absolute path and use `P="$QLRG_ENV/bin/python"`. Slurm inherits `QLRG_ENV` from the launching shell.

Dataset and offline model cache paths are in `configs/paths_hpc.yaml`. Override them with `QLRG_TEXTVQA`, `QLRG_TEXTOCR` and `QLRG_HF_HUB`, or set `QLRG_PATHS_CONFIG` for the Slurm workers. The model snapshot must exist in the Hugging Face hub cache at the pinned revision.

Prepare and validate separate manifests for smoke and the initial diagnostic evaluation.

```bash
"$P" -B -m qlrg.cli build-manifest --images 12 --output local/smoke12.json
"$P" -B -m qlrg.cli validate-dataset --manifest local/smoke12.json
"$P" -B -m qlrg.cli build-manifest --images 500 --output local/diag500.json
"$P" -B -m qlrg.cli validate-dataset --manifest local/diag500.json
"$P" -B -m qlrg.cli prepare-review --manifest local/diag500.json --limit 150 --output local/o3-review
```

Open `local/o3-review/review.html` in a local browser and inspect the candidate regions. Edit `review.jsonl` in a copy, updating each status to `accepted` or `rejected`. Correct the OCR word IDs where needed and give reasons for rejections. Finalized rows carry the image hash. Treat these files as evaluation data and keep them outside Git.

Render the D0 and S0 smoke commands for review. Add `--submit` after approving each command and run S0 after D0 passes.

```bash
"$P" scripts/submit.py d0 --mode smoke --manifest local/smoke12.json --run-dir runs/smoke12-v1
"$P" -B -m qlrg.cli verify-run --stage d0 --run-dir runs/smoke12-v1
"$P" scripts/submit.py s0 --mode smoke --manifest local/smoke12.json --run-dir runs/smoke12-v1
"$P" -B -m qlrg.cli verify-run --stage s0 --run-dir runs/smoke12-v1
```

Size the full diagnostic jobs from smoke latency and peak VRAM, then set the approved `QLRG_FULL_TIME`, `QLRG_FULL_CPUS` and `QLRG_FULL_MEM`. The launcher prints the full commands for review. Run and verify D0 before S0.

```bash
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
