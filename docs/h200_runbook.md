# UIT-HPC runbook

Run these commands from the UIT-HPC login node.

```bash
cd /datastore/dungmt/textvqa/Tuan
git clone https://github.com/ncat218/Q-LRG-Robust-Scene-Text-VQA.git
cd Q-LRG-Robust-Scene-Text-VQA
python3 run_all.py --mode full
```

The launcher saves `full.sbatch` and submits one job with one H200 35 GB MIG slice, four CPUs, 32 GB RAM and a three-hour walltime limit. After allocation, the job checks or creates the pinned environment, builds the 500-image manifest, runs D0, and verifies its artifacts before running S0. If D0 fails, the job stops at D0.

The command prints the run directory and the Slurm Job ID. Full runs go under `runs/full500-*/`, with logs in `logs/`, the S0 memo at `s0/decision.md`, and the interactive case report at `s0/review.html`. Add `--dry-run` to generate the script without setup or submission. `python3 run_all.py --mode smoke` runs the 12-image check.

Download `s0/review.html` and open it in a browser. It contains the per-question predictions and the masks saved by the run. The comparison only uses questions scored in both selected arms. For a completed run created before this feature, generate the report on CPU using `"${QLRG_ENV:-$PWD/.venv}/bin/python" -m qlrg.cli build-review --run-dir <run-directory>`. Pass `--manifest` if it lives outside the run directory.

Setup uses `QLRG_UV` when set, then checks PATH and the shared UIT-HPC uv executable. The first setup downloads the pinned dependencies inside the job and records progress in the Slurm logs. If compute-node downloads fail, prepare the environment once from the login node with `bash scripts/setup_hpc.sh` and submit again. For an existing verified Q-LRG environment, export `QLRG_ENV` with its absolute path. Slurm inherits it from the launching shell.

Dataset and offline model cache paths are in `configs/paths_hpc.yaml`. Override them with `QLRG_TEXTVQA`, `QLRG_TEXTOCR` and `QLRG_HF_HUB`, or set `QLRG_PATHS_CONFIG` for the pipeline. The model snapshot must exist in the Hugging Face hub cache at the pinned revision.

The full pipeline reports O0–O2 on all 500 images. It runs O3–O5 on up to 150 unambiguous OCR candidates, one per image, and records the provisional grounding policy in the report. For a separate human-reviewed run, prepare a grounding review from the same deterministic manifest.

```bash
P="${QLRG_ENV:-$PWD/.venv}/bin/python"
"$P" -B -m qlrg.cli build-manifest --images 500 --output local/diag500.json
"$P" -B -m qlrg.cli validate-dataset --manifest local/diag500.json
"$P" -B -m qlrg.cli prepare-review --manifest local/diag500.json --limit 150 --output local/o3-review
```

Open `local/o3-review/review.html` in a local browser and inspect the candidate regions. Edit `review.jsonl` in a copy, updating each status to `accepted` or `rejected`. Correct the OCR word IDs where needed and give reasons for rejections. Finalized rows carry the image hash. Treat these files as evaluation data and keep them outside Git.

Pass the finalized JSONL to the full pipeline. For separate stage jobs with custom resource requests, set `QLRG_FULL_TIME`, `QLRG_FULL_CPUS` and `QLRG_FULL_MEM` and run D0 before S0.

```bash
python3 run_all.py --mode full --review local/o3-review/review.approved.jsonl
```

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

## Planned S0 mask sensitivity run

Use the same checkout, fixed seed, paths, manifest and grounding choices for both arms of the sensitivity check. If using reviewed grounding, pass the same `--review` JSONL to both commands. Generate the scripts with `--dry-run` first.

```bash
mkdir -p local
sed 's/^mask_rule: overlap$/mask_rule: center/' configs/s0_oracle.yaml > local/s0-center.yaml
python3 run_all.py --mode full --s0-config configs/s0_oracle.yaml --dry-run
python3 run_all.py --mode full --s0-config local/s0-center.yaml --dry-run
```

For the scheduled comparison, run the two commands without `--dry-run` from the same checkout. Each launch snapshots the supplied YAML in its run directory. Verify both manifests have the same SHA-256 before joining predictions. Compare per-question scores and token counts for O2 (all TextOCR) across the two rules, with a paired image-cluster interval. For O3 (annotated answer region), report accuracy on the intersection of nonempty masks and report each rule's coverage separately. Inspect small answer words and cases where the token-center rule drops the mask.

## S1 pilot from a verified D0/S0 run

From the existing checkout on the UIT-HPC login node, update the code and submit one job.

```bash
cd /datastore/dungmt/textvqa/Tuan/Q-LRG-Robust-Scene-Text-VQA
git pull --ff-only
python3 run_all.py --mode s1 --from-run runs/full500-20261106T120210Z-6z0yzfi_
```

The launcher snapshots the S1 YAML, generates `s1.sbatch`, and submits one H200 35 GB MIG slice with 8 CPUs, 48 GB of host RAM and a 6-hour walltime limit. Actual training time and peak GPU memory need validation on UIT-HPC. The compute job checks the environment, verifies the source D0/S0 checksums and repeats the identity gate. It checks BF16 gradients through the restorer and LoRA before training all four adapters. The original D0/S0 run provides evaluation provenance. The job regenerates A0 predictions with the same feature injection path as the S1 comparisons.

S1 uses one epoch on 2,048 deterministically selected `train_inner` images. The four trained arms use the same examples in the same order, with 20% of images CLEAN and the rest LR4. All evaluate on both conditions for the same questions on the 500 held-out images in the anchor. S1a learns to match clean features. S1b combines answer loss, feature matching and a residual penalty. A1 is an answer-supervised token-wise visual adapter. A2 is an answer-supervised rank-4 LoRA on the language layers' Q/V projections. A0 keeps the frozen Qwen features. Use `--s1-config PATH` to provide a complete alternative YAML before launching a new run.

Use the printed run path and job ID. Set `run` and `job_id` to those values.

```bash
sacct -j "$job_id" --format=JobID,JobName,State,ExitCode,Elapsed
"${QLRG_ENV:-$PWD/.venv}/bin/python" -B -m qlrg.cli verify-run --stage s1 --run-dir "$run"
```

Download `s1/review.html` for LR4 and `s1/review-clean.html` for CLEAN. Open them in a browser to filter per-question predictions and inspect residual heatmaps for the first eight evaluation images in the visual variants. Each heatmap scales colors within that image and arm. `s1/summary.csv` reports paired 95% intervals, including the numeric/date slice. `s1/summary.json` also records accuracy, parameter counts, training history and compute measurements. `s1/feature_regressions.json` lists cases where feature MSE improved but answer score fell. `s1/baseline_audit.json` compares the archived native and injected predictions per question.

If the job stops, resubmit the same run directory with the original checkout and environment. The job restores each completed optimizer checkpoint and resumes evaluation from completed image shards.

```bash
python3 run_all.py --mode s1 --resume "$run"
```
## Expanded S1 and A0 audit

Apply the [predeclared audit decision rules](audit_decision_rules.md) before choosing the next allocation.

All comparisons use the project's own frozen Qwen control. Train on train_inner, select hyperparameters on val_inner, and report the locked models on labeled official validation. The official test set is optional final submission only. Keep validation labels out of training. The first S1 pilot already selected training images without TextOCR. Its 500-image evaluation anchor and the old common preflight depended on TextOCR.

The next job repeats both native A0 and cached-injection A0 twice over the full 500-image anchor. It compares token IDs and scores within each path and between paths. Native versus injected mismatches trigger fresh-feature and prefill-logit diagnostics. These are two full passes inside one job. Cross-job variability requires a second audit job with the same settings.

```bash
python3 run_all.py --mode audit \
  --from-run runs/full500-20261106T120210Z-6z0yzfi_ \
  --pilot-run runs/s1-20261106T161413Z-lm0jjs3z
```

The audit job also verifies the pilot's actual training manifest against its saved hash and the evaluation manifest, checking disjoint image IDs, question IDs and image SHA-256 values. It writes `audit/pilot_train_manifest.json` and `audit/pilot_split_audit.json`. The summary separates current repeatability, current path equivalence, and historical D0/pilot differences. Read the per-question records before attributing any difference to numeric nondeterminism.

After the GPU audit, the job prepares three independent OCR-free manifests under `datasets/`. Image files are resolved under the configured TextVQA root. The builder checks source annotation hashes, image bytes, answer counts and overlap across train_inner, val_inner and official_val. Counts come from the actual data. Dataset preparation stops if required validation images are unavailable. The job packages the audit and complete manifests into `review.tar.gz` in its run directory.

Expanded training requires the audit stage directory, for example `runs/audit-…/audit`, containing `COMPLETE.json`. The worker rechecks its artifacts and environment and assesses the prediction rows against the observed within-path repeatability. The recorded audit with source hash `d22a29cf…` can be reused when the inference path fingerprint matches. Its original exact-token result remains in the archived report. The supplied expanded configuration is a candidate with all train_inner images, two epochs, 40% clean images and a 0.1 additional clean-residual penalty. It includes six trained variants plus A0. Tune these choices on val_inner. It is not a convergence claim.

The following is a training template for a later approved allocation. Replace the audit directory with the actual printed run path.

```bash
audit_run=runs/audit-<timestamp>-<suffix>
export QLRG_FEATURE_CACHE="$PWD/runs/shared-features"
mkdir -p "$QLRG_FEATURE_CACHE"
df -h "$QLRG_FEATURE_CACHE"
python3 run_all.py --mode s1 \
  --data-dir "$audit_run/datasets" --baseline-audit "$audit_run/audit" \
  --s1-config configs/s1_expanded.json --time 2-00:00:00
```

The command above uses the default learning rate of 2e-4 and seed 42 for the full run. For a shorter sweep first, use the following commands with the same audit and cache variables.

```bash
python3 scripts/prepare_s1_study.py --short-sweep --output local/s1-lr-pilot
python3 run_all.py --mode s1 \
  --data-dir "$audit_run/datasets" --baseline-audit "$audit_run/audit" \
  --s1-config local/s1-lr-pilot/seed42-lr0.0002.json
```

The generator prepares 1e-4, 2e-4 and 5e-4 with 4,000 training images and one epoch. Each job trains S1b, A1 and A2 on the same data, and evaluates them with A0 on val_inner. Use the other generated configs when running those rates. Reserve time for evaluation and cache setup in addition to the estimated training time. Keep A0 from the same job for comparisons.

After interruption, use `python3 run_all.py --mode s1 --resume "$run" --time 2-00:00:00` to continue from the last saved checkpoint in a large run.

Expanded S1 saves once per 1,000 optimizer steps (4,000 questions at the default accumulation) and at the end of each arm. Each arm atomically replaces its single `checkpoint.pt`. Keep the configuration copied into the run directory when resuming, including the checkpoint interval of an older run.

For a learning-rate sweep and repeated seeds, generate candidate JSON configs from `configs/s1_expanded.json` using `python3 scripts/prepare_s1_study.py --output local/s1-study`. The default grid is three seeds (42, 43, 44) and three learning rates (5e-5, 1e-4, 2e-4). This prepares nine candidate configurations. Schedule the budget after the A0 audit and data counts are reviewed. Give every arm the same candidate grid, data, example count and optimizer-step budget. Evaluate candidates on val_inner, record the selection rule, then use at least three distinct seeds for the selected regime. Record each seed separately before aggregating image-paired effects and seed variability. The expanded run reports A1clean and A2clean, which use the same question and answer schedule as the other arms but CLEAN inputs throughout.

The feature cache stores both CLEAN and LR4 tensors. At full scale it may require hundreds of GB depending on token grids. The training preflight records a conservative storage bound and filesystem-wide free space in `s1/storage.json`. Check the user or project quota on the chosen mount separately. Matching seeds and learning rates can share cached features under an unchanged code/model/preprocessing signature. Walltime estimates must account for initial cache construction and additional clean-only controls.

If the feature cache exceeds quota, point `QLRG_FEATURE_CACHE` to a writable location with sufficient confirmed quota before resubmitting. A dedicated inactive feature cache can be removed and rebuilt. Keep the audit's `audit/` and `datasets/` directories and the interrupted run's S1 state and checkpoints for resume. The audit `review.tar.gz` is a downloadable copy and can be removed after verifying the original artifacts or an external copy.

To estimate remaining writes against a known quota budget, run `"${QLRG_ENV:-$PWD/.venv}/bin/python" -B -m scripts.check_s1_cache --run "$run" --budget-gb 30` from the repository. It counts matching feature keys for the original manifests and estimates the bytes for missing entries. Use the actual quota headroom for the budget. The check reserves 3 GiB for other artifacts and exits nonzero when the estimate exceeds the budget.

Once hyperparameters and checkpoints are locked, official validation is a separate evaluation job.

```bash
python3 run_all.py --mode s1-eval --from-run runs/<selected-expanded-S1-run>
```

It verifies the saved checkpoints and manifests, performs no optimizer steps and writes `s1_eval/` results. Both development and official-validation reports include `degradation_interaction.csv`, with image-cluster intervals for the paired four-score quantity (target LR4 − baseline LR4) − (target CLEAN − baseline CLEAN). Preserve paired image sampling across both conditions.

Keep S2/S3 on hold while resolving A0 differences and testing the expanded restoration versus A1/A2. Require clear paired evidence across seeds before advancing under this protocol. A negative result supports the conclusion that the observed gains mainly reflect adaptation within the evaluated regime. Verify any claim about the base Qwen model's TextVQA pretraining exposure separately.
