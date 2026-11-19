# Full/crop S1b convergence and the locked 11305 factorial

Requested 2026-11-20. These are two independent workflows. All original runs,
checkpoints, data and Danh's directories remain unchanged.

## 1. Old checkpoint evaluation: exactly 32 arms

Reader: A0 (unmodified frozen Qwen2.5-VL-3B) and A2 (frozen LLM Q/V LoRA from 11305).
Restorer: frozen S1b from 11305 in every restoration arm. Region selector:
Vision-RL² 7B, replay the existing per-query LR4 proposals and 20% padding.
No answer labels or CLEAN images select the region at inference.

For each reader and each budget, evaluate these eight arms:

| Input | Restore full | Restore crop |
|---|---|---|
| Full | no | — |
| Full | yes | — |
| Crop | — | no |
| Crop | — | yes |
| Full + crop | no | no |
| Full + crop | yes | no |
| Full + crop | no | yes |
| Full + crop | yes | yes |

Fair budget: 576 reader visual tokens total; fusion uses 288+288.
Native budget: the original reader max_pixels=786432 **per view**; report actual
tokens, rather than claiming equal computation. All 32 arms use the same 5,000
official validation questions / 3,166 images, LR4, same query and decoding.
No-ROI cases remain included, with an explicit full-image fallback and effective
restoration flags. Crop-only S1b fallback restores its full-image substitute;
fusion with no crop applies only the requested full flag.

Artifacts: `accuracy.csv`, `paired.csv` (95% paired image-cluster bootstrap,
10,000 resamples), `error_cases.csv`, `predictions.jsonl`, `flops.csv`, `run.json`,
`COMPLETE.json`, per-image resume shards. FLOPs are registered reader operators
including vision/restorer with eight forced decode tokens, on three images;
unsupported kernels and the selector are excluded. Latency in prediction rows
reuses frozen vision/restorer costs and archived selector timings and is labeled
as such; it is not a fresh end-to-end measurement. Record tokens and peak CUDA
memory as well. No training or checkpoint replacement occurs in this job.

## 2. New S1b full and S1b crop training

Same residual architecture, hidden=224, two spatial blocks, MSE-to-CLEAN + answer
CE + residual penalties as the actual 11305 config. Frozen Qwen, BF16, native
resolution protocol, effective batch/accumulation=4, seed 42, AdamW LR=2e-4,
clean fraction 0.4. Both views start from the same fresh zero-output initialization;
this is not a continuation of 11305 or joint A2/S1b training. No new blur protocol.

Crop proposals are generated once using frozen Vision-RL² on LR4 and the query.
The exact same padded box cuts LR4 student and CLEAN teacher, then their grids
are checked for alignment. The selector never sees answer strings. Include
no-ROI full-image fallbacks and all training questions; no TextOCR filter.
No-ROI frequency/coverage must be reported when interpreting crop training.

### Selection

- Full train_inner: 19,758 images / 31,129 questions (90% of official train).
- val_inner: 2,195 images / 3,473 questions; evaluate CLEAN and LR4 every 0.5 epoch.
- Maximum 8 epochs, minimum 2 before stopping; patience 4 validation checks,
  meaningful-progress threshold 0.1 **percentage points**, no loss-based stopping.
- Save highest LR4 validation checkpoint satisfying CLEAN >= same-view A0 CLEAN −1 pp.
  Best-score selection and the min-delta patience reference are separate.
- A zero-initialized baseline is an eligible fallback. Report `early_stopping` vs
  `epoch_cap`; neither is a proof of global convergence. If the cap is reached
  while performance is improving, report that the budget was exhausted.
- Retain only `best.pt` + `latest.pt` in each view's selection directory; atomic
  replacement, fixed filenames, no epoch checkpoint archive, no disk feature cache.

### Full official-train refit

Only after selecting the duration, merge train_inner + val_inner: **21,953 images /
34,602 questions**, not official validation. Start afresh with the same seed and
train for the selected fractional number of epochs, scaled to the full train size.
No early stopping on former val_inner after it becomes training data. Retain one
`latest.pt` (also final) for each view's refit. Thus at most **three persistent
checkpoint files per view**, with an additional temporary file during atomic save.
If no trained checkpoint beats the eligible initialization, skip refit and report it.

The 5,000 official-validation questions are not evaluated by this training job.
They have already been examined in earlier project experiments, so subsequent
official-val results are not a new untouched holdout. A separate locked evaluation
will compare the new crop/full checkpoints after training, as requested.

## Execution

Run from a pinned independent snapshot of crop-branch. The existing environment
and model/dataset caches are reused read-only; PYTHONPATH points at the snapshot.

```bash
python3 run_s1_crop_study.py --mode crop_matrix --from-run "$OLD_CROP_RUN"
python3 run_s1_crop_study.py --mode s1_convergence --from-run "$OLD_CROP_RUN" \
  --train-config configs/s1_convergence.json --view both \
  --selector-assets "$ORIGINAL_CROP_REPO/local/visionrl2" --time 2-00:00:00
# Resume the same training run if walltime expires, retaining best/latest:
python3 run_s1_crop_study.py --mode s1_convergence --resume "$TRAIN_RUN" --time 2-00:00:00
```

`--smoke-images 2` validates the same pipeline and gradient path in a short
allocation before a full run; it cannot support a scientific conclusion. Full
evaluation/training must use new run directories without this smoke limit.
