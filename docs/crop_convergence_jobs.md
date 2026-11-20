# Submitted full/crop study jobs — 2026-11-20

Code: crop-branch commit `81bcfd0`, deployed as an independent archive at
`/datastore/dungmt/textvqa/Tuan/qlrg-s1-crop-81bcfd0`.
Existing crop/main checkouts and Danh's files were not modified.

| Job | Role | Initial state after submit | Walltime ceiling |
|---|---|---|---|
| 68268 | 32-arm old-11305 S1b full/crop factorial; A0 and A2 readers | PENDING (Priority) | 16 hours |
| 68269 | Fresh matched S1b full and crop convergence selection, then all-train refit | PENDING (Priority) | 72 hours |

These states are submission-time observations, not live job status or GPU validation.
Thirteen CPU unit tests passed in the pinned HPC Python environment before submit.
Identity and gradient gates run in the allocated GPU job before long inference/training.

Evaluation run:
`/datastore/dungmt/textvqa/Tuan/qlrg-s1-crop-81bcfd0/runs/crop_matrix-8ikp1szr`

Training run:
`/datastore/dungmt/textvqa/Tuan/qlrg-s1-crop-81bcfd0/runs/s1_convergence-q8cefict`

Each run has `launch.json`, a generated `.sbatch`, `job-id.txt`, and `logs/`.
The training job runs full first and crop second; it does not retrain A2. Its
retention is selection `best.pt` + `latest.pt`, refit `latest.pt` only, per view.
Thus six persistent checkpoint files maximum across both views and both stages.
Checkpoint replacements may create one short-lived temporary file for atomic save.

The evaluation reuses immutable S1b and A2 checkpoints under:
`/datastore/dungmt/textvqa/Tuan/Q-LRG-Robust-Scene-Text-VQA/runs/s1-20261107T051836Z-wymhdtwk/s1/variants/`

The proposal source is the completed official crop selector benchmark:
`/datastore/dungmt/textvqa/Tuan/Q-LRG-crop/runs/crop-benchmark-02j6px_m`.

```bash
sacct -j 68268,68269 --format=JobID,JobName%25,State,ExitCode,Elapsed

snapshot=/datastore/dungmt/textvqa/Tuan/qlrg-s1-crop-81bcfd0
tail -F "$snapshot/runs/crop_matrix-8ikp1szr/logs/qlrg-crop-matrix-68268.out" \
        "$snapshot/runs/crop_matrix-8ikp1szr/logs/qlrg-crop-matrix-68268.err"
tail -F "$snapshot/runs/s1_convergence-q8cefict/logs/qlrg-s1-convergence-68269.out" \
        "$snapshot/runs/s1_convergence-q8cefict/logs/qlrg-s1-convergence-68269.err"

# If walltime expires, resume only after the previous job is inactive:
cd "$snapshot"
python3 run_s1_crop_study.py --mode s1_convergence \
  --resume "$snapshot/runs/s1_convergence-q8cefict" --time 3-00:00:00
```

See `docs/crop_convergence_study.md` for all 32 arms, model/metric meanings,
early-stopping criteria, and how all 21,953 official train images are used without
early-stopping on images already merged into the final refit training set.

## Verified results and status on 2026-11-20

Job **68268 completed, exit 0:0, elapsed 06:09:08**. All eight COMPLETE artifact
SHA-256 values and the run binding passed verification. Predictions reproduce
all 32 reported accuracies: exactly **160,000 predictions**, 5,000 common
questions / 3,166 images in every arm. All 12 identity gates passed.

Job **68269 was RUNNING** when checked; its full-image S1b gradient probe passed
(gradient norm 0.575175). It was computing the initial full-image baseline; no
completed training epoch, selected checkpoint, crop-training result or refit
result was available at that observation. This status is not a live monitor.

All numbers below are TextVQA accuracy (%) on LR4. A0 is the frozen original
Qwen2.5-VL-3B; A2 is the frozen 11305 LLM Q/V LoRA. Every restored arm uses the
same old full-image-trained S1b checkpoint from 11305. Vision-RL² selects crops
from LR4 + query, with 20% padding. These are not the new 68269 checkpoints.

### Same total budget of 576 reader visual tokens

| Input | S1b on full | S1b on crop | A0 reader | A2 reader |
|---|---|---|---:|---:|
| Full | no | — | 62.686 | 65.216 |
| Full | yes | — | 63.972 | 64.312 |
| Crop | — | no | 61.072 | 63.442 |
| Crop | — | yes | 62.426 | 62.726 |
| Full + crop | no | no | 63.878 | 65.982 |
| Full + crop | yes | no | 65.246 | 66.050 |
| Full + crop | no | yes | 64.934 | 65.874 |
| Full + crop | yes | yes | 65.174 | 65.604 |

Fusion allocates 288 tokens to each view. With A2, restoring full only raises
the point estimate by 0.068 pp versus no restoration, CI [−0.328, +0.460]; this
is not evidence of a restoration benefit despite its top point estimate.

### Native resolution budgets with different token counts

| Input | Mean reader tokens | S1b on full | S1b on crop | A0 reader | A2 reader |
|---|---:|---|---|---:|---:|
| Full | 929.42 | no | — | 63.660 | 65.220 |
| Full | 929.42 | yes | — | 64.944 | 64.232 |
| Crop | 393.40 | — | no | 62.264 | 64.510 |
| Crop | 393.40 | — | yes | 63.718 | 64.244 |
| Full + crop | 1320.99 | no | no | 65.926 | 67.558 |
| Full + crop | 1320.99 | yes | no | 66.964 | 66.804 |
| Full + crop | 1320.99 | no | yes | 66.836 | 67.248 |
| Full + crop | 1320.99 | yes | yes | 67.006 | 66.638 |

Within an input layout, toggles retain exactly the same images, query and token
counts. Across native layouts, accuracy comparisons are not token-budget matched.

### S1b effects and reader interaction

| Budget | Input | S1b effect with A0, pp [95% CI] | S1b effect with A2, pp [95% CI] |
|---|---|---:|---:|
| 576 | Full | +1.286 [0.517, 2.066] | −0.904 [−1.475, −0.360] |
| 576 | Crop | +1.354 [0.571, 2.111] | −0.716 [−1.261, −0.158] |
| Native | Full | +1.284 [0.481, 2.082] | −0.988 [−1.518, −0.455] |
| Native | Crop | +1.454 [0.747, 2.146] | −0.266 [−0.752, 0.229] |

For native full images, the reader interaction is
`(A2+S1b − A2) − (A0+S1b − A0) = −2.272 pp`, CI [−3.187, −1.345].
For native crop it is −1.720 pp, CI [−2.510, −0.918]. All ten tested interactions
(full, crop, three fusion toggles × two budgets) are negative with pointwise CIs
below zero. These quantify non-additivity on the accuracy scale; they do not
identify a feature-level cause. CIs resample image clusters with 10,000 draws,
conditional on these checkpoints, with no training-seed or multiplicity correction.

At native full resolution, 217/372 questions improved by S1b over A0 were already
fully correct with A2. Twenty-one questions improved with S1b on A0 but worsened
when the same S1b was attached to A2; ten of those show simultaneous A0 0→1 and
A2 1→0. Both overlap and harmful interactions need inspection.

**Correction to the earlier crop interpretation:** the former review concerned
an A2 reader and deliberately selected regressing cases. S1b trained on full
images improves crop VQA with A0 in this matched experiment. Domain shift from
full to crop can still matter, but it cannot explain the results by itself.

Artifacts downloaded and verified locally under `local/crop-matrix-68268/`.
`scripts/analyze_crop_matrix.py` rechecks integrity/cohort/scores and writes
`interactions.csv`, interaction case IDs and an analysis summary. Reproduce:

```bash
python -B -m scripts.analyze_crop_matrix \
  --run local/crop-matrix-68268 --output local/crop-matrix-68268-analysis
```

## Training failure and checkpoint recovery

Checked again 2026-11-20: 68269 FAILED (1:0) after 09:39:26; 74081 FAILED
(1:0) after 06:23:54. Neither finished selection/refit or crop training.
Both stderr files were only 329 bytes and ended at the Traceback header,
without an exception type/message or source frame. This does not establish
an OOM, quota error or software exception. Node system logs were inaccessible
to this account. Current 4 KiB and full-checkpoint-size (~12 MB) writes/fsync
passed in both output directories, so current quota exhaustion was not observed.

Validated latest checkpoints include model, optimizer, RNG, selection state and
history. Bindings and original worker SHA-256 values match:

| Old job | Latest example visits | Fractional epoch | Recovery job |
|---|---:|---:|---|
| 68269, S1b through A0 | 56,000 | 1.79897 | **81949** |
| 74081, restoration through A2 | 32,000 | 1.02798 | **81950** |

Both recovery jobs were PENDING (Priority) immediately after submission. They
reuse the same run directories and original pinned workers/configurations;
model/data/loss/optimizer/schedule are not changed, and training is not restarted.
Work after those saved offsets must be repeated; stdout's later progress was
not a recoverable checkpoint.

Recovery snapshot `4a78a6c` adds explicit atomic-write retries on transient I/O
errors, an independent exception journal, and a fixed emergency checkpoint
backup only if all checkpoint-write retries fail. No claim is made that the
unknown original root cause has been fixed. The improved observability will
capture the actual exception if it recurs. Recovery behavior is recorded in
each run's `recovery/launch.json` and `/home/dungmt/.qlrg_recovery/<run-name>/`.

The recovery tests passed locally and on HPC, including preservation of an
existing checkpoint when writes fail. CPU preflight verified that I/O aliases
are patched while original source hashes and checkpoint bindings stay intact.

```bash
sacct -j 81949,81950 --format=JobID,JobName%35,State,ExitCode,Elapsed
tail -F /datastore/dungmt/textvqa/Tuan/qlrg-s1-crop-81bcfd0/runs/s1_convergence-q8cefict/logs/qlrg-recover-81949.out \
        /datastore/dungmt/textvqa/Tuan/qlrg-s1-crop-81bcfd0/runs/s1_convergence-q8cefict/logs/qlrg-recover-81949.err
```
