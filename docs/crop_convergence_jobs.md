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
