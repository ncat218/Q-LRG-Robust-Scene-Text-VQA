# Q-LRG experiment registry

Completed rows reference immutable run directories on UIT-HPC. The job number and artifact paths locate evidence. Per-question predictions, manifests and checkpoints stay with the run. Add a new row when changing code, preprocessing, cohort, grounding policy, or model. Record the manifest and source hashes, checkpoint/model revision, seed, comparison, paired interval and measured cost in the run metadata.

| Experiment | Status | Cohort and reference | Evidence |
| --- | --- | --- | --- |
| D0, job 10756 | Completed | 500 val_inner images / 792 questions. Native CLEAN versus LR4 Bicubic. Seed 42. | CLEAN gap +17.95 pp [14.94, 21.02]. Exact identity logits and tokens on four distinct grids. `runs/full500-20261106T120210Z-6z0yzfi_/d0/` |
| S0 overlap 5%, job 10756 | Completed, provisional grounding | Same 500 images. 150/792 grounded questions (18.94%), one per image. | O2 (all TextOCR) versus O0 (LR4 unchanged) +16.72 pp [13.95, 19.57] overall. O3 (answer region) versus O4 (equal-budget random) +18.47 pp [12.53, 24.73] on the grounded cohort. `runs/full500-20261106T120210Z-6z0yzfi_/s0/` |
| S1, job 10820 | Completed pilot | 2,048 train_inner images, 3,199 questions, one epoch. All five arms evaluated on the same 500 held-out development images. Seed 42. | S1b (answer + feature + residual losses) versus A0 (frozen Qwen) +1.39 pp [−0.42, 3.24] LR4. A2 (LoRA) versus A0 +2.63 pp [0.82, 4.47] LR4. S1b used about 7.55 GiB peak PyTorch allocated VRAM with gradient checkpointing. Job elapsed 01:15:26 on H200 MIG 35 GB. `runs/s1-20261106T161413Z-lm0jjs3z/s1/` |
| S0 center sensitivity | Planned | Pair overlap 5% and center at a fixed manifest and grounding policy. | Report per-question accuracy deltas, mask sizes and grounding lost by projection. |
| D0 expanded parity | Planned | Cover EXIF orientation and grid-size extremes explicitly. | Log which stress cases were available and tested. |
| S2 and S3 | Deferred pending expanded S1 versus adaptation controls | Evaluate textness and question guidance with S1b and adaptation controls. | Record localization and answer-level effects separately. |

## Interpretation notes

- In job 10756, O1 (all CLEAN features) versus O0 (LR4 unchanged) was +18.17 pp [15.17, 21.19]. O2 (TextOCR regions) recovered about 92% of that gap as a point estimate. On the 150 grounded questions, O3 (answer region) exceeded O2 by +1.07 pp [0.00, 2.80], with mean token fractions of 2.28% versus 12.97%. O4 (equal-budget random) shared no answer tokens in this run. It had the same score as O0 on all 150 questions despite differing generated tokens on six. This is one random control realization.
- The legacy `small_ocr_text` flag fired on 707/792 questions. Using the center rule would reduce the O2 (all TextOCR) token count on 790/792 questions. Average mask sizes were 145.8 tokens under overlap and 85.2 under center. These are mask counts. Accuracy sensitivity requires the paired center run.
- The archived native versus injected audit in job 10820 found a +0.13 pp shift for CLEAN and −0.09 pp for LR4 on 792 questions. Keep the matched within-run controls for comparisons.

## Next protocol

- Compare with the project's own native and injected frozen Qwen. Resolve current repeatability/path differences before interpreting historical flips as numeric noise.
- Job 10820 is an under-training pilot. Continued loss reduction motivates an expanded run but does not establish the cause of the current accuracy ranking.
- Bootstrap of the paired degradation-specific interaction from job 10820 gives S1b versus A0 −0.81 pp [−2.98, 1.29], A1 versus A0 −0.81 pp [−2.90, 1.23], and A2 versus A0 +1.59 pp [−0.52, 3.63]. These marginal intervals use 10,000 image-cluster resamples and seed 42. A2's point estimate is positive, with uncertainty spanning zero.
- Expanded manifests preserve train_inner for training, val_inner for selection, and labeled official validation for locked reporting. The pilot's training selection had no TextOCR filter. Main-path environment checks now require the TextVQA annotations and model artifacts.
- Multi-epoch, learning-rate and seed studies, including CLEAN-only controls, remain pending actual A0 audit results and resource review. S2/S3 are deferred under the current decision rule.
