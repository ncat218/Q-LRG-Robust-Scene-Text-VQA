# D0 and S0 protocol

This implementation follows the D0/S0 scope and the O0–O5 matrix in section 19 of Q-LRG Research Blueprint v1.1.

| Arm | Input and feature substitution |
| --- | --- |
| CLEAN | Native Qwen on the decoded clean image |
| LR4-BIC-CONTROL | Native Qwen on Pillow Bicubic downsample by four with floor dimensions, followed by Bicubic upsample to the original dimensions |
| O0 | Inject the original LR4-BIC-CONTROL features |
| O1 | Inject all aligned clean features on the LR4-BIC-CONTROL prompt and grid |
| O2 | Inject clean features at the TextOCR text mask |
| O3 | Inject clean features at the reviewed answer-region mask |
| O4 | Inject clean features at a seeded random region with the O3 token budget |
| O5 | Inject clean features at the O3 mask dilated by one merged cell in each direction |

D0 checks clean/degraded input IDs, attention masks and grids. Its parity gate checks exact final prefill logits and greedy generated token IDs on four distinct grids. The parity comparisons are native LR4 vs O0 and native clean vs O1. Features enter at the image feature boundary after Qwen has restored token order.

S0 reports O0/O1/O2 on the whole diagnostic manifest and all six arms on the same subset with usable O3 grounding. Automatic candidates require a unique exact OCR match and six agreeing annotators. The full launcher can use up to 150 automatic candidates, one per image, for a provisional diagnostic. A reviewed run instead uses explicit human review decisions. Rejected, unreviewed and empty projected masks contribute to coverage. The reports distinguish the grounding policies. The candidate pool favors direct questions with strong answer agreement.

Report the MMF TextVQA soft score and paired 95% percentile intervals. Resample image IDs with replacement, preserving every question of each sampled image. Report both the direction of any score change and crossings of the fully correct score. Inspect masks and cohort sizes before interpreting effects. O1 is a clean-feature reference and O2–O5 use annotation information for diagnosis. Decide on training after examining S0.
