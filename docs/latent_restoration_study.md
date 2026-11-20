# Q-LRG latent restoration study

## Protocol

This study uses visual features as input and output. It never decodes restored
features to RGB. The first round trains a new, matched S1b control and six
ablations on the same paired CLEAN/LR4 train_inner examples, with the same
order, seed, learning rate, answer objective and frozen Qwen2.5-VL-3B.
Report all val_inner questions by default. Keep official_val for confirmation
after locking methods, hyperparameters and reporting contrasts. Include A0
(frozen reader), A2 (frozen trained language LoRA), the restorer with A0, and
the same restorer with A2. Fit channel scales and TextOCR objectives using
training images only. TextOCR supplies no inference input. Keep the crop study
(job 17818) separate. A subsequent crop-trained restorer should use LR4/CLEAN
pairs from the same training box, with the same resize and token grid.

All effects are paired TextVQA accuracy differences with 95% image-cluster
bootstrap intervals. Compare each new method with the matched re-trained S1b
on the same cases and reader. Report CLEAN regressions, the degradation
interaction, and the number of methods compared when discussing statistical
evidence. A first seed screens hypotheses. Repeat promising contrasts with
additional seeds before confirming on official_val. Loss-scale settings below
are initial Q-LRG choices, not values attributed to the papers.

## First-round methods implemented

| CLI method | Mechanism implemented | Attribution and changes |
|---|---|---|
| `s1b` | Original post-Merger spatial residual, global MSE, answer loss and residual regularization | Q-LRG control. Same architecture and schedule as every loss-only ablation. |
| `text_weighted` | Mask-aligned MSE with 5× weight on TextOCR cells. The loss stays normalized by total weight. Unannotated images use the ordinary loss. | Inspired by foreground-only supervision in TERA §4.3 and adaptive spatial weighting in Yang et al. (ECCV 2022). TERA distills detection-head responses. This implementation weights Qwen visual feature errors using text polygons. |
| `text_relational` | Text-weighted loss plus squared errors in the cosine similarities of adjacent visual tokens in text regions | Inspired by Parikh et al. §3, relational distillation. They supervise intermediate LLM states with a relational distribution over selected tokens. We compare horizontal and vertical neighbors after Merger, with Qwen frozen and no synthetic token masking. |
| `scale_aware` | MSE after per-token channel normalization, plus squared differences of log standard deviation across channels | Adapted from Uni-DocRobust §4.3.2, Eqs. 6–7. Kept after Merger for the matched loss ablation. The paper uses a pretrained Transformer core and adapters before Merger. Q-LRG's coefficient is 1. |
| `frequency` | AvgPool residual and MaxPool residual on the 2D bottleneck grid with learned per-channel gains | Inspired by FMVR §3.2, Eqs. 2–7. Uses the pooling cues inside S1b at unchanged grid size. The paper uses multiplicative modulation on Matryoshka-compressed visual tokens. This tests transfer to LR4 rather than replicating the paper. |
| `bounded` | Scale the residual by a fixed CLEAN train_inner per-channel scale and bound it with tanh at ±3 standard deviations | Adapted from TERA §4.2, Eqs. 8–9. Estimate scale from the first 128 selected training images (or all if fewer). TERA uses detector-neck states and both received and restored streams. Q-LRG uses one Qwen feature stream. |
| `text_gate` | Text-weighted S1b plus a learned per-token sigmoid gate, trained from TextOCR masks and predicted from visual features at inference | Q-LRG extension motivated by our S0 substitution oracle and TERA's foreground supervision. Compare directly against text_weighted to isolate the learned gate. The oracle uses clean tokens at evaluation and establishes headroom only. |

The stage writes independent checkpoints and per-question predictions. The
launcher emits Slurm scripts by default and submits with `--submit`. Each
checkpoint path is overwritten atomically at sparse intervals. Paired visual
features are kept in memory for the current image to limit persistent storage.

```bash
main=/datastore/dungmt/textvqa/Tuan/Q-LRG-Robust-Scene-Text-VQA
"$main/.venv/bin/python" -B run_latent_study.py \
  --s1-run "$main/runs/s1-20261107T051836Z-wymhdtwk" \
  --env "$main/.venv" --train-images 4000 --epochs 1 \
  --methods s1b text_weighted text_relational scale_aware frequency bounded text_gate
```

## Planned follow-ups

| Method | Isolated experiment and required control | Citation status |
|---|---|---|
| MAA | Add a train-only discriminator on Qwen projected visual tokens. Compare paired MSE with paired MSE + adversarial loss at the same injection sites. Evaluate single-site first, then distributed ViT adapters with the same placements in both arms. | Li et al. ICML 2026 §3.2–3.3 and Fig. 3. MAA freezes the vision encoder and projector and discards the discriminator at inference. Q-LRG's single-site variant would be an adaptation. |
| Pre-Merger | Compare the same narrow restorer on the last ViT grid versus the merged grid, using audited feature order and identity checks for the two paths. | Uni-DocRobust §4.4 motivates the site. A single-encoder implementation is an adaptation. |
| Degradation conditioning | Compare a shared restorer trained on matched CLEAN/LR4/Blur against one conditioned by a degradation embedding inferred from the input. Hold architecture width and sample exposure constant. Evaluate LR4 + Blur separately. | RobustVisRAG §3.2 separates degradation and semantic paths. ROFER conditions feature correction on degradation. DARA routes low-rank residual experts. A single lightweight conditioning branch is a Q-LRG adaptation. |
| FLEX and flow | Compare normalized feature MSE with and without percentile masking. Track how many masked activations come from text. For flow, compare a time-conditioned velocity objective at 1 and 2 steps to a one-step residual with a matched compute control. | RestoRect OpenReview manuscript, §3 and FLEX appendix. The original reconstructs images, so only the ideas transfer. A one-step Euler update is still an additive residual at inference and extra evaluations add cost. Publication decision unverified. |
| Character supervision | Train an auxiliary recognizer over region-aligned visual features using train-only TextOCR transcripts. Evaluate transcription and VQA separately, and drop the auxiliary head at inference. | Motivated by TOLA's OCR supervision. TOLA operates in diffusion/VAE latents and decodes RGB. A Qwen feature OCR head is a Q-LRG adaptation. FiDeSR is a related diffusion-SR mechanism, with no VLM feature reproduction planned. |
| Crop teacher / feature super-resolution | Compare full-trained and crop-trained S1b at fixed token count. Evaluate token expansion separately against interpolation at the same resulting token count. | Blink §3.2.1 and App. A.2, Pixel Distillation feature super-resolution. FeatUp/AnyUp concern feature upsampling, with different input assumptions. |
| Question conditioning | After establishing the text gate, add the question embedding. Compare to text-only gate, same parameter budget, and a shuffled-query diagnostic. | TG-LLaVA, QA-ViT and iGVLM are precedents for query-conditioned visual feature modulation. They do not establish LR4 restoration efficacy. |
| Reading vs answering | Freeze a crop, compare CLEAN and LR4, and score transcription exact match and CER alongside VQA. Use gold transcription only in an explicitly labeled oracle. | Vatsi et al. motivates measuring downstream VQA alongside OCR. It does not validate this latent restorer. |

Text Gestalt, TPGSR, TATT, LEMMA and MARCONet are related text-SR papers.
Their OCR and glyph ideas can inform auxiliary supervision. They are not
treated as Qwen visual-feature baselines. Similarly, visual quality studies
and pixel restoration outputs are background literature only for this study.

## Primary references

- Zhou et al. 2026. *Uni-DocRobust: Universal Plug-and-Play Robustness Enhancement for Multi-modal LLMs via Feature Restoration*. ICML. [Proceedings and PDF](https://proceedings.mlr.press/v306/zhou26aw.html).
- Li et al. 2026. *Robust Vision-Language Models via Manifold-Adversarial Adapters*. ICML. [Proceedings and PDF](https://proceedings.mlr.press/v306/li26hs.html).
- Pan, Dou and Li. 2026. *Frequency-Modulated Visual Restoration for Matryoshka Large Multimodal Models*. CVPR Findings. [Proceedings](https://openaccess.thecvf.com/content/CVPR2026F/html/Pan_Frequency-Modulated_Visual_Restoration_for_Matryoshka_Large_Multimodal_Models_CVPRF_2026_paper.html).
- *TERA: Task-Evidence Restoration for Underwater Object Detection under Bandwidth-Constrained Transmission*. 2026 preprint. [arXiv:2611.02419](https://arxiv.org/abs/2611.02419).
- Parikh et al. 2026. *Latent Denoising Improves Visual Alignment in Large Multimodal Models*. [arXiv:2604.21343v2](https://arxiv.org/abs/2604.21343v2), [official code](https://github.com/dhruvashp/latent-denoising-for-lmms).
- Yang et al. 2022. *Self-Feature Distillation with Uncertainty Modeling for Degraded Image Recognition*. ECCV. [Proceedings](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3874_ECCV_2022_paper.php).
- Chen et al. 2026. *RobustVisRAG*. CVPR. [Proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_RobustVisRAG_Causality-Aware_Vision-Based_Retrieval-Augmented_Generation_under_Visual_Degradations_CVPR_2026_paper.html).
- Zhou et al. 2023. *Robust Feature Rectification of Pretrained Vision Models for Object Recognition*. AAAI. [Paper](https://ojs.aaai.org/index.php/AAAI/article/download/25492/25264).
- *DARA*. 2026. [arXiv:2607.16644](https://arxiv.org/abs/2607.16644).
- *RestoRect: Degraded Image Restoration via Latent Rectified Flow & Feature Distillation*. [OpenReview manuscript](https://openreview.net/pdf?id=weh7JpT1lF).
- *TOLA: Text-aware One-Step Latent Adaptation for Diffusion-based Text Image Super-Resolution*. 2026 preprint. [arXiv:2609.29240](https://arxiv.org/abs/2609.29240).
- Kim et al. 2026. *FiDeSR*. [arXiv:2603.02692](https://arxiv.org/abs/2603.02692).
- *Blink: Dynamic Visual Token Resolution for Enhanced Multimodal Understanding*. [arXiv:2512.10548v3](https://arxiv.org/abs/2512.10548v3).
- *Pixel Distillation*. [arXiv:2112.09532](https://arxiv.org/abs/2112.09532). *FeatUp*. [arXiv:2403.10516](https://arxiv.org/abs/2403.10516). *AnyUp*. [arXiv:2510.12764](https://arxiv.org/abs/2510.12764).
- *TG-LLaVA*. [arXiv:2409.09564](https://arxiv.org/abs/2409.09564). *QA-ViT*. [arXiv:2402.05472](https://arxiv.org/abs/2402.05472). *iGVLM*. [arXiv:2603.02748](https://arxiv.org/abs/2603.02748).
- Vatsi et al. 2026. *Beyond OCR Accuracy: Text-Centric VQA Under Image Degradation with Modular and End-to-End*. [arXiv:2609.13815](https://arxiv.org/abs/2609.13815).

Use `--grouped --time 2-00:00:00` to run these methods serially in one
Slurm allocation. Each method writes into its own subdirectory, including
`worker.log` and its checkpoint. The batch continues with the remaining
methods if one fails and exits nonzero if any method did not finish.

## Restoration and A2 cooperation after the 32-arm evaluation

Evidence checked 2026-11-20: job 68268, all 5,000 official-val questions, same
full-trained 11305 S1b, same views/budgets with A0 and A2. See
`docs/crop_convergence_jobs.md` for all 32 scores, paired CIs, and the correction
to the earlier crop-only interpretation. Job 68269 trains independent full/crop
S1b through A0, not through A2; it remains a useful convergence/data control.

### What is established and what is still a hypothesis

- S1b improves A0 for both full and crop, in both token regimes. Its gains do not
  transfer additively to A2; some A2 combinations regress with CIs below zero.
- Native full: the effect changes from +1.284 pp on A0 to −0.988 pp on A2. The
  interaction is −2.272 [−3.187, −1.345] pp. This is score-level evidence, not
  proof of off-manifold features, scale collapse, hallucination or attention failure.
- A0 and A2 solve overlapping cases; more than half the native-full A0 improvements
  from S1b are already correct with A2. A negative interaction alone does not
  distinguish redundant benefits from harmful representation changes.
- S1b already has an answer CE loss, but that loss was computed through A0.
  A2 was separately trained on native CLEAN/LR4 features. Neither was trained on
  the composed S1b→A2 path. Whether this causes the regressions needs a controlled test.
- A current A2 regression observed on crop cannot be generalized to all crop
  restoration: the same crop-trained/full-trained issue also needs a reader control.

### First candidate: train restoration through a fixed A2 reader

Keep vision encoder, Merger, base LLM and 11305 LoRA fixed. Only update a COPY
of S1b, passing its output into that A2 reader. Use the actual CLEAN/LR4 pair
and query, with the same LR4-selected crop coordinates on both images for crop
training. There is no RGB decoder and no answer-dependent inference selector.

`L = CE(A2(F_restored), answer) + λ MSE(F_restored, F_CLEAN) + μ ||delta||²`

Retain CLEAN identity supervision/mixing and select on val_inner VQA plus the
same-view CLEAN safeguard. Frozen A2 parameters still permit gradients through
the LLM to S1b. The change is the task-loss reader, not the introduction of an
answer loss for the first time. Use a fresh output directory and preserve originals.

Controls: frozen A2 alone; frozen A2 + unchanged old S1b; S1b continued for the
same added updates through A0; and the same restoration module trained through
A2 without feature loss. The last control tests whether feature restoration adds
anything beyond a spatial adapter. Start with a matched short continuation,
select on val_inner, report actual GPU-hours and updates, and scale promising
contrasts. Full and crop should not be pooled into an unlabelled result.

### Second candidate: constrained joint S1b and LoRA fine-tuning

If fixed-reader training is insufficient, jointly update S1b and a COPY of A2,
with feature restoration and answer objectives together. Include A2-only further
training with the same data/order/updates, and a no-feature-loss joint control.
Otherwise, extra language adaptation could explain all the gain. Preserve base
Qwen weights, use separate optimizer groups, validate CLEAN/LR4 separately,
and keep only best/latest. Learning rates require val_inner selection; values
from another model are not justified defaults for this warm-started combination.

A CLEAN-A2 teacher may later supply answer-distribution consistency under the
same reference-answer prefix, including EOS. This is optional: CLEAN teacher
predictions can be wrong, so ground-truth CE remains primary and teacher errors
must be measured before adding a KL constraint. This is a Q-LRG proposal, not a
reproduction of the relational feature KL in Parikh et al.

### Cheap diagnostics before further full training

On val_inner, test `F + alpha * delta` at alpha 0, 0.25, 0.5 and 1 under both
readers, holding the crop/tokens/query fixed. Alpha=0 is the exact reader baseline;
an intermediate optimum suggests residual magnitude is worth investigating,
without proving it is the only cause. Track feature norms, normalized residual
size, CLEAN distance and answer/EOS loss; inspect wrong-character versus
over-answer cases. These diagnostics are proposed, not already executed.

Do not select hyperparameters on the 5,000 official-val questions. They have
already been examined repeatedly in project experiments and are not a pristine
holdout. Additional training seeds are needed before generalizing a gain beyond
the selected checkpoints.

### Paper mechanism ledger for this cooperation question

- **Parikh et al., ACM MM 2026, v2, §3.3 and Appendix E:** visual reconstruction,
  relational and contrastive supervision accompany language training; LoRA
  ablations also train projector/auxiliary modules. This supports studying
  co-optimization, not post-hoc additivity. Their latent corruption and mid-LLM
  recovery differ from our input LR4 and post-Merger S1b. [Full paper](https://arxiv.org/html/2604.21343v2).
- **Li et al., MAA, ICML 2026:** paired alignment plus a token adversary targets
  the clean feature manifold; Euclidean distance alone need not preserve task
  semantics. Its official release targets LLaVA-v1.6-Mistral-7B. It motivates
  testing semantic compatibility but does not establish our failure's cause or
  provide a ready Qwen3B+A2 recipe. The first candidate does not implement MAA's
  discriminator. [Paper](https://proceedings.mlr.press/v306/li26hs.html),
  [official code](https://github.com/acnul/MAA).
- **LLaVA official LoRA recipe:** loads a pretrained multimodal projector and
  trains it alongside LoRA with separate learning rates. This is an engineering
  precedent for co-adaptation, not a latent-restoration baseline.
  [Official training script](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_lora.sh).

These candidates and diagnostics are research proposals. No additional GPU job
was submitted for them, and the running 68269 snapshot was not changed.
