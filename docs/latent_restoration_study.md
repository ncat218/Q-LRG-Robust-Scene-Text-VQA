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
