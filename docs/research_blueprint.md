# Q-LRG research blueprint

This is the repository's implementation specification. It consolidates section 19 of the original Q-LRG Research Blueprint v1.1. The original notes remain the source for optional extensions. See the [experiment registry](experiment_registry.md) for observed results and open checks.

## Research questions

1. Can learned latent restoration improve degraded TextVQA against a frozen Bicubic control and adaptation baselines trained on the same data?
2. Does modifying features in text regions improve on global restoration? Does conditioning the region on the question improve on textness alone?

Intervene on Qwen2.5-VL-3B's merged visual tokens after token order has been restored. Preserve the visual grid, token count and position IDs. Report clean performance and question-level regressions alongside LR4 gains. A soft gate on a dense grid carries the same token count into the language model. Measure latency independently.

## Data and D0

- Split official TextVQA train by image ID into roughly 90% train_inner and 10% val_inner. Keep every question for each selected image. The diagnostic sample is a deterministic 500-image prefix of TextOCR-covered val_inner, with 792 questions in the first full run. It is a development set.
- Select hyperparameters on val_inner. Report the locked configuration on labeled official validation with paired intervals. Reserve official test submission for an optional final score. Preserve manifest hashes and check image, question and duplicate-file overlap. TextOCR supervision for learned heads must come from training data.
- Decode RGB in raw annotation orientation. For LR4-BIC-CONTROL, downsample by four with floor dimensions and Bicubic-upsample to the original canvas. CLEAN uses the same processor on the original decoded image. Keep directly resized LR inputs distinct from Bicubic control.
- D0 verifies preprocessing, prompt IDs, attention masks, grids, and exact prefill logits and generated tokens for native versus injected features on its identity examples. The current four-grid gate checks integration. Broader validation should deliberately cover non-identity EXIF, narrow and wide images, and both ends of the grid-size range. Stop on an identity mismatch.

## S0 — Oracle substitutions

| Arm | Operation |
| --- | --- |
| O0 | Inject unaltered LR4 features. |
| O1 | Replace the entire grid with aligned CLEAN features. |
| O2 | Replace tokens covered by TextOCR regions with aligned CLEAN features. |
| O3 | Replace tokens in an annotated answer region with aligned CLEAN features. |
| O4 | Replace tokens in a seeded random region matched to the O3 token count. |
| O5 | Replace tokens in the O3 mask expanded by one merged cell. |

O1 approximates the native CLEAN arm. Its comparison with O0 describes clean-versus-LR4 headroom. Identity checks verify the intended equivalence on sampled grids, while archived predictions can differ slightly in aggregate. Use the within-cohort gap to contextualize localized recovery. Do not use the clean gap alone to decide whether a localized intervention worked.

- On all questions, O2 versus O0 measures the effect of substituting text-region features. A separate control matched to O2's token budget is optional.
- On the same grounded questions, O3 versus O0 measures the benefit of answer-region substitution. O3 versus O4 controls token count. O4 matches O3, so an O2 versus O4 comparison also changes the amount of feature replacement. Record mask fractions and overlap with the answer mask.
- Use O3 versus O2 to compare answer-region and all-text coverage, and O5 versus O3 to test dilation. Both contrasts can change the token budget. Evaluate masks, paired confidence intervals and representative failures before drawing conclusions.
- Automatic grounding favors a unique exact OCR match with at least six agreeing answers. The provisional run selects at most one question per image and 150 images. It may miss multi-word answers, multiple matching regions, and context-dependent questions. Hand review can add valid multi-word spans and label context requirements. Record reviewed, rejected, ungroundable, empty-mask and unreviewed cases. Keep all oracle annotation and CLEAN substitutions in diagnostic evaluation.
- Digit-containing consensus answers are a numeric/date proxy. For small-text questions, use OCR word height in the grounded answer region. Record the fraction of small OCR words separately as an image-level statistic. Analyze context dependence on human-labeled grounded cases.
- Run overlap at 5% versus token-center selection on the same manifest and grounding file. Compare accuracy and mask size on matched questions. Report groundings lost by either mask rule separately. See the [runbook](h200_runbook.md).

## S1 — Global latent restoration

Use a zero-initialized residual visual adapter with local spatial layers. S1a learns by matching aligned CLEAN features. S1b combines feature matching, causal answer-token loss through the frozen language model, and a residual penalty. Both apply the same transformation for all questions on an image.

Compare against A0 (frozen Qwen), A1 (token-wise visual adapter with answer loss), and A2 (Q/V LoRA with answer loss). Add A1clean and A2clean with CLEAN-only training. Train on all train_inner without an OCR filter, and distinguish selection on val_inner from locked reporting on official validation. Use the same data, condition schedule, scoring and evaluation IDs. Log trainable parameters, optimizer steps, wall time and peak allocated memory. Non-reentrant gradient checkpointing and an early BF16 gradient check are required for training through the frozen 3B model. Inspect errors where feature similarity improves while answer accuracy worsens.

## S2 and S3 — Localization

Expanded S1 and A0 repeatability have priority. Under the current protocol, advance to S2/S3 after S1 shows clear paired gains against both A1 and A2 across seeds.

- **S2, textness.** Learn a soft gate over visual tokens with TextOCR masks on train_inner. Mask out textness supervision on examples without usable labels. Compare with S1b, ungated restoration and the adaptation controls. Report text and background token changes, localization quality and answer-level effects. Ground-truth boxes can serve as a separate oracle.
- **S3, question relevance.** Use the question and degraded image features to predict a relevance gate over the grid. Answer-region supervision may use trusted annotations in training. At evaluation, use the model's own predictions for the gate. Compare against S2, a question-shuffled gate and equal-budget controls. Measure localization on a held-out grounded sample, with coverage and context-dependent errors.
- Proceed to pre-merger high-resolution restoration (Phase B) if failures concentrate at small text that is unresolved after merging. Consider frequency-decomposed restoration (Phase C) if feature diagnostics motivate it. Both are conditional research extensions.

## Reporting

Use pinned MMF TextVQA soft accuracy and paired image-cluster bootstrap intervals. Report clean and degraded scores, the paired interaction Δ(LR4) − Δ(CLEAN), gain and regression counts, cohort and slice sizes, mask fractions and resource use. Compare LR4 with other degradation regimes separately, using a matched control and a fixed manifest. For final results, document seed variability and test the locked configuration on held-out data. A positive oracle result is evidence about a controlled substitution with CLEAN features. Learned restoration must demonstrate its own gains against adaptation controls.
