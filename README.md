# Q-LRG for robust scene text VQA

Q-LRG tests whether substituting clean post-Merger visual features recovers TextVQA accuracy on LR4 images. D0 validates the frozen Qwen2.5-VL-3B control and feature injection. S0 compares all-token, text-region, answer-region, equal-budget random-region and dilated answer-region substitutions.

After cloning on the UIT-HPC login node, run `python3 run_all.py --mode full` from the checkout. It generates a Slurm script and submits one 500-image D0/S0 evaluation. S0 starts after D0 passes. Add `--review PATH` to use approved answer-region grounding for O3–O5. Without a review, the full job labels those arms as provisional OCR diagnostics. Use `--mode smoke` for a 12-image check.

The diagnostic set uses a fixed prefix of the TextOCR-covered images in a 90/10 image-level split of official train. The default 500 images form an initial precision and coverage check. Extend the prefix with the same seed when the paired intervals are too wide for the planned effect or the grounding cohort is too small.

- [UIT-HPC runbook](docs/h200_runbook.md)
- [D0 and S0 protocol](docs/research_blueprint.md)
- [Implementation decisions](docs/decisions.md)
- [Validation status](docs/implementation_review.md)

Source datasets, per-question manifests, review images, predictions, model weights and caches stay in the HPC workspace or another user-selected private location.
