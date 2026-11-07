# Implementation review

See the [experiment registry](experiment_registry.md) for completed H200 runs 10756 and 10820. The checks below record the initial D0/S0 preflight.

CPU preflight on UIT-HPC passed with the isolated Python 3.10.22 environment, the pinned CUDA 13.0 Torch build, matching Qwen source and matching MMF scorer.

- Nine CPU tests passed. They cover the Bicubic protocol, row-major mask mapping, exact-size random control, paired bootstrap, grounding review and resume integrity.
- The tiny random Qwen CPU test exercises mixed image-grid lengths through actual Transformers prefill and cached generation.
- The pinned fast processor produced aligned clean/LR4 inputs on 23 real questions across eight distinct grids, with six provisional O3 masks.
- The 500-image diagnostic manifest contains 792 questions and 267 literal O3 candidates. The review pack samples 150 different images. Those matches require review before full S0.
- Real H200 acceptance requires the pinned model to load, D0 identity and alignment to pass, and S0 to produce validated prediction, coverage, mask and scoring artifacts.
