"""Run the four locked checkpoints on an image and a question inside a Slurm GPU allocation."""

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch

from qlrg.common import read_json, write_json
from qlrg.data import load_clean
from qlrg.qwen_wrapper import Qwen
from scripts.final_evaluate import load_models, prepare
from scripts.final_review import LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this demo inside a Slurm GPU allocation")
    if args.output and args.output.exists():
        parser.error("Choose a new output file")
    output = args.evaluation.resolve()
    _, paths, config, trained, _, module, states, _ = prepare(output)
    os.environ["QLRG_FEATURE_CACHE_MODE"] = "read-only"
    source = Path(read_json(output / "launch.json")["s1_run"]) / "s1"
    os.environ["QLRG_FEATURE_CACHE"] = read_json(source / "storage.json")["cache"]
    image = args.image.resolve()
    clean, orientation, digest = load_clean(image)
    record = {"image_id": "demo-" + digest, "image_rel": str(image), "image_sha256": digest,
              "exif_orientation": orientation, "size": list(clean.size),
              "questions": [{"question_id": "demo", "question": args.question}]}
    model = Qwen(config, paths["hf_hub"])
    cache, adapters = load_models(model, module, paths, output, trained, states)
    results = []
    with torch.inference_mode():
        cache.load(record)
        for condition, base in cache.features.items():
            batch = cache.prompt(condition, args.question)
            for arm, label in LABELS.items():
                feature = base if arm in ("A0", "A2") else adapters[arm](base, cache.grid)[0].to(model.dtype)
                with adapters["A2"].attached() if arm == "A2" else nullcontext():
                    answer = model.generate(batch, feature)
                results.append({"condition": condition, "arm": arm, "label": label,
                                "visual_tokens": len(feature), **answer})
    result = {"image": str(image), "question": args.question, "results": results}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        write_json(args.output, result)


if __name__ == "__main__":
    main()
