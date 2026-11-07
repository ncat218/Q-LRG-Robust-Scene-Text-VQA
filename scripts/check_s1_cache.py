"""Estimate additional feature-cache writes for an interrupted expanded S1 run."""

import argparse
import math
import shutil
from pathlib import Path

from qlrg.common import object_hash, read_json


def estimate_cache(images, shared, cache, model_config, resize):
    model = shared["model"]
    patch = model_config.vision_config.patch_size
    merge = model_config.vision_config.spatial_merge_size
    factor = patch * merge
    # Two feature arrays, each stored in BF16.
    bytes_per_token = 4 * model_config.text_config.hidden_size
    entries = {path.name for path in cache.iterdir() if path.suffix == ".safetensors"}
    required = {}
    for image in images:
        width, height = image["size"]
        h, w = resize(height, width, factor=factor, min_pixels=model["min_pixels"], max_pixels=model["max_pixels"])
        key = object_hash({
            "image_sha256": image["image_sha256"], "grid": [[1, h // patch, w // patch]],
            "protocol": "qlrg-post-reverse-indices-lr4-bicubic-v1",
            "model": model, "source_hash": shared["source_hash"], "packages": shared["packages"],
            "torch_build": shared["torch_build"], "gpu": shared["device"],
        })
        # Round to 4 KiB and allow an extra page for the safetensors header.
        required[key + ".safetensors"] = math.ceil((h // factor) * (w // factor) * bytes_per_token / 4096) * 4096 + 4096
    missing = required.keys() - entries
    return len(required) - len(missing), len(missing), sum(required[name] for name in missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Parent directory of the existing s1/ stage")
    parser.add_argument("--budget-gb", required=True, type=float, help="Writable quota headroom in decimal GB")
    parser.add_argument("--cache", type=Path, help="Cache directory, defaults to the run's recorded location")
    args = parser.parse_args()
    if not math.isfinite(args.budget_gb) or args.budget_gb <= 0:
        parser.error("--budget-gb must be finite and positive")

    import transformers
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

    output = args.run / "s1"
    shared = read_json(output / "run.json")["shared"]
    model = shared["model"]
    if model["dtype"] != "bfloat16" or transformers.__version__ != model["transformers"]:
        raise ValueError("Use the original pinned S1 environment and BF16 model")
    model_cache = Path(shared["paths"]["hf_hub"]) / ("models--" + model["id"].replace("/", "--"))
    snapshot = model_cache / "snapshots" / model["revision"]
    model_config = transformers.AutoConfig.from_pretrained(snapshot, local_files_only=True)
    cache = args.cache or Path(read_json(output / "storage.json")["cache"])
    images = [*read_json(output / "train_manifest.json"), *read_json(output / "eval_manifest.json")["images"]]
    present, missing, needed = estimate_cache(images, shared, cache, model_config, smart_resize)
    reserve = 3 * 1024**3
    filesystem_free = shutil.disk_usage(cache).free
    budget = args.budget_gb * 10**9
    print(f"Cache: {cache}")
    print(f"Matching cache entries: {present:,}. Missing entries: {missing:,}.")
    print(f"Estimated additional cache: {needed / 10**9:.2f} GB")
    print(f"Estimated total with 3 GiB reserved for checkpoints, reports and crop: {(needed + reserve) / 10**9:.2f} GB")
    print(f"Declared writable quota: {args.budget_gb:g} GB. Filesystem-wide free: {filesystem_free / 10**9:.2f} GB.")
    if needed + reserve > min(budget, filesystem_free):
        print("STOP: estimated writes exceed the supplied budget.")
        return 2
    print("Estimate fits the supplied budget. Confirm quota and the original cache path before submitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
