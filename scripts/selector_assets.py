"""Stage pinned region selectors while sharing identical checkpoint shards."""

import json
import os
import subprocess
import sys
from pathlib import Path

from filelock import FileLock
from huggingface_hub import hf_hub_download, snapshot_download


UPSTREAM = "https://github.com/YuHengsss/VisionRL2.git"
UPSTREAM_REVISION = "2141e48cf57f374ae63f0a862ecdd3c3185cc2a5"
MODELS = {
    "sdrpn_7b": ("YuhengSSS/SDRPN-Qwen2.5-VL-7B", "2dedcb3c4ed298c96f3174771b96288078d469ee"),
    "visionrl2_7b": ("YuhengSSS/VisionRL2-Qwen2.5-VL-7B", "47d656abd90a837d902d296c4f986ca6acc7415d"),
}
# Hugging Face lists identical LFS SHA-256 values for these three model shards.
SHARED_SHARDS = [f"model-{index:05d}-of-00004.safetensors" for index in (1, 2, 3)]
DATA_REVISION = "a947010b7014e35c6b0b492f10814990f1ab7c96"
DATA_FILES = ("rl_pools/candidates_visualcot_50k.jsonl", "rl_pools/rl_pool_qwen2_5_vl_7b.jsonl")


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    vendor = root / "source"
    cache = root / "hub"
    with FileLock(str(root / ".assets.lock"), timeout=3600):
        vendor.mkdir(exist_ok=True)
        if not (vendor / ".git").exists():
            subprocess.run(["git", "init", "-q", str(vendor)], check=True)
        head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=vendor,
                              capture_output=True, text=True)
        if head.stdout.strip() != UPSTREAM_REVISION:
            subprocess.run(["git", "fetch", "--depth=1", UPSTREAM, UPSTREAM_REVISION], cwd=vendor, check=True)
            subprocess.run(["git", "checkout", "--detach", UPSTREAM_REVISION], cwd=vendor, check=True)
        checkpoints = {}
        for arm, (name, revision) in MODELS.items():
            print(f"Preparing selector weights: {name}@{revision}", flush=True)
            path = Path(snapshot_download(
                name, revision=revision, cache_dir=cache, max_workers=2,
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
                ignore_patterns=SHARED_SHARDS if arm == "visionrl2_7b" else None,
            ))
            if arm == "visionrl2_7b":
                for shard in SHARED_SHARDS:
                    destination = path / shard
                    if not destination.is_file():
                        destination.symlink_to(os.path.relpath(checkpoints["sdrpn_7b"] / shard, path))
            index = json.loads((path / "model.safetensors.index.json").read_text())
            if not all((path / name).is_file() for name in set(index["weight_map"].values())):
                raise ValueError(f"Incomplete checkpoint: {path}")
            checkpoints[arm] = path
        training = [Path(hf_hub_download("YuhengSSS/VisionRL2-data", file, repo_type="dataset",
                                         revision=DATA_REVISION, cache_dir=cache)) for file in DATA_FILES]
    return vendor, checkpoints, training


if __name__ == "__main__":
    prepare(Path(sys.argv[1]).resolve())
