"""Prepare the SD-RPN selector process without changing the Q-LRG environment."""

import subprocess
from pathlib import Path

from qlrg.common import read_json, sha256, write_json
from scripts.sdrpn_runtime import selector_environment


def selector_provenance(settings, repo):
    checkpoint = Path(settings["sdrpn_checkpoint"])
    config = read_json(checkpoint / "config.json")
    if (config.get("enable_twig"), config.get("twig_K"), config.get("twig_T"), config.get("roi_source")) != (
        True, 21, 3, "qk",
    ):
        raise ValueError("Expected the SD-RPN Qwen2.5-VL-3B K21/T3 checkpoint")
    index = checkpoint / "model.safetensors.index.json"
    weights = sorted(set(read_json(index)["weight_map"].values()))
    if not weights or any(Path(name).name != name or not (checkpoint / name).is_file() for name in weights):
        raise ValueError("SD-RPN checkpoint has missing or invalid weight shards")
    vendor = Path(settings["sdrpn_vendor"])
    return {
        "id": "SD-RPN Qwen2.5-VL-3B K21/T3", "revision": checkpoint.name,
        "max_visual_tokens": 288, "roi_threshold": 0.15,
        "config_sha256": sha256(checkpoint / "config.json"), "index_sha256": sha256(index),
        "weight_blobs": {name: (checkpoint / name).resolve().name for name in weights},
        "source_sha256": {p.relative_to(vendor).as_posix(): sha256(p) for p in sorted((vendor / "qwen_src").rglob("*.py"))},
        "backend_sha256": sha256(Path(__file__)),
        "worker_sha256": sha256(repo / "scripts" / "sdrpn_select.py"),
        "runtime_sha256": sha256(repo / "scripts" / "sdrpn_runtime.py"),
        "libexpat_sha256": sha256(settings["sdrpn_libexpat"]) if settings.get("sdrpn_libexpat") else None,
    }


def run_selector(settings, repo, output, paths, config, selected, donors):
    request = {
        "textvqa": str(paths["textvqa"]), "prompt": config["prompt"], "seed": settings["seed"],
        "images": [{
            **{key: image[key] for key in ("image_id", "image_rel", "image_sha256", "size", "exif_orientation")},
            "questions": [{
                "question_id": q["question_id"], "query": q["question"],
                **({"shuffled": donors[q["question_id"]][1]} if donors is not None else {}),
            } for q in image["questions"]],
        } for image in selected],
    }
    request_path = output / "sdrpn-request.json"
    write_json(request_path, request)
    snapshot = paths["hf_hub"] / ("models--" + config["id"].replace("/", "--")) / "snapshots" / config["revision"]
    proposals = output / "sdrpn-proposals"
    cache = output / ".sdrpn-runtime-cache"
    env = selector_environment(settings)
    env.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": str(cache / "hf"),
        "HF_HUB_CACHE": str(cache / "hf" / "hub"), "XDG_CACHE_HOME": str(cache), "MPLCONFIGDIR": str(cache / "mpl"),
        "TORCH_HOME": str(cache / "torch"), "PYTHONDONTWRITEBYTECODE": "1",
    })
    subprocess.run([
        str(Path(settings["sdrpn_env"]) / "bin" / "python"), "-B", str(repo / "scripts" / "sdrpn_select.py"),
        "--request", str(request_path), "--output", str(proposals), "--vendor", settings["sdrpn_vendor"],
        "--checkpoint", settings["sdrpn_checkpoint"], "--processor", str(snapshot),
    ], cwd=repo, env=env, check=True)
    return proposals
