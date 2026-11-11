"""Shared cohort, checkpoint and output handling for the two follow-up jobs."""

import csv
import io
import os
from pathlib import Path

from qlrg.common import atomic_write, dataset_paths, object_hash, read_config, read_json, sha256, write_json, write_jsonl
from qlrg.data import verify_pillow
from qlrg.s1_manifests import read_s1_manifests
from qlrg.s1_run import runtime_shared
from scripts.query_crop import checkpoints_for


def load_study(output, mode, arms):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this worker in a Slurm GPU allocation")
    verify_pillow()
    launch = read_json(output / "launch.json")
    if launch["study"] != mode or launch["split"] != "official_val":
        raise ValueError("Expected a locked official validation launch")
    paths = dataset_paths(read_config(launch["paths"]))
    config = read_config(Path(launch["repo"]) / "configs/model_qwen.yaml")
    manifests, hashes = read_s1_manifests(Path(launch["data_dir"]), paths["textvqa"])
    manifest = manifests["official_val"]
    if len(manifest["images"]) != 3166 or sum(len(im["questions"]) for im in manifest["images"]) != 5000:
        raise ValueError("Expected all 3,166 official validation images and 5,000 questions")
    shared = runtime_shared(paths, config)
    checkpoints = checkpoints_for({**launch, "arms": list(arms)}, shared, hashes)
    binding = {"stage": mode, "manifest_hash": hashes["official_val"], "shared": shared,
               "checkpoints": {arm: {**record["provenance"], "source": str(
                   Path(launch["s1_run"]) / "s1/variants" / arm / "checkpoint.pt")}
                               for arm, record in checkpoints.items()}}
    return launch, paths, config, manifest, checkpoints, binding


def bind(output, binding):
    path = output / "run.json"
    if path.exists() and read_json(path) != binding:
        raise ValueError("Run settings changed")
    write_json(path, binding)


def write_table(path, rows):
    contents = io.StringIO()
    writer = csv.DictWriter(contents, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, contents.getvalue().encode("utf-8"))


def finish(output, binding, summary, rows):
    write_json(output / "summary.json", summary)
    write_jsonl(output / "predictions.jsonl", rows)
    artifacts = {p.name: sha256(p) for p in output.iterdir() if p.is_file() and
                 p.suffix in (".csv", ".json", ".jsonl") and p.name != "COMPLETE.json"}
    write_json(output / "COMPLETE.json", {
        "stage": binding["stage"], "binding": object_hash(binding), "artifacts": artifacts,
    })
