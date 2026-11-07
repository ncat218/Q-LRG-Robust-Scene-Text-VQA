"""S1 storage settings and compatibility with pre-quota-recovery runs."""

import os
from pathlib import Path

from .common import read_json


def feature_cache_mode():
    mode = os.environ.get("QLRG_FEATURE_CACHE_MODE", "write")
    if mode not in ("write", "read-only"):
        raise ValueError("QLRG_FEATURE_CACHE_MODE must be write or read-only")
    return mode


def checkpoint_steps(config, example_count):
    steps = (example_count + config["accumulate"] - 1) // config["accumulate"]
    override = os.environ.get("QLRG_CHECKPOINT_SAVES")
    saves = int(override) if override is not None else config.get("checkpoint_saves")
    if saves is None:
        return set(range(config["checkpoint_every"], steps, config["checkpoint_every"])) | {steps}
    if saves < 1:
        raise ValueError("The checkpoint save count must be positive")
    saves = min(saves, steps)
    return {(steps * i + saves - 1) // saves for i in range(1, saves + 1)}


def _storage_transition(current):
    path = Path(__file__).resolve().parents[1] / "configs" / "s1_storage_compat.json"
    transition = read_json(path)
    return transition if current["source_hash"] == transition["after_source"] else None


def accepts_prior_audit(audited_path, current):
    transition = _storage_transition(current)
    return transition is not None and audited_path == transition["before_a0_path"]


def resume_runtime(saved, current):
    if saved == current:
        return current
    transition = _storage_transition(current)
    if transition is not None and saved == {
        **current, "source_hash": transition["before_source"], "a0_path_hash": transition["before_a0_path"],
    }:
        return saved
    raise ValueError("S1 runtime changed; use the original or the verified storage-only upgrade")
