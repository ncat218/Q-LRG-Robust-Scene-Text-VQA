"""Configuration, provenance and atomic output helpers."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import yaml


def read_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return result


def dataset_paths(config):
    return {
        key: Path(os.environ.get(f"QLRG_{key.upper()}", value)).expanduser().resolve()
        for key, value in config.items()
    }


def sha256(path):
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def canonical_bytes(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def object_hash(obj):
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, obj):
    atomic_write(path, canonical_bytes(obj) + b"\n")


def write_jsonl(path, rows):
    atomic_write(path, b"".join(canonical_bytes(row) + b"\n" for row in rows))


def code_hash():
    return object_hash({
        path.name: sha256(path) for path in Path(__file__).parent.glob("*.py")
    })
