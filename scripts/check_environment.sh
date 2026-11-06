#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python="${QLRG_ENV:-$PWD/.venv}/bin/python"
"$python" -B - <<'PY'
import inspect
import ssl
import sys

import PIL
import torch
import torchvision
import transformers
from transformers import Qwen2_5_VLForConditionalGeneration

from qlrg.common import dataset_paths, read_config, sha256
from qlrg.evaluation import evaluator

paths = dataset_paths(read_config("configs/paths_hpc.yaml"))
model = read_config("configs/model_qwen.yaml")
snapshot = paths["hf_hub"] / ("models--" + model["id"].replace("/", "--")) / "snapshots" / model["revision"]
for path in (
    paths["textvqa"] / "TextVQA_0.5.1_train.json", paths["textocr"], snapshot / "config.json",
    snapshot / "tokenizer.json", snapshot / "preprocessor_config.json",
):
    if not path.is_file():
        raise FileNotFoundError(path)
weights = list(snapshot.glob("*.safetensors"))
if not weights or not all(weight.is_file() for weight in weights):
    raise RuntimeError(f"Missing model weights under {snapshot}")
assert sys.version_info[:3] == (3, 10, 22), sys.version
assert torch.__version__ == "2.14.1+cu130", torch.__version__
assert torchvision.__version__ == "0.29.1+cu130", torchvision.__version__
assert transformers.__version__ == model["transformers"], transformers.__version__
assert sha256(inspect.getfile(Qwen2_5_VLForConditionalGeneration)) == model["source_sha256"]
assert PIL.__version__ == "11.3.0", PIL.__version__
assert abs(evaluator().eval_pred_list([{"pred_answer": "test", "gt_answers": ["test"] * 10}]) - 1.0) < 1e-9
print(sys.version.split()[0], ssl.OPENSSL_VERSION, torch.__version__, torchvision.__version__, transformers.__version__)
print(f"CPU preflight passed. Model snapshot {snapshot}")
PY
