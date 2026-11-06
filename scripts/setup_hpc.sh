#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo="$PWD"
venv="${QLRG_ENV:-$repo/.venv}"
uv="${QLRG_UV:-uv}"
if [[ -z "${QLRG_UV:-}" ]] && ! command -v uv >/dev/null; then
    uv=/datastore/dungmt/textvqa/Tuan/monkeyocr_workspace/environment_and_models__phase1_verified_stack/uv_bootstrap/bin/uv
fi
command -v "$uv" >/dev/null || { echo "Set QLRG_UV to the absolute path to uv" >&2; exit 1; }
export UV_CACHE_DIR="${QLRG_UV_CACHE:-$repo/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${QLRG_UV_PYTHON:-$repo/.uv-python}"
export UV_LINK_MODE=copy

if [[ ! -x "$venv/bin/python" ]]; then
    "$uv" venv --python 3.10.22 --managed-python "$venv"
fi
python="$venv/bin/python"
"$uv" pip install --python "$python" -c requirements-hpc.lock --index https://download.pytorch.org/whl/cu130 \
    'torch==2.14.1+cu130' 'torchvision==0.29.1+cu130'
"$uv" pip install --python "$python" -c requirements-hpc.lock -r requirements.txt -e .
"$uv" pip check --python "$python"
scorer="$venv/m4c_evaluators.py"
tmp="$(mktemp "$venv/.scorer.XXXXXX")"
trap 'rm -f -- "$tmp"' EXIT
curl --fail --silent --show-error --location --retry 2 --max-time 120 \
    https://raw.githubusercontent.com/facebookresearch/mmf/72f898ec35af86567423c7a019dd1dc5175d1a5a/mmf/utils/m4c_evaluators.py \
    -o "$tmp"
printf '%s  %s\n' e10a99e15c4658f8d0d83a05cc536c835922e701171fce0016718bec021b8c0a "$tmp" | sha256sum -c
mv -- "$tmp" "$scorer"
"$uv" pip freeze --python "$python" > "$venv/requirements.resolved.txt"
bash scripts/check_environment.sh
