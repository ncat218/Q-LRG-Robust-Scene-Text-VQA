#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${QLRG_ENV:-$PWD/.venv}/bin/python" -B -m qlrg.cli --paths "${QLRG_PATHS_CONFIG:-configs/paths_hpc.yaml}" s0 "$@"
