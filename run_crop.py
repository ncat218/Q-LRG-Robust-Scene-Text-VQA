"""Launch the standalone crop oracle evaluation on UIT-HPC."""

import argparse
import json
import math
import os
import shlex
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from run_all import REPO, WITH_SLURM, write_script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(os.environ.get("QLRG_ENV", REPO / ".venv")))
    parser.add_argument("--paths", type=Path, default=Path(os.environ.get("QLRG_PATHS_CONFIG", "configs/paths_hpc.yaml")))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_qwen.yaml"))
    parser.add_argument("--review", type=Path)
    parser.add_argument("--detections", type=Path)
    parser.add_argument("--padding", type=float, default=0.1)
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--time")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.padding) or not 0 <= args.padding <= 1:
        parser.error("--padding must be in [0, 1]")
    if not math.isfinite(args.zoom) or not 1 <= args.zoom <= 4:
        parser.error("--zoom must be in [1, 4]")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    if args.worker_run_dir:
        from scripts.crop_oracle import run

        args.run_dir = args.worker_run_dir
        run(args)
        return

    if os.environ.get("SLURM_JOB_ID"):
        parser.error("Launch the crop job from the UIT-HPC login node")
    source = args.from_run.resolve()
    if not (source / "manifest.json").is_file() or not (source / "s0" / "COMPLETE.json").is_file():
        parser.error("--from-run must contain manifest.json and a completed s0 stage")
    env_dir = args.env.expanduser().resolve()
    python = env_dir / "bin/python"
    if not python.is_file():
        parser.error("--env must point to the existing Q-LRG environment")
    for path in (args.paths, args.model_config, args.review, args.detections):
        if path is not None and not path.is_file():
            parser.error(f"Input file does not exist: {path}")

    runs = REPO / "runs"
    runs.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(tempfile.mkdtemp(prefix=f"crop-oracle-{timestamp}-", dir=runs))
    logs = run_dir / "logs"
    command = [
        str(python), "-B", str(REPO / "run_crop.py"), "--worker-run-dir", str(run_dir),
        "--from-run", str(source), "--paths", str(args.paths.resolve()),
        "--model-config", str(args.model_config.resolve()), "--padding", str(args.padding),
        "--zoom", str(args.zoom), "--seed", str(args.seed),
    ]
    for flag, path in (("--review", args.review), ("--detections", args.detections)):
        if path is not None:
            command.extend((flag, str(path.resolve())))
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    script = write_script(run_dir, "crop-oracle", [
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"',
        "export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1",
        "export QLRG_ENV=" + shlex.quote(str(env_dir)),
        "export QLRG_PATHS_CONFIG=" + shlex.quote(str(args.paths.resolve())),
        "bash scripts/check_environment.sh",
        "nvidia-smi",
        "exec " + shlex.join(command),
    ])
    resources = json.loads((REPO / "configs/slurm_hpc.json").read_text(encoding="utf-8"))
    smoke = resources["smoke"]
    submit = [
        "sbatch", "--parsable", f"--chdir={REPO}", "--job-name=qlrg-crop",
        f"--account={resources['account']}", f"--partition={resources['partition']}",
        f"--qos={resources['qos']}", f"--gres={resources['gres']}", f"--cpus-per-task={smoke['cpus']}",
        f"--mem={smoke['memory']}", f"--time={args.time or smoke['time']}",
        f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err", str(script),
    ]
    print(f"Slurm script: {script}\nLogs: {logs}", flush=True)
    print(shlex.join(submit), flush=True)
    if not args.dry_run:
        logs.mkdir(exist_ok=True)
        subprocess.run(["bash", "-c", WITH_SLURM, "qlrg-slurm", *submit], cwd=REPO, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None
