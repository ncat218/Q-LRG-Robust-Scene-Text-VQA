"""Submit one D0/S0 job from the UIT-HPC login node."""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent
WITH_SLURM = """\
set -e
if ! command -v sbatch >/dev/null; then
    . /etc/profile.d/modules.sh
    module load slurm/slurm/25.05
fi
exec "$@"
"""


def write_job(run_dir, manifest, python, paths_config, mode, images, review):
    python_cmd = [str(python), "-B", "-m", "qlrg.cli"]
    data_cmd = [*python_cmd, "--paths", paths_config]
    commands = [
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"',
        "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1",
        "(",
        "    flock 9",
        f"    if [[ ! -x {shlex.quote(str(python))} ]] || ! bash scripts/check_environment.sh; then",
        "        bash scripts/setup_hpc.sh",
        "    fi",
        f") 9>{shlex.quote(str(run_dir.parent / '.environment.lock'))}",
        f"if [[ -f {shlex.quote(str(manifest))} ]]; then",
        "    " + shlex.join([*data_cmd, "validate-dataset", "--manifest", str(manifest)]),
        "else",
        "    " + shlex.join([*data_cmd, "build-manifest", "--images", str(images), "--output", str(manifest)]),
        "fi",
    ]
    s0_options = []
    if mode == "full":
        s0_options = ["--review", str(review)] if review else ["--provisional-grounding"]
    for stage in ("d0", "s0"):
        stage_command = f'bash slurm/job.sh {stage} "$@"'
        if stage == "s0" and s0_options:
            stage_command += " " + shlex.join(s0_options)
        commands.extend([
            stage_command,
            shlex.join([*python_cmd, "verify-run",
                        "--stage", stage, "--run-dir", str(run_dir)]),
        ])
    script = run_dir / f"{mode}.sbatch"
    header = "#!/usr/bin/env bash\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\nset -euo pipefail\n"
    script.write_bytes((header + "\n".join(commands) + "\n").encode("utf-8"))
    return script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--review", type=Path, help="Approved grounding JSONL for full S0")
    parser.add_argument("--dry-run", action="store_true", help="Generate the script without setup or submission")
    args = parser.parse_args()
    if args.review and (args.mode != "full" or not args.review.is_file()):
        parser.error("--review requires full mode and an existing JSONL")
    if not args.dry_run and os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this launcher from the UIT-HPC login node")

    env = os.environ.copy()
    env["QLRG_ENV"] = str(Path(env.get("QLRG_ENV", REPO / ".venv")).expanduser().absolute())
    python = Path(env["QLRG_ENV"]) / "bin/python"
    slurm = ["bash", "-c", WITH_SLURM, "qlrg-slurm"]
    runs = REPO / "runs"
    runs.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    images = 500 if args.mode == "full" else 12
    run_dir = Path(tempfile.mkdtemp(prefix=f"{args.mode}{images}-{stamp}-", dir=runs))
    manifest = run_dir / "manifest.json"
    script = write_job(
        run_dir, manifest, python,
        env.get("QLRG_PATHS_CONFIG", "configs/paths_hpc.yaml"),
        args.mode, images, args.review.resolve() if args.review else None,
    )
    submit = [
        sys.executable, "-B", "scripts/submit.py", "pipeline", "--mode", args.mode,
        "--manifest", str(manifest), "--run-dir", str(run_dir), "--submit",
    ]
    if args.mode == "full":
        grounding = "reviewed JSONL" if args.review else "provisional OCR, up to 150 images"
        print(f"Full D0/S0 evaluation on {images} images. O3-O5 grounding: {grounding}.", flush=True)
    print(f"Slurm script: {script}\nLogs: {run_dir / 'logs'}", flush=True)
    if args.dry_run:
        print(shlex.join(submit))
        return
    subprocess.run([*slurm, *submit], cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None
