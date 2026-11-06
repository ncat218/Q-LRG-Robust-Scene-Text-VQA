"""Prepare Q-LRG and submit one D0/S0 smoke job from the UIT-HPC login node."""

import argparse
import os
import shlex
import subprocess
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


def write_job(run_dir, python):
    commands = []
    for stage in ("d0", "s0"):
        commands.extend([
            f'bash slurm/job.sh {stage} "$@"',
            shlex.join([str(python), "-B", "-m", "qlrg.cli", "verify-run",
                        "--stage", stage, "--run-dir", str(run_dir)]),
        ])
    script = run_dir / "smoke.sbatch"
    header = "#!/usr/bin/env bash\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\nset -euo pipefail\n"
    script.write_bytes((header + "\n".join(commands) + "\n").encode("utf-8"))
    return script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Generate the script without setup or submission")
    args = parser.parse_args()
    if not args.dry_run and os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this launcher from the UIT-HPC login node")

    env = os.environ.copy()
    env["QLRG_ENV"] = str(Path(env.get("QLRG_ENV", REPO / ".venv")).expanduser().absolute())
    python = Path(env["QLRG_ENV"]) / "bin/python"
    slurm = ["bash", "-c", WITH_SLURM, "qlrg-slurm"]
    if not args.dry_run:
        subprocess.run([*slurm, "sbatch", "--version"], cwd=REPO, env=env, check=True)
        ready = python.is_file() and subprocess.run(
            ["bash", "scripts/check_environment.sh"], cwd=REPO, env=env
        ).returncode == 0
        if not ready:
            subprocess.run(["bash", "scripts/setup_hpc.sh"], cwd=REPO, env=env, check=True)

    runs = REPO / "runs"
    runs.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(tempfile.mkdtemp(prefix=f"smoke12-{stamp}-", dir=runs))
    manifest = run_dir / "manifest.json"
    script = write_job(run_dir, python)
    build = [
        str(python), "-B", "-m", "qlrg.cli", "--paths",
        env.get("QLRG_PATHS_CONFIG", "configs/paths_hpc.yaml"),
        "build-manifest", "--images", "12", "--output", str(manifest),
    ]
    submit = [
        str(python), "-B", "scripts/submit.py", "pipeline", "--mode", "smoke",
        "--manifest", str(manifest), "--run-dir", str(run_dir), "--submit",
    ]
    print(f"Slurm script: {script}\nLogs: {run_dir / 'logs'}", flush=True)
    if args.dry_run:
        print(shlex.join(build))
        print(shlex.join(submit))
        return
    subprocess.run(build, cwd=REPO, env=env, check=True)
    subprocess.run([*slurm, *submit], cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None
