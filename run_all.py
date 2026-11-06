"""Submit one Q-LRG job from the UIT-HPC login node."""

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


def setup_commands(run_dir, python, paths_config):
    return [
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"',
        "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1",
        "export QLRG_ENV=" + shlex.quote(str(python.parent.parent)),
        "export QLRG_PATHS_CONFIG=" + shlex.quote(paths_config),
        "(",
        "    flock 9",
        f"    if [[ ! -x {shlex.quote(str(python))} ]] || ! bash scripts/check_environment.sh; then",
        "        bash scripts/setup_hpc.sh",
        "    fi",
        f") 9>{shlex.quote(str(run_dir.parent / '.environment.lock'))}",
    ]


def write_script(run_dir, mode, commands):
    script = run_dir / f"{mode}.sbatch"
    header = "#!/usr/bin/env bash\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\nset -euo pipefail\n"
    script.write_bytes((header + "\n".join(commands) + "\n").encode("utf-8"))
    return script


def write_job(run_dir, manifest, python, paths_config, mode, images, review):
    python_cmd = [str(python), "-B", "-m", "qlrg.cli"]
    data_cmd = [*python_cmd, "--paths", paths_config]
    commands = [
        *setup_commands(run_dir, python, paths_config),
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
    return write_script(run_dir, mode, commands)


def write_s1_job(run_dir, source, python, paths_config, config):
    return write_script(run_dir, "s1", [
        *setup_commands(run_dir, python, paths_config),
        ': ${SLURM_JOB_ID:?Run this script in a Slurm GPU allocation}',
        "export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1",
        "nvidia-smi",
        "exec " + shlex.join([str(python), "-B", "-m", "qlrg.cli", "--paths", paths_config, "s1",
                             "--from-run", str(source), "--run-dir", str(run_dir), "--config", str(config)]),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full", "s1"), default="smoke")
    parser.add_argument("--from-run", type=Path, help="Completed full D0/S0 anchor for S1")
    parser.add_argument("--s1-config", type=Path, help="S1 YAML configuration")
    parser.add_argument("--resume", type=Path, help="Resume an existing S1 run")
    parser.add_argument("--review", type=Path, help="Approved grounding JSONL for full S0")
    parser.add_argument("--dry-run", action="store_true", help="Generate the script without setup or submission")
    args = parser.parse_args()
    if args.review and (args.mode != "full" or not args.review.is_file()):
        parser.error("--review requires full mode and an existing JSONL")
    if args.mode != "s1" and (args.from_run or args.s1_config or args.resume):
        parser.error("--from-run, --s1-config and --resume apply to S1")
    if args.mode == "s1":
        if args.resume:
            if args.from_run or args.s1_config or not (args.resume / "s1.sbatch").is_file():
                parser.error("--resume needs an existing S1 run and reuses its original settings")
        elif not args.from_run or not all((args.from_run / path).is_file()
                                          for path in ("manifest.json", "d0/COMPLETE.json", "s0/COMPLETE.json")):
            parser.error("S1 requires --from-run pointing to a completed D0/S0 run")
        if args.s1_config and not args.s1_config.is_file():
            parser.error("--s1-config requires an existing YAML file")
    if not args.dry_run and os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this launcher from the UIT-HPC login node")

    env = os.environ.copy()
    env["QLRG_ENV"] = str(Path(env.get("QLRG_ENV", REPO / ".venv")).expanduser().absolute())
    python = Path(env["QLRG_ENV"]) / "bin/python"
    slurm = ["bash", "-c", WITH_SLURM, "qlrg-slurm"]
    runs = REPO / "runs"
    runs.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths_config = env.get("QLRG_PATHS_CONFIG", "configs/paths_hpc.yaml")
    submit = [sys.executable, "-B", "scripts/submit.py", "pipeline", "--mode", args.mode, "--submit"]
    if args.mode == "s1":
        if args.resume:
            run_dir = args.resume.resolve()
            script = run_dir / "s1.sbatch"
        else:
            run_dir = Path(tempfile.mkdtemp(prefix=f"s1-{stamp}-", dir=runs))
            config_path = run_dir / "s1.yaml"
            config_path.write_bytes((args.s1_config or REPO / "configs/s1.yaml").read_bytes())
            script = write_s1_job(run_dir, args.from_run.resolve(), python, paths_config, config_path)
    else:
        images = 500 if args.mode == "full" else 12
        run_dir = Path(tempfile.mkdtemp(prefix=f"{args.mode}{images}-{stamp}-", dir=runs))
        manifest = run_dir / "manifest.json"
        script = write_job(run_dir, manifest, python, paths_config, args.mode, images,
                           args.review.resolve() if args.review else None)
        submit.extend(("--manifest", str(manifest)))
    submit.extend(("--run-dir", str(run_dir)))
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
