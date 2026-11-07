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


def write_job(run_dir, manifest, python, paths_config, mode, images, review, s0_config=None):
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
    if s0_config is not None:
        s0_options.extend(("--config", str(s0_config)))
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


def write_s1_job(run_dir, source, python, paths_config, config, data_dir=None, audit=None):
    inputs = ["--from-run", str(source)] if source else ["--data-dir", str(data_dir), "--baseline-audit", str(audit)]
    return write_script(run_dir, "s1", [
        *setup_commands(run_dir, python, paths_config),
        ': ${SLURM_JOB_ID:?Run this script in a Slurm GPU allocation}',
        "export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1",
        "nvidia-smi",
        "exec " + shlex.join([str(python), "-B", "-m", "qlrg.cli", "--paths", paths_config, "s1",
                             *inputs, "--run-dir", str(run_dir), "--config", str(config)]),
    ])



def write_audit_or_evaluation_job(run_dir, mode, source, python, paths_config, pilot=None):
    command = [str(python), "-B", "-m", "qlrg.cli", "--paths", paths_config]
    stage = "audit-a0" if mode == "audit" else "evaluate-s1"
    commands = [
        *setup_commands(run_dir, python, paths_config),
        ': ${SLURM_JOB_ID:?Run this script in a Slurm GPU allocation}',
        "export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1",
        "nvidia-smi",
        shlex.join([*command, stage, "--from-run", str(source), "--run-dir", str(run_dir),
                    *(["--pilot-run", str(pilot)] if pilot else [])]),
    ]
    if mode == "audit":
        commands.append(shlex.join([*command, "build-s1-data", "--output", str(run_dir / "datasets")]))
        files = ["audit/run.json", "audit/COMPLETE.json", "audit/manifest.json",
                 "audit/summary.json", "audit/diagnostics.json", "audit/predictions.jsonl",
                 "audit/historical_comparisons.json", "datasets/COMPLETE.json",
                 "datasets/train_inner.json", "datasets/val_inner.json", "datasets/official_val.json"]
        if pilot:
            files.extend(("audit/pilot_train_manifest.json", "audit/pilot_split_audit.json"))
        package = [
            "package_review() {",
            "    local candidates=(" + shlex.join(files) + ")",
            "    local present=() path",
            '    for path in "${candidates[@]}"; do',
            "        if [[ -f " + shlex.quote(str(run_dir)) + '/"$path" ]]; then present+=("$path"); fi',
            "    done",
            '    if (( ${#present[@]} )); then',
            "        " + shlex.join(["tar", "-czf", str(run_dir / "review.tar.gz"), "-C", str(run_dir)]) + ' "${present[@]}"',
            "    fi",
            "}",
            "trap package_review EXIT",
        ]
        commands[-2:-2] = package
    return write_script(run_dir, mode, commands)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full", "s1", "audit", "s1-eval"), default="smoke")
    parser.add_argument("--pilot-run", type=Path, help="Existing S1 run whose training split and A0 are audited")
    parser.add_argument("--data-dir", type=Path, help="Validated independent S1 manifests")
    parser.add_argument("--baseline-audit", type=Path, help="Passing run's audit/ directory for expanded S1")
    parser.add_argument("--time", help="Slurm walltime override for an explicitly requested allocation")
    parser.add_argument("--from-run", type=Path, help="Completed full D0/S0 anchor for S1")
    parser.add_argument("--s0-config", type=Path, help="S0 YAML for a diagnostic or mask sensitivity run")
    parser.add_argument("--s1-config", type=Path, help="S1 YAML configuration")
    parser.add_argument("--resume", type=Path, help="Resume an existing S1 run")
    parser.add_argument("--review", type=Path, help="Approved grounding JSONL for full S0")
    parser.add_argument("--dry-run", action="store_true", help="Generate the script without setup or submission")
    args = parser.parse_args()
    if args.pilot_run and (args.mode != "audit" or args.resume or not (args.pilot_run / "s1/COMPLETE.json").is_file()):
        parser.error("--pilot-run requires audit mode and an existing completed S1 run")
    if args.review and (args.mode != "full" or not args.review.is_file()):
        parser.error("--review requires full mode and an existing JSONL")
    if args.s0_config and (args.mode not in {"smoke", "full"} or not args.s0_config.is_file()):
        parser.error("--s0-config requires smoke/full mode and an existing YAML")
    if args.mode != "s1" and (args.data_dir or args.baseline_audit or args.s1_config):
        parser.error("--data-dir, --baseline-audit and --s1-config apply to S1")
    if args.resume:
        if args.mode not in {"s1", "audit", "s1-eval"} or any(
            (args.from_run, args.data_dir, args.baseline_audit, args.s1_config)
        ) or not (args.resume / f"{args.mode}.sbatch").is_file():
            parser.error("--resume requires the existing run and its original mode")
    elif args.mode == "s1":
        if bool(args.from_run) == bool(args.data_dir):
            parser.error("Choose --from-run for the pilot, or --data-dir for expanded S1")
        if args.data_dir:
            if not args.baseline_audit or not (args.data_dir / "COMPLETE.json").is_file():
                parser.error("Expanded S1 needs --baseline-audit and complete S1 manifests")
            if not (args.baseline_audit / "COMPLETE.json").is_file():
                parser.error("Finish the A0 audit first")
        elif not all((args.from_run / path).is_file()
                     for path in ("manifest.json", "d0/COMPLETE.json", "s0/COMPLETE.json")):
            parser.error("Pilot S1 requires a completed D0/S0 run")
    elif args.mode in {"audit", "s1-eval"}:
        required = "manifest.json" if args.mode == "audit" else "s1/COMPLETE.json"
        if not args.from_run or not (args.from_run / required).is_file():
            parser.error(f"{args.mode} requires --from-run containing {required}")
    elif args.from_run:
        parser.error("--from-run applies to audit, s1 and s1-eval")
    if args.s1_config and not args.s1_config.is_file():
        parser.error("--s1-config requires an existing configuration")
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
    if args.mode in {"s1", "audit", "s1-eval"}:
        if args.resume:
            run_dir = args.resume.resolve()
            script = run_dir / f"{args.mode}.sbatch"
        else:
            run_dir = Path(tempfile.mkdtemp(prefix=f"{args.mode}-{stamp}-", dir=runs))
            if args.mode == "s1":
                config_path = run_dir / "s1.yaml"
                default = "configs/s1_expanded.json" if args.data_dir else "configs/s1.yaml"
                config_path.write_bytes((args.s1_config or REPO / default).read_bytes())
                script = write_s1_job(run_dir, args.from_run.resolve() if args.from_run else None,
                                      python, paths_config, config_path,
                                      args.data_dir.resolve() if args.data_dir else None,
                                      args.baseline_audit.resolve() if args.baseline_audit else None)
            else:
                script = write_audit_or_evaluation_job(run_dir, args.mode, args.from_run.resolve(), python, paths_config,
                                                        args.pilot_run.resolve() if args.pilot_run else None)
    else:
        images = 500 if args.mode == "full" else 12
        run_dir = Path(tempfile.mkdtemp(prefix=f"{args.mode}{images}-{stamp}-", dir=runs))
        manifest = run_dir / "manifest.json"
        config_path = None
        if args.s0_config:
            config_path = run_dir / "s0.yaml"
            config_path.write_bytes(args.s0_config.read_bytes())
        script = write_job(run_dir, manifest, python, paths_config, args.mode, images,
                           args.review.resolve() if args.review else None, config_path)
        submit.extend(("--manifest", str(manifest)))
    submit.extend(("--run-dir", str(run_dir)))
    if args.time:
        submit.extend(("--time", args.time))
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
