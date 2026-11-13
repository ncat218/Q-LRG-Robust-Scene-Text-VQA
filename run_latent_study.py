"""Prepare or submit the first controlled latent restoration experiments."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from run_all import REPO, WITH_SLURM, write_script


def submit(output, name, commands, resources, walltime, execute):
    script = write_script(output, "latent", commands)
    logs = output / "logs"
    logs.mkdir()
    command = ["sbatch", "--parsable", f"--chdir={REPO}", f"--job-name=qlrg-latent-{name}",
               f"--account={resources['account']}", f"--partition={resources['partition']}",
               f"--qos={resources['qos']}", f"--gres={resources['gres']}", "--cpus-per-task=4",
               "--mem=48G", f"--time={walltime}", f"--output={logs}/%x-%j.out",
               f"--error={logs}/%x-%j.err", str(script)]
    print(output, flush=True)
    if execute:
        job = subprocess.check_output(["bash", "-c", WITH_SLURM, "qlrg-slurm", *command],
                                      cwd=REPO, text=True).strip()
        (output / "job-id.txt").write_text(job + "\n")
        print(f"{name}: {job}", flush=True)
    else:
        print(shlex.join(command))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-run", type=Path, required=True)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--train-images", type=int, default=4000)
    parser.add_argument("--eval-images", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time", help="Walltime per Slurm allocation")
    parser.add_argument("--grouped", action="store_true", help="Run all selected methods sequentially on one GPU")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    from scripts.latent_methods import METHODS

    if not set(args.methods) <= set(METHODS) or len(args.methods) != len(set(args.methods)):
        parser.error(f"Choose distinct methods from {', '.join(METHODS)}")
    if min(args.train_images, args.epochs, args.eval_images or args.train_images) < 1 or args.eval_images == 0:
        parser.error("Image counts and epochs must be positive")
    python = args.env.resolve() / "bin/python"
    source = args.s1_run.resolve()
    if not python.is_file() or not (source / "s1/COMPLETE.json").is_file():
        parser.error("Provide the existing HPC environment and completed expanded S1 run")
    resources = json.loads((REPO / "configs/slurm_hpc.json").read_text())
    runs = REPO / "runs"
    runs.mkdir(exist_ok=True)
    setup = [
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"',
        "export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1",
        "export QLRG_ENV=" + shlex.quote(str(args.env.resolve())),
        "bash scripts/check_environment.sh",
    ]
    group = Path(tempfile.mkdtemp(prefix="latent-group-", dir=runs)) if args.grouped else None
    commands = [*setup, "status=0"]
    for method in args.methods:
        if group is None:
            output = Path(tempfile.mkdtemp(prefix=f"latent-{method}-", dir=runs))
        else:
            output = group / method
            output.mkdir()
        launch = {"s1_run": str(source), "method": method, "seed": args.seed, "epochs": args.epochs,
                  "train_images": args.train_images, "eval_images": args.eval_images}
        (output / "launch.json").write_text(json.dumps(launch) + "\n")
        worker = shlex.join([str(python), "-B", "-m", "scripts.latent_study", str(output)])
        if group is None:
            submit(output, method, [*setup, "exec " + worker], resources, args.time or "06:00:00", args.submit)
        else:
            log = shlex.quote(str(output / "worker.log"))
            marker = shlex.quote(str(output / "COMPLETE.json"))
            commands.extend([
                f'if [[ -s {marker} ]]; then echo "Skip completed {method}"; else',
                f'    echo "Starting {method}"',
                f'    if {worker} >>{log} 2>&1 && [[ -s {marker} ]]; then',
                f'        echo "Completed {method}"',
                "    else",
                f'        echo "Failed {method}. See {log}" >&2',
                "        status=1",
                "    fi",
                "fi",
            ])
    if group is not None:
        submit(group, "group", [*commands, 'exit "$status"'], resources, args.time or "2-00:00:00", args.submit)


if __name__ == "__main__":
    main()
