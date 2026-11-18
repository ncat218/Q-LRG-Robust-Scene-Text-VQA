"""Prepare an isolated, resumable four-arm TextVQA evaluation on UIT-HPC."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from run_all import WITH_SLURM, write_script


ROOT = Path(__file__).resolve().parent
FILES = ("final_evaluate.py", "final_review.py", "compute_cost.py", "followup_common.py")


def slurm(command):
    return subprocess.check_output(["bash", "-c", WITH_SLURM, "qlrg-final", *command], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-run", type=Path)
    parser.add_argument("--pre-run", type=Path)
    parser.add_argument("--env", type=Path)
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--split", choices=("val_inner", "official_val"), default="official_val")
    parser.add_argument("--cost-images", type=int, default=40)
    parser.add_argument("--flops-images", type=int, default=10)
    parser.add_argument("--after-job", help="Run only after this training job succeeds")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--time", default="06:00:00")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_ID"):
        parser.error("Submit from the login node")
    if args.after_job and not args.after_job.isdecimal():
        parser.error("--after-job must be a numeric Slurm ID")
    if not 1 <= args.flops_images <= args.cost_images <= 200:
        parser.error("Use 1 <= flops-images <= cost-images <= 200")
    if args.resume:
        output = args.resume.resolve()
        launch = json.loads((output / "launch.json").read_text())
    else:
        if any(value is None for value in (args.s1_run, args.pre_run, args.env, args.paths)):
            parser.error("Provide --s1-run, --pre-run, --env and --paths")
        source, pre, env, paths = (value.resolve() for value in (args.s1_run, args.pre_run, args.env, args.paths))
        for path in (source / "s1/COMPLETE.json", pre / "run.json", pre / "pre_11305.py", env / "bin/python", paths):
            if not path.is_file():
                raise FileNotFoundError(path)
        if not (pre / "COMPLETE.json").exists() and not args.after_job:
            parser.error("Pre training is incomplete. Supply --after-job or wait for completion")
        runs = ROOT / "runs"
        runs.mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=f"final-{args.split}-", dir=runs))
        # Reuse the exact pinned Qwen/QLRG package from the successful Pre training allocation.
        shutil.copytree(pre / "code", output / "code")
        (output / "logs").mkdir()
        for name in FILES:
            shutil.copy2(ROOT / "scripts" / name, output / "code/scripts" / name)
        shutil.copy2(ROOT / "run_final_demo.py", output / "code/run_final_demo.py")
        launch = {"s1_run": str(source), "pre_run": str(pre), "env": str(env), "paths": str(paths),
                  "split": args.split, "seed": 42, "cost_images": args.cost_images,
                  "flops_images": args.flops_images, "training_dependency": args.after_job,
                  "protocol": "A0,S1b,Pre,A2; CLEAN,LR4; two-epoch checkpoints; paired image-cluster CI"}
        (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    resources = json.loads((ROOT / "configs/slurm_hpc.json").read_text())
    if args.submit:
        for job_file in output.glob("job-id*.txt"):
            job_id = job_file.read_text().strip().split(";")[0]
            active = slurm(["squeue", "-h", "-j", job_id, "-o", "%i %T"])
            if active:
                raise RuntimeError(f"Evaluation already submitted: {active}")
        if (output / "COMPLETE.json").exists():
            raise RuntimeError("Evaluation is already complete")
    python = str(Path(launch["env"]) / "bin/python")
    script = write_script(output, "final-study", [
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"',
        'export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false',
        "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1",
        "export QLRG_FEATURE_CACHE_MODE=read-only",
        "export QLRG_ENV=" + shlex.quote(launch["env"]),
        "export QLRG_PATHS_CONFIG=" + shlex.quote(launch["paths"]),
        "bash scripts/check_environment.sh",
        "exec " + shlex.join([python, "-B", "-m", "scripts.final_evaluate", str(output)]),
    ])
    command = ["sbatch", "--parsable", f"--chdir={output / 'code'}", "--job-name=qlrg-final-eval",
               f"--account={resources['account']}", f"--partition={resources['partition']}",
               f"--qos={resources['qos']}", f"--gres={resources['gres']}", "--cpus-per-task=8", "--mem=48G",
               f"--time={args.time}", f"--output={output}/logs/%x-%j.out", f"--error={output}/logs/%x-%j.err"]
    dependency = launch.get("training_dependency")
    if dependency and not (Path(launch["pre_run"]) / "COMPLETE.json").is_file():
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(script))
    subprocess.run(["bash", "-n", str(script)], check=True)
    print(f"Run: {output}", flush=True)
    if args.submit:
        job = slurm(command)
        (output / f"job-id-{job.split(';')[0]}.txt").write_text(job + "\n")
        print(f"Submitted {job}", flush=True)
    else:
        print(shlex.join(command))


if __name__ == "__main__":
    main()
