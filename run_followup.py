"""Submit independent restoration and crop benchmarks using the locked S1 run."""

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from run_all import REPO, WITH_SLURM, write_script


JOBS = {"restoration": "qlrg-blur-eval", "crop": "qlrg-crop-selectors"}


def submit(mode, settings, source, dry_run=False):
    for key in ("env", "paths", "data_dir", "s1_run"):
        if not settings.get(key):
            raise ValueError(f"Missing {key} in {source / 'launch.json'}")
    python = Path(settings["env"]) / "bin/python"
    required = [python, Path(settings["paths"]), Path(settings["data_dir"]) / "official_val.json",
                Path(settings["s1_run"]) / "s1/COMPLETE.json"]
    if mode == "crop":
        required.extend([source / "COMPLETE.json", source / "run.json"])
        if not (source / "sdrpn-proposals").is_dir():
            raise FileNotFoundError(source / "sdrpn-proposals")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    resources = json.loads((REPO / "configs/slurm_hpc.json").read_text(encoding="utf-8"))
    (REPO / "runs").mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"{mode}-benchmark-", dir=REPO / "runs"))
    launch = {**settings, "repo": str(REPO), "source": str(source), "study": mode}
    (run / "launch.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    script = write_script(run, mode, [
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"',
        "export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1",
        "export QLRG_ENV=" + shlex.quote(settings["env"]),
        "export QLRG_PATHS_CONFIG=" + shlex.quote(settings["paths"]),
        "bash scripts/check_environment.sh",
        "nvidia-smi",
        "exec " + shlex.join([str(python), "-B", "-m", f"scripts.{mode}_followup", str(run)]),
    ])
    logs = run / "logs"
    logs.mkdir()
    command = ["sbatch", "--parsable", f"--chdir={REPO}", f"--job-name={JOBS[mode]}",
               f"--account={resources['account']}", f"--partition={resources['partition']}",
               f"--qos={resources['qos']}", f"--gres={resources['gres']}", "--cpus-per-task=4",
               "--mem=48G", "--time=16:00:00", f"--output={logs}/%x-%j.out",
               f"--error={logs}/%x-%j.err", str(script)]
    print(f"{mode}: {run}", flush=True)
    if dry_run:
        print(shlex.join(command))
        return
    result = subprocess.check_output(["bash", "-c", WITH_SLURM, "qlrg-slurm", *command], text=True).strip()
    (run / "job-id.txt").write_text(result + "\n", encoding="utf-8")
    print(f"{mode} job: {result}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("both", *JOBS), default="both")
    parser.add_argument("--from-confirm", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_ID"):
        parser.error("Launch from the Slurm login node")
    source = args.from_confirm.resolve()
    settings = json.loads((source / "launch.json").read_text(encoding="utf-8"))
    if not settings.get("confirm_a2_fusion") or settings["split"] != "official_val":
        parser.error("Use the locked A2 official validation run as the source")
    for mode in JOBS if args.mode == "both" else (args.mode,):
        submit(mode, settings, source, args.dry_run)


if __name__ == "__main__":
    main()
