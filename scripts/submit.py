"""Print a concrete Slurm command, or submit it after an explicit --submit."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("d0", "s0", "pipeline"))
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--d0-run", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--memory")
    parser.add_argument("--time")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.stage == "s0" and args.mode == "full" and args.review is None:
        parser.error("Full S0 requires --review")
    if args.stage != "s0" and (args.review or args.d0_run):
        parser.error("--review and --d0-run only apply to S0")
    if args.stage == "pipeline" and args.config:
        parser.error("The generated pipeline uses default stage configs")
    if args.stage != "pipeline" and not args.manifest.is_file():
        parser.error("Manifest must exist before preparing a Slurm command")

    repo = Path(__file__).resolve().parents[1]
    with (repo / "configs/slurm_hpc.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    resources = config[args.mode] if args.mode == "smoke" or args.stage == "pipeline" else {}
    cpus = args.cpus or resources.get("cpus")
    memory = args.memory or resources.get("memory")
    time = args.time or resources.get("time")
    if not cpus or not memory or not time:
        parser.error("Full jobs require explicit --cpus, --memory and --time")
    if not 1 <= cpus <= 8:
        parser.error("mig2-35 supports 1 to 8 CPUs per job")
    run_dir = args.run_dir.resolve()
    logs = run_dir / "logs"
    if args.stage == "pipeline":
        script = run_dir / f"{args.mode}.sbatch"
        if not script.is_file():
            parser.error("Generate the pipeline with run_all.py first")
        job_name = f"qlrg-{args.mode}"
    else:
        template = f"d0_{args.mode}.sbatch" if args.stage == "d0" else "s0_oracle.sbatch"
        script = repo / "slurm" / template
        job_name = f"qlrg-{args.stage}-{args.mode}"
    command = [
        "sbatch", "--parsable", f"--chdir={repo}",
        f"--job-name={job_name}",
        f"--account={config['account']}", f"--partition={config['partition']}",
        f"--qos={config['qos']}", f"--gres={config['gres']}",
        f"--cpus-per-task={cpus}", f"--mem={memory}", f"--time={time}",
        f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err",
        str(script), "--mode", args.mode,
        "--manifest", str(args.manifest.resolve()), "--run-dir", str(run_dir),
    ]
    for flag, path in (("--review", args.review), ("--d0-run", args.d0_run), ("--config", args.config)):
        if path is not None:
            command.extend((flag, str(path.resolve())))
    print(shlex.join(command), flush=True)
    if args.submit:
        if args.stage == "s0":
            from qlrg.common import read_json, sha256
            from qlrg.experiment import verified_complete
            from qlrg.grounding import read_review

            d0_path = args.d0_run or run_dir / "d0"
            d0 = verified_complete(d0_path, "d0")["shared"]
            if d0["manifest_sha256"] != sha256(args.manifest) or d0["mode"] != args.mode:
                parser.error("S0 requires a completed D0 for the same manifest and mode")
            if args.review:
                read_review(args.review, read_json(args.manifest))
        logs.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
