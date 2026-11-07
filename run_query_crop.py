"""Submit the query-selected crop evaluation from the UIT-HPC login node."""

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
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--env", type=Path)
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--selector", choices=("sdrpn", "clip"), help="Default for new runs is sdrpn")
    parser.add_argument("--sdrpn-env", type=Path, help="Existing SD-RPN environment with Transformers 4.51.3")
    parser.add_argument("--sdrpn-vendor", type=Path, help="Directory containing qwen_src")
    parser.add_argument("--sdrpn-checkpoint", type=Path, help="Local SD-RPN Qwen2.5-VL-3B snapshot")
    parser.add_argument("--selector-cache", type=Path)
    parser.add_argument("--s1-run", type=Path)
    parser.add_argument("--arm", choices=("S1a", "S1b"))
    parser.add_argument("--split", choices=("val_inner", "official_val"))
    parser.add_argument("--images", type=int)
    parser.add_argument("--crop-fraction", type=float)
    parser.add_argument("--zoom", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--time")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_run_dir is not None:
        from scripts.query_crop import run

        run(args.worker_run_dir)
        return
    if os.environ.get("SLURM_JOB_ID"):
        parser.error("Launch from the UIT-HPC login node")

    sdrpn_options = ("sdrpn_env", "sdrpn_vendor", "sdrpn_checkpoint")
    settings = ("data_dir", "env", "paths", "selector", "selector_cache", "s1_run", "arm", "split",
                "images", "crop_fraction", "zoom", "seed", *sdrpn_options)
    if args.resume is not None:
        if any(getattr(args, name) is not None for name in settings):
            parser.error("--resume reuses the original settings. Only --time and --dry-run may be overridden.")
        run_dir = args.resume.resolve()
        launch = json.loads((run_dir / "launch.json").read_text(encoding="utf-8"))
        if Path(launch["repo"]) != REPO or (run_dir / "COMPLETE.json").exists():
            parser.error("Resume an unfinished run from its original checkout")
    else:
        if args.data_dir is None:
            parser.error("--data-dir is required")
        if (args.s1_run is None) != (args.arm is None):
            parser.error("Provide --s1-run and --arm together")
        split = args.split or "val_inner"
        if split == "official_val" and (args.s1_run is None or args.images is not None):
            parser.error("Use the full official validation split with a locked S1 checkpoint")
        if args.images is not None and args.images < 1:
            parser.error("--images must be positive")
        selector = args.selector or "sdrpn"
        sdrpn_paths = {name: getattr(args, name) for name in sdrpn_options}
        if selector == "sdrpn":
            if any(path is None for path in sdrpn_paths.values()):
                parser.error("SD-RPN requires --sdrpn-env, --sdrpn-vendor and --sdrpn-checkpoint")
            if args.crop_fraction is not None or args.selector_cache is not None:
                parser.error("--crop-fraction and --selector-cache apply to --selector clip")
        elif any(path is not None for path in sdrpn_paths.values()):
            parser.error("SD-RPN paths apply to --selector sdrpn")
        fraction = args.crop_fraction if args.crop_fraction is not None else 0.6
        zoom = args.zoom if args.zoom is not None else 1.5
        if not math.isfinite(fraction) or not 0.25 <= fraction <= 0.9:
            parser.error("--crop-fraction must be between 0.25 and 0.9")
        if not math.isfinite(zoom) or not 1 <= zoom <= 4:
            parser.error("--zoom must be between 1 and 4")
        paths = args.paths or Path(os.environ.get("QLRG_PATHS_CONFIG", REPO / "configs/paths_hpc.yaml"))
        env = args.env or Path(os.environ.get("QLRG_ENV", REPO / ".venv"))
        selector_cache = args.selector_cache or (REPO / "local" / "selector-cache")
        launch = {
            "repo": str(REPO), "data_dir": str(args.data_dir.resolve()), "env": str(env.expanduser().resolve()),
            "selector": selector,
            "paths": str(paths.expanduser().resolve()), "selector_cache": str(selector_cache.expanduser().resolve()),
            "s1_run": str(args.s1_run.resolve()) if args.s1_run else None, "arm": args.arm, "split": split,
            "images": args.images, "crop_fraction": fraction, "zoom": zoom,
            "seed": args.seed if args.seed is not None else 42, "time": args.time or "06:00:00",
        }
        if selector == "sdrpn":
            launch.update({key: str(path.expanduser().resolve()) for key, path in sdrpn_paths.items()})
    python = Path(launch["env"]) / "bin/python"
    required = [python, Path(launch["paths"]), Path(launch["data_dir"]) / "COMPLETE.json",
                Path(launch["data_dir"]) / f"{launch['split']}.json"]
    if launch["s1_run"] is not None:
        required.append(Path(launch["s1_run"]) / "s1" / "COMPLETE.json")
    if launch.get("selector") == "sdrpn":
        required.extend((
            Path(launch["sdrpn_env"]) / "bin/python",
            Path(launch["sdrpn_vendor"]) / "qwen_src/qwen2_5_vl/modeling_qwen2_5_vl.py",
            Path(launch["sdrpn_checkpoint"]) / "config.json",
            Path(launch["sdrpn_checkpoint"]) / "model.safetensors.index.json",
        ))
    for path in required:
        if not path.is_file():
            parser.error(f"Required file not found: {path}")

    if args.resume is None:
        runs = REPO / "runs"
        runs.mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = "sdrpn-crop" if launch["selector"] == "sdrpn" else "query-crop"
        run_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}-{timestamp}-", dir=runs))
        (run_dir / "launch.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")

    logs = run_dir / "logs"
    script = write_script(run_dir, "query-crop", [
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"',
        "export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1",
        "export QLRG_ENV=" + shlex.quote(launch["env"]),
        "export QLRG_PATHS_CONFIG=" + shlex.quote(launch["paths"]),
        "bash scripts/check_environment.sh",
        "nvidia-smi",
        "exec " + shlex.join([str(python), "-B", str(REPO / "run_query_crop.py"),
                              "--worker-run-dir", str(run_dir)]),
    ])
    resources = json.loads((REPO / "configs/slurm_hpc.json").read_text(encoding="utf-8"))
    full = resources["full"]
    job = "qlrg-sdrpn-crop" if launch.get("selector") == "sdrpn" else "qlrg-query-crop"
    submit = [
        "sbatch", "--parsable", f"--chdir={REPO}", f"--job-name={job}",
        f"--account={resources['account']}", f"--partition={resources['partition']}",
        f"--qos={resources['qos']}", f"--gres={resources['gres']}", f"--cpus-per-task={full['cpus']}",
        f"--mem={full['memory']}", f"--time={args.time or launch['time']}",
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
