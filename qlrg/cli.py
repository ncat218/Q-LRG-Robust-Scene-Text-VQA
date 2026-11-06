"""Command line entry point for data preparation, inference and verification."""

import argparse
from pathlib import Path

from .common import dataset_paths, read_config, read_json, sha256, write_json
from .data import annotation_file, build_manifest, load_clean, verify_pillow
from .experiment import execute, validate_manifest, verified_complete
from .grounding import write_review_pack


def main():
    parser = argparse.ArgumentParser(prog="qlrg")
    parser.add_argument("--paths", type=Path, default=Path("configs/paths_hpc.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-manifest")
    build.add_argument("--images", type=int, default=500)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-dataset")
    validate.add_argument("--manifest", type=Path, required=True)
    review = commands.add_parser("prepare-review")
    review.add_argument("--manifest", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--limit", type=int, default=150)
    for stage in ("d0", "s0"):
        run = commands.add_parser(stage)
        run.add_argument("--manifest", type=Path, required=True)
        run.add_argument("--mode", choices=("smoke", "full"), required=True)
        run.add_argument("--run-dir", type=Path, required=True)
        run.add_argument("--model-config", type=Path, default=Path("configs/model_qwen.yaml"))
        run.add_argument("--config", type=Path, default=Path(
            "configs/d0_baseline.yaml" if stage == "d0" else "configs/s0_oracle.yaml"
        ))
        if stage == "s0":
            grounding = run.add_mutually_exclusive_group()
            grounding.add_argument("--review", type=Path)
            grounding.add_argument("--provisional-grounding", action="store_true",
                                   help="Allow provisional OCR grounding for full S0")
            run.add_argument("--d0-run", type=Path, help="Matching D0 stage directory if kept under a different run")
    verify = commands.add_parser("verify-run")
    verify.add_argument("--stage", choices=("d0", "s0"), required=True)
    verify.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "verify-run":
        stage_dir = args.run_dir / args.stage
        verified_complete(stage_dir, args.stage)
        print(f"Verified {stage_dir}")
        return
    paths = dataset_paths(read_config(args.paths))
    if args.command == "build-manifest":
        if args.output.exists():
            raise FileExistsError(args.output)
        manifest = build_manifest(paths["textvqa"], paths["textocr"], args.images, args.seed)
        validate_manifest(manifest)
        write_json(args.output, manifest)
        questions = sum(len(image["questions"]) for image in manifest["images"])
        candidates = sum(
            q["candidate"] is not None for image in manifest["images"] for q in image["questions"]
        )
        print(f"{len(manifest['images'])} images, {questions} questions, {candidates} O3 candidates")
        print(f"{manifest['eligible_val_inner_images']} eligible images in val_inner")
        print(f"{args.output} SHA256 {sha256(args.output)}")
    elif args.command == "validate-dataset":
        verify_pillow()
        manifest = read_json(args.manifest)
        validate_manifest(manifest)
        for split, digest in manifest["annotation_sha256"].items():
            if sha256(annotation_file(paths["textvqa"], split)) != digest:
                raise ValueError(f"Annotation file changed for {split}")
        if sha256(paths["textocr"]) != manifest["textocr_sha256"]:
            raise ValueError("TextOCR source changed")
        for image in manifest["images"]:
            decoded, orientation, _ = load_clean(paths["textvqa"] / image["image_rel"], image["image_sha256"])
            if list(decoded.size) != image["size"] or orientation != image["exif_orientation"]:
                raise ValueError(f"Geometry changed for {image['image_id']}")
        print(f"Verified {len(manifest['images'])} images and source annotation hashes")
    elif args.command == "prepare-review":
        manifest = read_json(args.manifest)
        validate_manifest(manifest)
        count = write_review_pack(manifest, paths["textvqa"], args.output, args.limit)
        print(f"Prepared {count} distinct image/question pairs for grounding review in {args.output}")
    else:
        execute(
            args.command, args.mode, args.manifest, paths,
            read_config(args.model_config), read_config(args.config), args.run_dir / args.command,
            review_path=getattr(args, "review", None),
            allow_provisional=getattr(args, "provisional_grounding", False),
            d0_path=getattr(args, "d0_run", None) or args.run_dir / "d0",
        )


if __name__ == "__main__":
    main()
