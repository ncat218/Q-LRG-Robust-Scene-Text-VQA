"""Prepare reproducible S1 candidate configs without submitting jobs."""

import argparse
import json
import math
from pathlib import Path


def prepare(output, template, seeds, learning_rates):
    if len(set(seeds)) != len(seeds) or len(seeds) < 3:
        raise ValueError("Provide at least three distinct seeds")
    if not learning_rates or len(set(learning_rates)) != len(learning_rates) or any(
        not math.isfinite(value) or value <= 0 for value in learning_rates
    ):
        raise ValueError("Learning rates must be distinct positive finite values")
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(template.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    candidates = []
    for rate in learning_rates:
        for seed in seeds:
            candidate = {**config, "seed": seed, "learning_rate": rate}
            name = f"seed{seed}-lr{rate:g}.json"
            (output / name).write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
            candidates.append({"config": name, "seed": seed, "learning_rate": rate})
    (output / "study.json").write_text(json.dumps({
        "selection": "val_inner", "reporting": "official_val after configuration lock",
        "candidates": candidates, "template": str(template.resolve()),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(candidates)} configurations in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/s1_expanded.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.00005, 0.0001, 0.0002])
    args = parser.parse_args()
    prepare(args.output, args.template, args.seeds, args.learning_rates)


if __name__ == "__main__":
    main()
