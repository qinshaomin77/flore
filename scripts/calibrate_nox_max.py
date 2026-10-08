"""Calibrate one global NOx_max from E2-300 lane-step logs.

All demand scenarios, episodes, lanes, and post-warmup time steps are pooled
into one sample population. No lane-type or demand-level threshold is created.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "results" / "calibration" / "grid36"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "configs" / "thresholds" / "grid36"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate one global NOx_max from E2-300 lane_step.csv files."
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for one JSON result per demand (D20.json, ...).",
    )
    parser.add_argument("--quantile", type=float, default=0.80)
    parser.add_argument("--warmup-seconds", type=float, default=300.0)
    parser.add_argument("--zero-eps", type=float, default=1e-9)
    return parser.parse_args()


def collect_samples(
    run_dir: Path,
    warmup_seconds: float,
    zero_eps: float,
) -> tuple[list[float], int, list[str]]:
    paths = sorted(run_dir.glob("physical/episode_*/lane_step.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No lane_step.csv files found under {run_dir}"
        )

    values: list[float] = []
    demands: set[str] = {run_dir.name}
    for path in paths:
        # For both root/physical/episode_* and root/D20/physical/episode_*,
        # the demand name is the directory immediately above ``physical``.
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"sim_time", "lane_nox_mg"}
            missing = sorted(required - set(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"{path} missing required columns: {missing}")
            for row in reader:
                try:
                    sim_time = float(row["sim_time"])
                    nox = float(row["lane_nox_mg"])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(sim_time) or not math.isfinite(nox):
                    continue
                if sim_time < warmup_seconds or nox <= zero_eps:
                    continue
                values.append(nox)

    if not values:
        raise ValueError("No positive lane_nox_mg samples remain after filtering")
    return values, len(paths), sorted(demands)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.quantile < 1.0:
        raise ValueError("--quantile must be between 0 and 1")
    if args.warmup_seconds < 0.0 or args.zero_eps < 0.0:
        raise ValueError("--warmup-seconds and --zero-eps must be non-negative")

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if (run_dir / "physical").is_dir():
        demand_dirs = [run_dir]
    else:
        demand_dirs = sorted(
            (
                p
                for p in run_dir.iterdir()
                if p.is_dir() and (p / "physical").is_dir()
            ),
            key=lambda p: p.name,
        )
    if not demand_dirs:
        raise FileNotFoundError(
            f"No demand directories containing physical/ found under {run_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for demand_dir in demand_dirs:
        values, file_count, demands = collect_samples(
            demand_dir, args.warmup_seconds, args.zero_eps
        )
        raw_value = float(np.percentile(np.asarray(values, dtype=np.float64), args.quantile * 100.0))
        nox_max = max(int(math.ceil(raw_value)), 1)
        output = output_dir / f"{demand_dir.name}.json"
        payload = {
            "NOx_max": nox_max,
            "observation_scope": "upstream_300m",
            "observation_length": 300.0,
            "calibration_meta": {
                "quantile": float(args.quantile),
                "warmup_seconds": float(args.warmup_seconds),
                "zero_eps": float(args.zero_eps),
                "sample_count": int(len(values)),
                "lane_step_files": int(file_count),
                "demand": demand_dir.name,
            },
        }
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"demand": demand_dir.name, "raw_quantile_value": raw_value, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
