"""Calibrate one pooled global lane-NOx threshold for each fixed E2 scale.

The supported scales are 100 m, 300 m, and 500 m.  Full-lane calibration is
intentionally excluded.  For 500 m, short lanes keep their actual clipped E2
extent and are pooled with the other lanes without extrapolation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCALE_SPECS = (
    ("w100", "kunshan_actuated_w100", "E2_last_100m", 100.0),
    ("w300", "kunshan_actuated_w300", "E2_last_300m", 300.0),
    ("w500", "kunshan_actuated_w500", "E2_last_500m", 500.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate pooled global P80 lane-NOx thresholds for Kunshan E2 scales."
    )
    parser.add_argument(
        "--batch-dir",
        required=True,
        help="Batch directory containing kunshan_actuated_w100/w300/w500.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "configs" / "thresholds" / "kunshan"),
    )
    parser.add_argument("--quantile", type=float, default=0.80)
    parser.add_argument("--warmup-seconds", type=float, default=300.0)
    parser.add_argument("--zero-eps", type=float, default=1e-9)
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--decision-interval", type=int, default=10)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def validate_run_metadata(
    run_dir: Path,
    expected_episodes: int,
    decision_interval: int,
) -> list[dict[str, int]]:
    error_path = run_dir / "eval_errors.json"
    if error_path.exists():
        raise ValueError(f"Calibration run contains eval_errors.json: {error_path}")

    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = read_json(metadata_path)
    if str(metadata.get("emission_mode", "")).strip().lower() != "actual":
        raise ValueError(f"Calibration emission_mode must be actual: {metadata_path}")
    if int(metadata.get("decision_interval", -1)) != decision_interval:
        raise ValueError(
            f"Decision interval mismatch in {metadata_path}: "
            f"expected={decision_interval}, actual={metadata.get('decision_interval')}"
        )

    seeds_path = run_dir / "eval_sumo_seeds.json"
    if not seeds_path.is_file():
        raise FileNotFoundError(seeds_path)
    seeds_payload = read_json(seeds_path)
    seeds = seeds_payload.get("sumo_seeds", [])
    if not isinstance(seeds, list) or len(seeds) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} SUMO seeds in {seeds_path}, got {len(seeds)}"
        )
    return [
        {
            "eval_episode": int(item["eval_episode"]),
            "sumo_seed": int(item["sumo_seed"]),
        }
        for item in seeds
    ]


def collect_positive_samples(
    run_dir: Path,
    warmup_seconds: float,
    zero_eps: float,
    expected_episodes: int,
) -> tuple[np.ndarray, dict[str, int]]:
    paths = sorted(run_dir.glob("physical/episode_*/lane_step.csv"))
    if len(paths) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} lane_step.csv files under {run_dir}, "
            f"got {len(paths)}"
        )

    values: list[float] = []
    total_rows = 0
    finite_rows = 0
    post_warmup_rows = 0
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"sim_time", "lane_nox_mg"}
            missing = sorted(required - set(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"{path} missing required columns: {missing}")
            for row in reader:
                total_rows += 1
                try:
                    sim_time = float(row["sim_time"])
                    nox = float(row["lane_nox_mg"])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(sim_time) or not math.isfinite(nox):
                    continue
                finite_rows += 1
                if sim_time < warmup_seconds:
                    continue
                post_warmup_rows += 1
                if nox > zero_eps:
                    values.append(nox)

    if not values:
        raise ValueError(f"No positive post-warmup lane_nox_mg samples under {run_dir}")
    return np.asarray(values, dtype=np.float64), {
        "input_file_count": len(paths),
        "sample_count_raw": total_rows,
        "sample_count_finite": finite_rows,
        "sample_count_post_warmup": post_warmup_rows,
        "sample_count_positive": len(values),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.quantile < 1.0:
        raise ValueError("--quantile must be in (0, 1)")
    if args.warmup_seconds < 0.0 or args.zero_eps < 0.0:
        raise ValueError("--warmup-seconds and --zero-eps must be non-negative")
    if args.expected_episodes < 1 or args.decision_interval < 1:
        raise ValueError("--expected-episodes and --decision-interval must be positive")

    batch_dir = Path(args.batch_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not batch_dir.is_dir():
        raise FileNotFoundError(batch_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_seeds: list[dict[str, int]] | None = None
    audit_rows: list[dict[str, Any]] = []
    for scale, run_name, observation_scope, target_length in SCALE_SPECS:
        run_dir = batch_dir / run_name
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)

        seeds = validate_run_metadata(
            run_dir,
            expected_episodes=int(args.expected_episodes),
            decision_interval=int(args.decision_interval),
        )
        if reference_seeds is None:
            reference_seeds = seeds
        elif seeds != reference_seeds:
            raise ValueError(f"SUMO seed list differs across spatial scales: {run_dir}")

        values, counts = collect_positive_samples(
            run_dir,
            warmup_seconds=float(args.warmup_seconds),
            zero_eps=float(args.zero_eps),
            expected_episodes=int(args.expected_episodes),
        )
        nox_max = float(np.quantile(values, float(args.quantile)))
        if not math.isfinite(nox_max) or nox_max <= 0.0:
            raise ValueError(f"Invalid NOx_max for {scale}: {nox_max}")

        positive_ratio = float(
            counts["sample_count_positive"]
            / max(counts["sample_count_post_warmup"], 1)
        )
        output = output_dir / f"kunshan_{scale}.json"
        payload = {
            "schema_version": 4,
            "network_id": "kunshan_actuated",
            "baseline_emission_mode": "actual",
            "baseline_vehicle_dynamics": "actual_mixed_fleet",
            "observation_scope": observation_scope,
            "NOx_max": nox_max,
            "NOx_max_by_type": {},
            "lane_type_map": {},
            "type_positive_sample_count": {},
            "type_threshold_source": {},
            "calibration_meta": {
                "aggregation": "pooled_positive_lane_step_samples",
                "quantile": float(args.quantile),
                "warmup_seconds": float(args.warmup_seconds),
                "exclude_zero": True,
                "zero_eps": float(args.zero_eps),
                "observation_scope": observation_scope,
                "detector_target_length_m": target_length,
                "detector_length_policy": (
                    "actual_extent_including_clipped_short_lanes"
                    if scale == "w500"
                    else "fixed_target_extent"
                ),
                "decision_interval": int(args.decision_interval),
                "episode_count": int(args.expected_episodes),
                **counts,
                "positive_ratio": positive_ratio,
                "rounding": "none",
                "source_run_dir": str(run_dir),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        audit_rows.append({
            "scale": scale,
            "observation_scope": observation_scope,
            "NOx_max": nox_max,
            "p50": float(np.quantile(values, 0.50)),
            "p75": float(np.quantile(values, 0.75)),
            "p80": float(np.quantile(values, 0.80)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "positive_ratio": positive_ratio,
            **counts,
            "output_json": str(output),
        })
        print(f"{scale}: NOx_max={nox_max:.12g}, positive_samples={values.size}, output={output}")

    audit_path = output_dir / "kunshan_spatialscale_global_threshold_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"audit CSV: {audit_path}")


if __name__ == "__main__":
    main()
