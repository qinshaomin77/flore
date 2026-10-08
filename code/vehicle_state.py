"""Streaming per-vehicle, per-second Parquet output for evaluation runs."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np


SCHEMA_VERSION = "vehicle_state_v1"
HBEFA4_CLASS_BY_RAW_TYPE = {
    "sedan": "HBEFA4/PC_petrol_Euro-6ab",
    "sedan_fixed": "HBEFA4/PC_petrol_Euro-6ab",
    "truck": "HBEFA4/RT_gt14-20t_Euro-VI_D-E",
    "truck_fixed": "HBEFA4/RT_gt14-20t_Euro-VI_D-E",
}


def validate_hbefa4_vehicle_types(traci: Any) -> Dict[str, str]:
    """Validate every supported vType that is defined in the active SUMO route."""
    defined_type_ids = {str(type_id) for type_id in traci.vehicletype.getIDList()}
    actual: Dict[str, str] = {}
    for type_id, expected in HBEFA4_CLASS_BY_RAW_TYPE.items():
        # Evaluation route files do not all define the optional ``*_fixed``
        # aliases.  Querying an absent type is a TraCI error, so validate the
        # intersection here and still reject any unsupported type if it later
        # appears in append_vehicle().
        if type_id not in defined_type_ids:
            continue
        value = str(traci.vehicletype.getEmissionClass(type_id))
        if value != expected:
            raise RuntimeError(
                f"SUMO emissionClass mismatch for {type_id!r}: "
                f"expected {expected!r}, got {value!r}"
            )
        actual[type_id] = value
    return actual


class VehicleStateParquetWriter:
    """Collect one decision-step column batch and stream large row groups."""

    def __init__(
        self,
        output_root: str | Path,
        episode: int,
        sumo_seed: int,
        emission_class_by_type: Mapping[str, str],
        row_group_rows: int = 100_000,
        compression: str = "zstd",
        compression_level: int = 3,
    ) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "vehicle_state output requires pyarrow; install the project environment"
            ) from exc

        self.pa = pa
        self.pq = pq
        self.episode = int(episode)
        self.sumo_seed = int(sumo_seed)
        self.row_group_rows = max(1, int(row_group_rows))
        self.compression = str(compression)
        self.compression_level = int(compression_level)
        self.emission_class_by_type = dict(emission_class_by_type)
        if not self.emission_class_by_type:
            raise RuntimeError("No supported sedan/truck SUMO vehicle types are defined")

        output_dir = Path(output_root).resolve() / "vehicle_state"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.final_path = output_dir / f"vehicle_state_ep{self.episode:04d}.parquet"
        self.incomplete_path = Path(str(self.final_path) + ".incomplete")
        self.summary_path = output_dir / f".vehicle_state_ep{self.episode:04d}.summary.json"

        self.schema = pa.schema([
            ("episode", pa.int32()),
            ("sumo_seed", pa.int32()),
            ("simulation_time_s", pa.float64()),
            ("decision_step", pa.int32()),
            ("vehicle_id", pa.string()),
            ("vehicle_type_raw", pa.string()),
            ("vehicle_type_used", pa.string()),
            ("emission_class", pa.string()),
            ("lane_id", pa.string()),
            ("edge_id", pa.string()),
            ("in_observation_zone", pa.bool_()),
            ("observation_lane_id", pa.string()),
            ("speed_mps", pa.float32()),
            ("acceleration_mps2", pa.float32()),
            ("time_loss_accumulated_s", pa.float32()),
            ("is_waiting", pa.bool_()),
            ("included_in_moves_scope", pa.bool_()),
            ("nox_hbefa4_mg_s", pa.float64()),
            ("hbefa4_nox_negative", pa.bool_()),
            ("nox_moves_mg_s", pa.float64()),
            ("moves_speed_kmh_used", pa.int16()),
            ("moves_acceleration_mps2_used", pa.float32()),
            ("moves_lookup_clipped", pa.bool_()),
            ("moves_lookup_fallback", pa.bool_()),
        ])
        self._writer = pq.ParquetWriter(
            str(self.incomplete_path),
            self.schema,
            compression=self.compression,
            compression_level=self.compression_level,
            use_dictionary=[
                "vehicle_type_raw", "vehicle_type_used", "emission_class",
                "lane_id", "edge_id", "observation_lane_id",
            ],
            write_statistics=True,
        )
        self._queued_batches = []
        self._queued_rows = 0
        self._step = self._new_step()
        self._next_row_id = 0
        self._closed = False
        self._unique_vehicle_ids: set[str] = set()
        self._summary = {
            "episode": self.episode,
            "sumo_seed": self.sumo_seed,
            "status": "incomplete",
            "schema_version": SCHEMA_VERSION,
            "compression": self.compression,
            "compression_level": self.compression_level,
            "row_group_rows": self.row_group_rows,
            "row_count": 0,
            "moves_scope_row_count": 0,
            "moves_out_of_scope_row_count": 0,
            "moves_missing_in_scope_count": 0,
            "sedan_vehicle_seconds": 0,
            "truck_vehicle_seconds": 0,
            "hbefa4_nox_total_mg": 0.0,
            "hbefa4_negative_row_count": 0,
            "moves_nox_total_mg": 0.0,
            "sedan_hbefa4_nox_total_mg": 0.0,
            "truck_hbefa4_nox_total_mg": 0.0,
            "sedan_moves_nox_total_mg": 0.0,
            "truck_moves_nox_total_mg": 0.0,
        }

    @staticmethod
    def _new_step() -> Dict[str, list]:
        return {name: [] for name in (
            "row_id", "simulation_time_s", "decision_step", "vehicle_id",
            "vehicle_type_raw", "vehicle_type_used", "emission_class",
            "lane_id", "edge_id", "in_observation_zone",
            "observation_lane_id", "speed_mps", "acceleration_mps2",
            "time_loss_accumulated_s", "is_waiting",
            "included_in_moves_scope", "nox_hbefa4_mg_s",
        )}

    def append_vehicle(
        self,
        *,
        simulation_time_s: float,
        decision_step: int,
        vehicle_id: str,
        vehicle_type_raw: str,
        vehicle_type_used: str,
        lane_id: str,
        edge_id: str,
        speed_mps: float,
        acceleration_mps2: float,
        time_loss_accumulated_s: float,
        is_waiting: bool,
        in_observation_zone: bool,
        observation_lane_id: str,
        included_in_moves_scope: bool,
        nox_hbefa4_mg_s: float,
    ) -> int:
        raw_type = str(vehicle_type_raw)
        if raw_type not in self.emission_class_by_type:
            raise RuntimeError(f"Unsupported SUMO vehicle type in vehicle_state: {raw_type!r}")
        hbefa = float(nox_hbefa4_mg_s)
        if not math.isfinite(hbefa):
            raise RuntimeError(
                f"Invalid HBEFA4 NOx for vehicle {vehicle_id!r}: {hbefa!r}"
            )
        row_id = self._next_row_id
        self._next_row_id += 1
        values = {
            "row_id": row_id,
            "simulation_time_s": float(simulation_time_s),
            "decision_step": int(decision_step),
            "vehicle_id": str(vehicle_id),
            "vehicle_type_raw": raw_type,
            "vehicle_type_used": str(vehicle_type_used),
            "emission_class": self.emission_class_by_type[raw_type],
            "lane_id": str(lane_id),
            "edge_id": str(edge_id),
            "in_observation_zone": bool(in_observation_zone),
            "observation_lane_id": str(observation_lane_id) if in_observation_zone else "",
            "speed_mps": float(speed_mps),
            "acceleration_mps2": float(acceleration_mps2),
            "time_loss_accumulated_s": float(time_loss_accumulated_s),
            "is_waiting": bool(is_waiting),
            "included_in_moves_scope": bool(included_in_moves_scope),
            "nox_hbefa4_mg_s": hbefa,
        }
        for name, value in values.items():
            self._step[name].append(value)
        return row_id

    def commit_decision_step(self, detail: Optional[Any]) -> None:
        n = len(self._step["row_id"])
        if not n:
            return
        moves = np.full(n, np.nan, dtype=np.float64)
        speed_used = np.zeros(n, dtype=np.int16)
        accel_used = np.zeros(n, dtype=np.float32)
        clipped = np.zeros(n, dtype=np.bool_)
        fallback = np.zeros(n, dtype=np.bool_)
        first_row_id = int(self._step["row_id"][0])
        if detail is not None:
            positions = np.asarray(detail.row_ids, dtype=np.int64) - first_row_id
            if np.any(positions < 0) or np.any(positions >= n):
                raise RuntimeError("MOVES detail row ids do not match vehicle_state batch")
            moves[positions] = np.asarray(detail.nox_mg, dtype=np.float64)
            speed_used[positions] = np.asarray(detail.speed_kmh_used, dtype=np.int16)
            accel_used[positions] = np.asarray(detail.accel_ms2_used, dtype=np.float32)
            clipped[positions] = np.asarray(detail.lookup_clipped, dtype=np.bool_)
            fallback[positions] = np.asarray(detail.fallback_used, dtype=np.bool_)

        scope = np.asarray(self._step["included_in_moves_scope"], dtype=np.bool_)
        missing_scope = scope & ~np.isfinite(moves)
        if np.any(missing_scope):
            self._summary["moves_missing_in_scope_count"] += int(np.count_nonzero(missing_scope))
            raise RuntimeError("MOVES values are missing for in-scope vehicle_state rows")

        pa = self.pa
        arrays = [
            pa.array(np.full(n, self.episode, dtype=np.int32)),
            pa.array(np.full(n, self.sumo_seed, dtype=np.int32)),
            pa.array(self._step["simulation_time_s"], type=pa.float64()),
            pa.array(self._step["decision_step"], type=pa.int32()),
            pa.array(self._step["vehicle_id"], type=pa.string()),
            pa.array(self._step["vehicle_type_raw"], type=pa.string()),
            pa.array(self._step["vehicle_type_used"], type=pa.string()),
            pa.array(self._step["emission_class"], type=pa.string()),
            pa.array(self._step["lane_id"], type=pa.string()),
            pa.array(self._step["edge_id"], type=pa.string()),
            pa.array(self._step["in_observation_zone"], type=pa.bool_()),
            pa.array(self._step["observation_lane_id"], type=pa.string()),
            pa.array(self._step["speed_mps"], type=pa.float32()),
            pa.array(self._step["acceleration_mps2"], type=pa.float32()),
            pa.array(self._step["time_loss_accumulated_s"], type=pa.float32()),
            pa.array(self._step["is_waiting"], type=pa.bool_()),
            pa.array(scope, type=pa.bool_()),
            pa.array(self._step["nox_hbefa4_mg_s"], type=pa.float64()),
            pa.array(
                np.asarray(self._step["nox_hbefa4_mg_s"], dtype=np.float64) < 0.0,
                type=pa.bool_(),
            ),
            pa.array(moves, mask=~scope, type=pa.float64()),
            pa.array(speed_used, mask=~scope, type=pa.int16()),
            pa.array(accel_used, mask=~scope, type=pa.float32()),
            pa.array(clipped, mask=~scope, type=pa.bool_()),
            pa.array(fallback, mask=~scope, type=pa.bool_()),
        ]
        self._queued_batches.append(pa.RecordBatch.from_arrays(arrays, schema=self.schema))
        self._queued_rows += n
        self._update_summary(moves, scope)
        self._step = self._new_step()
        if self._queued_rows >= self.row_group_rows:
            self._flush()

    def _update_summary(self, moves: np.ndarray, scope: np.ndarray) -> None:
        n = len(scope)
        types = np.asarray(self._step["vehicle_type_used"], dtype=object)
        hbefa = np.asarray(self._step["nox_hbefa4_mg_s"], dtype=np.float64)
        self._summary["row_count"] += n
        self._summary["moves_scope_row_count"] += int(np.count_nonzero(scope))
        self._summary["moves_out_of_scope_row_count"] += int(np.count_nonzero(~scope))
        self._summary["hbefa4_nox_total_mg"] += float(np.sum(hbefa, dtype=np.float64))
        self._summary["hbefa4_negative_row_count"] += int(np.count_nonzero(hbefa < 0.0))
        self._summary["moves_nox_total_mg"] += float(np.nansum(moves, dtype=np.float64))
        self._unique_vehicle_ids.update(self._step["vehicle_id"])
        for vehicle_type in ("sedan", "truck"):
            mask = types == vehicle_type
            self._summary[f"{vehicle_type}_vehicle_seconds"] += int(np.count_nonzero(mask))
            self._summary[f"{vehicle_type}_hbefa4_nox_total_mg"] += float(
                np.sum(hbefa[mask], dtype=np.float64)
            )
            self._summary[f"{vehicle_type}_moves_nox_total_mg"] += float(
                np.nansum(moves[mask], dtype=np.float64)
            )

    def _flush(self) -> None:
        if not self._queued_batches:
            return
        table = self.pa.Table.from_batches(self._queued_batches, schema=self.schema)
        self._writer.write_table(table, row_group_size=self.row_group_rows)
        self._queued_batches.clear()
        self._queued_rows = 0

    def close(self) -> Dict[str, Any]:
        if self._closed:
            return dict(self._summary)
        if self._step["row_id"]:
            raise RuntimeError("Uncommitted vehicle_state rows remain at episode close")
        self._flush()
        self._writer.close()
        os.replace(self.incomplete_path, self.final_path)
        self._closed = True
        self._summary.update({
            "status": "ok",
            "unique_vehicle_count": len(self._unique_vehicle_ids),
            "parquet_file": str(self.final_path),
            "parquet_size_bytes": self.final_path.stat().st_size,
        })
        self.summary_path.write_text(
            json.dumps(self._summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return dict(self._summary)

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self._writer.close()
        finally:
            self._closed = True


def consolidate_vehicle_state_outputs(
    output_root: str | Path,
    replace_episodes: set[int] | None = None,
) -> None:
    """Create or repair the run-level vehicle-state summary and metadata."""
    root = Path(output_root).resolve()
    vehicle_dir = root / "vehicle_state"
    if not vehicle_dir.exists():
        return
    new_rows = []
    for path in sorted(vehicle_dir.glob(".vehicle_state_ep*.summary.json")):
        new_rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not new_rows:
        return

    if replace_episodes is None:
        rows = new_rows
    else:
        replaced = {int(episode) for episode in replace_episodes}
        replacement_rows = {
            int(row["episode"]): row
            for row in new_rows
            if int(row["episode"]) in replaced
        }
        missing = sorted(replaced - replacement_rows.keys())
        if missing:
            raise ValueError(
                "Missing vehicle-state summaries for repaired episodes: "
                f"{missing}"
            )

        rows_by_episode: dict[int, dict[str, Any]] = {}
        existing_summary = root / "vehicle_state_summary_by_episode.csv"
        if existing_summary.is_file() and existing_summary.stat().st_size > 0:
            with existing_summary.open("r", newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    try:
                        episode = int(row["episode"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if episode not in replaced:
                        rows_by_episode[episode] = dict(row)
        rows_by_episode.update(replacement_rows)
        rows = list(rows_by_episode.values())

    rows.sort(key=lambda row: int(row["episode"]))
    columns = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    summary_tmp = root / "vehicle_state_summary_by_episode.csv.tmp"
    with summary_tmp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(summary_tmp, root / "vehicle_state_summary_by_episode.csv")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "simulation_step_length_s": 1.0,
        "emission_units": "mg/s (equal to mg per simulation step for dt=1s)",
        "hbefa4_vehicle_type_mapping": HBEFA4_CLASS_BY_RAW_TYPE,
        "moves_scope": (
            "Rows where included_in_moves_scope is true; the project-specific "
            "predicate preserves its existing MOVES recorder scope"
        ),
        "out_of_scope_moves_value": None,
        "hbefa4_negative_policy": "preserve SUMO value and flag the row",
        "parquet": {
            "compression": rows[0].get("compression"),
            "compression_level": rows[0].get("compression_level"),
            "row_group_rows": rows[0].get("row_group_rows"),
        },
        "run_metadata": "metadata.json",
        "episode_count": len(rows),
        "episodes": [
            {
                "episode": row["episode"],
                "sumo_seed": row["sumo_seed"],
                "parquet_file": row["parquet_file"],
                "row_count": row["row_count"],
            }
            for row in rows
        ],
    }
    metadata_tmp = root / "vehicle_state_metadata.json.tmp"
    metadata_tmp.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(metadata_tmp, root / "vehicle_state_metadata.json")
