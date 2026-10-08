
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

LookupKey = Tuple[str, str, int, float]

@dataclass(frozen=True)
class EmissionQueryInfo:

    vehicle_type_raw: str
    vehicle_type_used: str
    pollutant_used: str
    speed_ms: float
    speed_kmh_used: int
    accel_ms2_raw: float
    accel_ms2_used: float
    factor_gs: float
    matched: bool
    fallback_used: bool

@dataclass(frozen=True)
class VehicleStateRecord:

    episode_id: int
    sim_time: float
    decision_step: int
    veh_id: str
    vehicle_type_raw: str
    vehicle_type_used: str
    speed_ms: float
    accel_ms2: float
    lane_id: str
    edge_id: str
    dt: float
    distance_m: float = 0.0
    emissions_mg: Dict[str, float] = field(default_factory=dict)

@dataclass(frozen=True)
class VehicleStepRecord:
    sim_time: float
    decision_step: int
    veh_id: str
    vehicle_type_raw: str
    vehicle_type_used: str
    speed_ms: float
    accel_ms2: float
    lane_id: str
    edge_id: str
    dt: float
    distance_m: float
    emissions_mg: Dict[str, float]

@dataclass(frozen=True)
class VehicleEmissionDetailBatch:
    decision_step: int
    row_ids: np.ndarray
    nox_mg: np.ndarray
    speed_kmh_used: np.ndarray
    accel_ms2_used: np.ndarray
    lookup_clipped: np.ndarray
    fallback_used: np.ndarray

@dataclass
class DecisionStepEmission:

    episode_id: int
    decision_step: int
    sim_time_start: float
    sim_time_end: float
    n_records: int
    n_vehicles: int
    output_unit: str
    total_by_pollutant: Dict[str, float] = field(default_factory=dict)
    by_vehicle_type: Dict[Tuple[str, str], float] = field(default_factory=dict)
    by_vehicle: Dict[Tuple[str, str], float] = field(default_factory=dict)
    by_edge: Dict[Tuple[str, str], float] = field(default_factory=dict)
    by_lane: Dict[Tuple[str, str], float] = field(default_factory=dict)
    by_lane_analysis_full: Dict[Tuple[str, str], float] = field(default_factory=dict)
    by_lane_observation_zone: Dict[Tuple[str, str], float] = field(default_factory=dict)
    by_lane_observation_zone_signal_sensitive: Dict[Tuple[str, str], float] = field(
        default_factory=dict
    )
    by_vehicle_delta: Dict[Tuple[str, str], float] = field(default_factory=dict)
    consistency_diffs: Dict[str, float] = field(default_factory=dict)

    def to_flat_rows(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for pollutant, value in sorted(self.total_by_pollutant.items()):
            rows.append({
                "episode": self.episode_id,
                "decision_step": self.decision_step,
                "sim_time_start": self.sim_time_start,
                "sim_time_end": self.sim_time_end,
                "pollutant": pollutant,
                f"total_emission_{self.output_unit}": float(value),
                "lane_minus_total": float(self.consistency_diffs.get(f"lane_minus_total_{pollutant}", 0.0)),
                "edge_minus_total": float(self.consistency_diffs.get(f"edge_minus_total_{pollutant}", 0.0)),
                "n_records": self.n_records,
                "n_vehicles": self.n_vehicles,
            })
        return rows

@dataclass
class EpisodeEmissionResult:

    episode_id: int
    output_unit: str
    step_summaries: List[DecisionStepEmission]
    vehicle_totals: Dict[Tuple[str, str], float]
    vehicle_types: Dict[str, str]
    vehicle_distance_m: Dict[str, float]
    step_df: pd.DataFrame
    vehicle_df: pd.DataFrame
    edge_step_df: pd.DataFrame
    lane_step_df: pd.DataFrame
    vehicle_step_df: pd.DataFrame

class EmissionFactorLookup:

    REQUIRED_COLUMNS = {
        "vehicle_type",
        "pollutant",
        "speed_kmh",
        "accel_ms2",
        "EmissionFactor_gs",
    }

    VEHICLE_TYPE_SEDAN = "sedan"
    VEHICLE_TYPE_TRUCK = "truck"
    ALLOWED_VEHICLE_TYPES = {VEHICLE_TYPE_SEDAN, VEHICLE_TYPE_TRUCK}

    TRUCK_ALIASES = {
        "truck", "lorry", "hdv", "heavy", "heavy_duty", "heavy-duty",
        "freight", "freight_truck", "hgv", "trailer", "container",
        "veh_truck", "type_truck",
    }

    SEDAN_ALIASES = {
        "sedan", "car", "passenger", "passenger_car", "auto",
        "vehicle", "veh_passenger", "type_sedan",
    }

    def __init__(
        self,
        csv_path: str | Path,
        default_pollutant: str = "NOx",
        default_vehicle_type: str = "sedan",
        enable_cache: bool = True,
        strict_two_types: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.default_pollutant = str(default_pollutant).strip()
        self.default_vehicle_type = self.normalize_vehicle_type(default_vehicle_type)
        self.enable_cache = bool(enable_cache)
        self.strict_two_types = bool(strict_two_types)

        self.df: Optional[pd.DataFrame] = None
        self.lookup: Dict[LookupKey, float] = {}

        self.vehicle_types: Set[str] = set()
        self.pollutants: Set[str] = set()

        self.min_speed_kmh: int = 0
        self.max_speed_kmh: int = 0
        self.min_accel_ms2: float = 0.0
        self.max_accel_ms2: float = 0.0

        self._query_cache: Dict[LookupKey, float] = {}
        self._pollutant_lower_map: Dict[str, str] = {}
        self._grid_types: List[str] = []
        self._type_to_idx: Dict[str, int] = {}
        self._grid_pols: List[str] = []
        self._pol_to_idx: Dict[str, int] = {}
        self._factor_grid: np.ndarray = np.zeros((0, 0, 0, 0), dtype=np.float64)
        self._factor_exact_grid: np.ndarray = np.zeros((0, 0, 0, 0), dtype=np.bool_)
        self._grid_accel_values: np.ndarray = np.zeros((0,), dtype=np.float64)

        self._load()

    @classmethod
    def normalize_vehicle_type(cls, vehicle_type: str | None) -> str:
        if vehicle_type is None:
            return cls.VEHICLE_TYPE_SEDAN

        vt = str(vehicle_type).strip().lower()
        if vt in cls.ALLOWED_VEHICLE_TYPES:
            return vt

        if vt in cls.TRUCK_ALIASES or "truck" in vt or "lorry" in vt or "hdv" in vt:
            return cls.VEHICLE_TYPE_TRUCK

        if vt in cls.SEDAN_ALIASES or "sedan" in vt or "passenger" in vt or "car" in vt:
            return cls.VEHICLE_TYPE_SEDAN

        return cls.VEHICLE_TYPE_SEDAN

    @staticmethod
    def _mass_factor(output_unit: str) -> float:
        unit = str(output_unit).strip().lower()
        if unit == "g":
            return 1.0
        if unit == "mg":
            return 1000.0
        raise ValueError(f"Unsupported output_unit={output_unit!r}; use 'g' or 'mg'.")

    def _load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Emission factor CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Emission factor CSV missing required columns: {sorted(missing)}"
            )

        df = df.copy()
        df["vehicle_type"] = df["vehicle_type"].map(self.normalize_vehicle_type)
        df["pollutant"] = df["pollutant"].astype(str).str.strip()
        df["speed_kmh"] = pd.to_numeric(df["speed_kmh"], errors="coerce")
        df["accel_ms2"] = pd.to_numeric(df["accel_ms2"], errors="coerce")
        df["EmissionFactor_gs"] = pd.to_numeric(df["EmissionFactor_gs"], errors="coerce")

        df = df.dropna(
            subset=[
                "vehicle_type",
                "pollutant",
                "speed_kmh",
                "accel_ms2",
                "EmissionFactor_gs",
            ]
        ).reset_index(drop=True)

        if self.strict_two_types:
            df = df[df["vehicle_type"].isin(self.ALLOWED_VEHICLE_TYPES)].copy()

        if df.empty:
            raise ValueError(
                "Emission factor table is empty after normalization. "
                "Need at least sedan/truck rows."
            )

        df["speed_kmh"] = df["speed_kmh"].round(0).astype(int)
        df["accel_ms2"] = df["accel_ms2"].round(1).astype(float)
        df["EmissionFactor_gs"] = df["EmissionFactor_gs"].astype(float)

        df = df.drop_duplicates(
            subset=["vehicle_type", "pollutant", "speed_kmh", "accel_ms2"],
            keep="first",
        ).reset_index(drop=True)

        self.df = df
        self.vehicle_types = set(df["vehicle_type"].unique().tolist())
        self.pollutants = set(df["pollutant"].unique().tolist())

        if self.VEHICLE_TYPE_SEDAN not in self.vehicle_types:
            raise ValueError("Emission factor CSV must contain sedan rows after normalization.")
        if self.VEHICLE_TYPE_TRUCK not in self.vehicle_types:
            raise ValueError("Emission factor CSV must contain truck rows after normalization.")

        if self.default_pollutant not in self.pollutants:
            lower_map = {p.lower(): p for p in self.pollutants}
            self.default_pollutant = lower_map.get(
                self.default_pollutant.lower(),
                sorted(self.pollutants)[0],
            )

        if self.default_vehicle_type not in self.vehicle_types:
            self.default_vehicle_type = self.VEHICLE_TYPE_SEDAN

        self.min_speed_kmh = int(df["speed_kmh"].min())
        self.max_speed_kmh = int(df["speed_kmh"].max())
        self.min_accel_ms2 = float(df["accel_ms2"].min())
        self.max_accel_ms2 = float(df["accel_ms2"].max())

        self.lookup = {
            (
                str(row.vehicle_type),
                str(row.pollutant),
                int(row.speed_kmh),
                float(row.accel_ms2),
            ): float(row.EmissionFactor_gs)
            for row in df.itertuples(index=False)
        }
        self._pollutant_lower_map = {p.lower(): p for p in self.pollutants}
        self._build_factor_grid()

    def _build_factor_grid(self) -> None:
        self._grid_types = sorted(self.vehicle_types)
        self._type_to_idx = {vehicle_type: idx for idx, vehicle_type in enumerate(self._grid_types)}
        self._grid_pols = sorted(self.pollutants)
        self._pol_to_idx = {pollutant: idx for idx, pollutant in enumerate(self._grid_pols)}

        n_types = int(len(self._grid_types))
        n_pols = int(len(self._grid_pols))
        n_speed = int(self.max_speed_kmh - self.min_speed_kmh + 1)
        n_accel = int(round((self.max_accel_ms2 - self.min_accel_ms2) / 0.1)) + 1
        self._grid_accel_values = np.round(
            self.min_accel_ms2 + np.arange(n_accel, dtype=np.float64) * 0.1,
            1,
        )
        grid = np.zeros((n_types, n_pols, n_speed, n_accel), dtype=np.float64)
        exact_grid = np.zeros((n_types, n_pols, n_speed, n_accel), dtype=np.bool_)

        for vt_i, vt in enumerate(self._grid_types):
            for pol_i, pol in enumerate(self._grid_pols):
                for s_idx in range(n_speed):
                    speed_kmh = int(self.min_speed_kmh + s_idx)
                    for a_idx, accel in enumerate(self._grid_accel_values):
                        accel_key = float(accel)
                        exact_key = (vt, pol, speed_kmh, accel_key)
                        factor = self.lookup.get(exact_key)
                        exact_grid[vt_i, pol_i, s_idx, a_idx] = factor is not None
                        if factor is None:
                            factor = self.lookup.get((self.default_vehicle_type, pol, speed_kmh, accel_key))
                        if factor is None:
                            factor = self.lookup.get((vt, self.default_pollutant, speed_kmh, accel_key))
                        if factor is None:
                            factor = self.lookup.get(
                                (self.default_vehicle_type, self.default_pollutant, speed_kmh, accel_key),
                                0.0,
                            )
                        grid[vt_i, pol_i, s_idx, a_idx] = float(factor)

        self._factor_grid = grid
        self._factor_exact_grid = exact_grid

    def _normalize_pollutant(self, pollutant: str | None) -> str:
        if pollutant is None:
            return self.default_pollutant

        p = str(pollutant).strip()
        if p in self.pollutants:
            return p

        return self._pollutant_lower_map.get(p.lower(), self.default_pollutant)

    def _normalize_speed_kmh(self, speed_ms: float | int | None) -> int:
        speed_ms = 0.0 if speed_ms is None else float(speed_ms)
        speed_kmh = int(round(speed_ms * 3.6))
        speed_kmh = max(self.min_speed_kmh, min(speed_kmh, self.max_speed_kmh))
        return int(speed_kmh)

    def _normalize_accel_ms2(self, accel_ms2: float | int | None) -> float:
        accel = 0.0 if accel_ms2 is None else float(accel_ms2)
        accel = round(accel, 1)
        accel = max(self.min_accel_ms2, min(accel, self.max_accel_ms2))
        return float(accel)

    def has_vehicle_type(self, vehicle_type: str) -> bool:
        return self.normalize_vehicle_type(vehicle_type) in self.vehicle_types

    def has_pollutant(self, pollutant: str) -> bool:
        return pollutant in self.pollutants or pollutant.lower() in {
            p.lower() for p in self.pollutants
        }

    def get_factor(
        self,
        vehicle_type: str | None,
        speed_ms: float,
        accel_ms2: float,
        pollutant: Optional[str] = None,
    ) -> float:
        vt = self.normalize_vehicle_type(vehicle_type)
        pol = self._normalize_pollutant(pollutant)
        speed_kmh = self._normalize_speed_kmh(speed_ms)
        accel = self._normalize_accel_ms2(accel_ms2)

        cache_key: LookupKey = (vt, pol, speed_kmh, accel)
        if self.enable_cache and cache_key in self._query_cache:
            return self._query_cache[cache_key]

        factor = self.lookup.get(cache_key)

        if factor is None:
            factor = self.lookup.get((self.default_vehicle_type, pol, speed_kmh, accel))

        if factor is None:
            factor = self.lookup.get((vt, self.default_pollutant, speed_kmh, accel))

        if factor is None:
            factor = self.lookup.get(
                (self.default_vehicle_type, self.default_pollutant, speed_kmh, accel),
                0.0,
            )

        factor = float(factor)

        if self.enable_cache:
            self._query_cache[cache_key] = factor

        return factor

    def get_emission(
        self,
        vehicle_type: str | None,
        speed_ms: float,
        accel_ms2: float,
        sim_step: float,
        pollutant: Optional[str] = None,
        output_unit: str = "mg",
    ) -> float:
        factor_gs = self.get_factor(vehicle_type, speed_ms, accel_ms2, pollutant)
        return float(factor_gs * float(sim_step) * self._mass_factor(output_unit))

    def vectorized_factors(
        self,
        type_idx: np.ndarray,
        pol_idx: np.ndarray,
        speed_idx: np.ndarray,
        accel_idx: np.ndarray,
    ) -> np.ndarray:
        return self._factor_grid[type_idx, pol_idx, speed_idx, accel_idx]

    def get_query_info(
        self,
        vehicle_type: str | None,
        speed_ms: float,
        accel_ms2: float,
        pollutant: Optional[str] = None,
    ) -> EmissionQueryInfo:
        vehicle_type_raw = "" if vehicle_type is None else str(vehicle_type)
        vt = self.normalize_vehicle_type(vehicle_type)
        pol = self._normalize_pollutant(pollutant)
        speed_kmh = self._normalize_speed_kmh(speed_ms)
        accel = self._normalize_accel_ms2(accel_ms2)

        matched_exact = (vt, pol, speed_kmh, accel) in self.lookup
        fallback_used = False

        factor = self.lookup.get((vt, pol, speed_kmh, accel))
        if factor is None:
            fallback_used = True
            factor = self.lookup.get(
                (self.default_vehicle_type, pol, speed_kmh, accel),
                0.0,
            )

        return EmissionQueryInfo(
            vehicle_type_raw=vehicle_type_raw,
            vehicle_type_used=vt,
            pollutant_used=pol,
            speed_ms=float(speed_ms),
            speed_kmh_used=int(speed_kmh),
            accel_ms2_raw=float(accel_ms2),
            accel_ms2_used=float(accel),
            factor_gs=float(factor),
            matched=bool(matched_exact),
            fallback_used=bool(fallback_used),
        )

    def compute_record_emissions(
        self,
        record: VehicleStateRecord,
        pollutants: Sequence[str] | None = None,
        output_unit: str = "mg",
    ) -> Dict[str, float]:
        pollutants = list(pollutants or [self.default_pollutant])
        return {
            pol: self.get_emission(
                vehicle_type=record.vehicle_type_used,
                speed_ms=record.speed_ms,
                accel_ms2=record.accel_ms2,
                sim_step=record.dt,
                pollutant=pol,
                output_unit=output_unit,
            )
            for pol in pollutants
        }

    def summary(self) -> Dict[str, object]:
        return {
            "csv_path": str(self.csv_path),
            "num_rows": 0 if self.df is None else int(len(self.df)),
            "vehicle_types": sorted(self.vehicle_types),
            "pollutants": sorted(self.pollutants),
            "speed_range_kmh": (self.min_speed_kmh, self.max_speed_kmh),
            "accel_range_ms2": (self.min_accel_ms2, self.max_accel_ms2),
            "default_pollutant": self.default_pollutant,
            "default_vehicle_type": self.default_vehicle_type,
            "strict_two_types": self.strict_two_types,
        }

    def print_summary(self) -> None:
        info = self.summary()
        print("=" * 70)
        print("Emission Factor Lookup Summary")
        print("=" * 70)
        for k, v in info.items():
            print(f"{k}: {v}")
        print("=" * 70)

class EmissionEpisodeRecorder:

    def __init__(
        self,
        lookup: EmissionFactorLookup,
        pollutants: Sequence[str] | None = None,
        output_unit: str = "mg",
        factor_vehicle_type_override: Optional[str] = None,
        record_vehicle_step: bool = True,
        enable_lane_mechanism_metrics: bool = False,
        signal_sensitive_halting_speed_mps: float = 0.1,
        signal_sensitive_low_speed_mps: float = 5.0,
        signal_sensitive_restart_accel_ms2: float = 0.5,
    ) -> None:
        self.lookup = lookup
        self.pollutants = list(pollutants or [lookup.default_pollutant])
        self.output_unit = str(output_unit).strip().lower()
        self._mass_factor_scalar = self.lookup._mass_factor(self.output_unit)
        self.record_vehicle_step = bool(record_vehicle_step)
        self.enable_lane_mechanism_metrics = bool(enable_lane_mechanism_metrics)
        self.signal_sensitive_halting_speed_mps = float(
            signal_sensitive_halting_speed_mps
        )
        self.signal_sensitive_low_speed_mps = float(signal_sensitive_low_speed_mps)
        self.signal_sensitive_restart_accel_ms2 = float(
            signal_sensitive_restart_accel_ms2
        )
        if self.signal_sensitive_halting_speed_mps < 0.0:
            raise ValueError("signal_sensitive_halting_speed_mps must be >= 0")
        if self.signal_sensitive_low_speed_mps <= self.signal_sensitive_halting_speed_mps:
            raise ValueError(
                "signal_sensitive_low_speed_mps must be greater than "
                "signal_sensitive_halting_speed_mps"
            )
        if self.signal_sensitive_restart_accel_ms2 < 0.0:
            raise ValueError("signal_sensitive_restart_accel_ms2 must be >= 0")
        if factor_vehicle_type_override is None:
            self.factor_vehicle_type_override: Optional[str] = None
        else:
            override_raw = str(factor_vehicle_type_override).strip().lower()
            if override_raw not in {
                self.lookup.VEHICLE_TYPE_SEDAN,
                self.lookup.VEHICLE_TYPE_TRUCK,
            }:
                raise ValueError(
                    "factor_vehicle_type_override must be None, 'sedan', or 'truck'"
                )
            self.factor_vehicle_type_override = self.lookup.normalize_vehicle_type(override_raw)

        self.episode_id: int = -1
        self.records_by_step: Dict[int, Dict[str, List[object]]] = {}
        self._finalized_step_keys: Set[int] = set()
        self.step_summaries: List[DecisionStepEmission] = []

        self.vehicle_totals: Dict[Tuple[str, str], float] = {}
        self.vehicle_types: Dict[str, str] = {}
        self.vehicle_types_raw: Dict[str, str] = {}
        self.vehicle_distance_m: Dict[str, float] = {}
        self._vehicle_step_rows: List[Dict[str, object]] = []
        self._previous_speed_by_vehicle: Dict[str, float] = {}
        self.last_detail_batch: Optional[VehicleEmissionDetailBatch] = None

    @staticmethod
    def _new_step_buffer() -> Dict[str, List[object]]:
        return {
            "veh_id": [],
            "vtype_raw": [],
            "vtype": [],
            "speed": [],
            "accel": [],
            "lane": [],
            "edge": [],
            "observation_lane": [],
            "signal_sensitive": [],
            "dt": [],
            "sim_time": [],
            "row_id": [],
        }

    def start_episode(self, episode_id: int) -> None:
        self.episode_id = int(episode_id)
        self.records_by_step.clear()
        self._finalized_step_keys.clear()
        self.step_summaries.clear()
        self.vehicle_totals.clear()
        self.vehicle_types.clear()
        self.vehicle_types_raw.clear()
        self.vehicle_distance_m.clear()
        self._vehicle_step_rows.clear()
        self._previous_speed_by_vehicle.clear()
        self.last_detail_batch = None

    def reset(self) -> None:
        self.start_episode(self.episode_id if self.episode_id >= 0 else 0)

    def record_vehicle_state(
        self,
        sim_time: float,
        decision_step: int,
        veh_id: str,
        vehicle_type: str | None,
        speed_ms: float,
        accel_ms2: float,
        lane_id: str | None = "",
        edge_id: str | None = "",
        dt: float = 1.0,
        distance_m: float | None = None,
        in_observation_zone: bool = False,
        observation_lane_id: str = "",
        row_id: int = -1,
    ) -> None:
        if self.episode_id < 0:
            raise RuntimeError("Please call start_episode(episode_id) before recording.")

        veh_id = str(veh_id)
        vt_raw = self.lookup.normalize_vehicle_type(vehicle_type)
        vt_used = (
            vt_raw
            if self.factor_vehicle_type_override is None
            else self.factor_vehicle_type_override
        )
        step_key = int(decision_step)
        step_buffer = self.records_by_step.get(step_key)
        if step_buffer is None:
            step_buffer = self._new_step_buffer()
            self.records_by_step[step_key] = step_buffer
        step_buffer["veh_id"].append(veh_id)
        step_buffer["vtype_raw"].append(vt_raw)
        step_buffer["vtype"].append(vt_used)
        step_buffer["speed"].append(float(speed_ms))
        step_buffer["accel"].append(float(accel_ms2))
        step_buffer["lane"].append("" if lane_id is None else str(lane_id))
        step_buffer["edge"].append("" if edge_id is None else str(edge_id))
        step_buffer["observation_lane"].append(
            str(observation_lane_id) if in_observation_zone else ""
        )
        previous_speed = self._previous_speed_by_vehicle.get(veh_id)
        halting = float(speed_ms) <= self.signal_sensitive_halting_speed_mps
        low_speed = (
            float(speed_ms) > self.signal_sensitive_halting_speed_mps
            and float(speed_ms) <= self.signal_sensitive_low_speed_mps
        )
        restart = (
            previous_speed is not None
            and float(previous_speed) <= self.signal_sensitive_halting_speed_mps
            and float(accel_ms2) >= self.signal_sensitive_restart_accel_ms2
        )
        step_buffer["signal_sensitive"].append(
            bool(self.enable_lane_mechanism_metrics and (halting or low_speed or restart))
        )
        self._previous_speed_by_vehicle[veh_id] = float(speed_ms)
        step_buffer["dt"].append(float(dt))
        step_buffer["sim_time"].append(float(sim_time))
        step_buffer["row_id"].append(int(row_id))

        self.vehicle_types[veh_id] = vt_used
        self.vehicle_types_raw[veh_id] = vt_raw

        if distance_m is not None:
            old = self.vehicle_distance_m.get(veh_id, 0.0)
            self.vehicle_distance_m[veh_id] = max(old, float(distance_m))

    def record_many(self, rows: Iterable[Mapping[str, object]]) -> None:
        for row in rows:
            self.record_vehicle_state(
                sim_time=float(row.get("sim_time", 0.0)),
                decision_step=int(row.get("decision_step", 0)),
                veh_id=str(row.get("veh_id", "")),
                vehicle_type=row.get("vehicle_type", None),
                speed_ms=float(row.get("speed_ms", 0.0)),
                accel_ms2=float(row.get("accel_ms2", 0.0)),
                lane_id=row.get("lane_id", ""),
                edge_id=row.get("edge_id", ""),
                dt=float(row.get("dt", 1.0)),
                distance_m=(
                    None if row.get("distance_m", None) is None
                    else float(row.get("distance_m"))
                ),
                in_observation_zone=bool(row.get("in_observation_zone", False)),
                observation_lane_id=str(row.get("observation_lane_id", "")),
                row_id=int(row.get("row_id", -1)),
            )

    def _records_for_step(self, decision_step: int) -> Dict[str, List[object]]:
        return self.records_by_step.get(int(decision_step), self._new_step_buffer())

    def finalize_decision_step(
        self,
        decision_step: int,
        allow_repeat: bool = False,
    ) -> DecisionStepEmission:
        decision_step = int(decision_step)

        if decision_step in self._finalized_step_keys:
            for s in self.step_summaries:
                if s.decision_step == decision_step:
                    return s
            raise RuntimeError(f"decision_step={decision_step} marked finalized but summary missing.")

        records = self.records_by_step.pop(decision_step, None)
        if records is None:
            records = self._new_step_buffer()
        self.last_detail_batch = None
        n_records = int(len(records.get("veh_id", [])))
        if n_records:
            sim_time_arr = np.asarray(records["sim_time"], dtype=np.float64)
            sim_time_start = float(np.min(sim_time_arr))
            sim_time_end = float(np.max(sim_time_arr))
        else:
            sim_time_start = 0.0
            sim_time_end = 0.0

        summary = DecisionStepEmission(
            episode_id=self.episode_id,
            decision_step=decision_step,
            sim_time_start=float(sim_time_start),
            sim_time_end=float(sim_time_end),
            n_records=n_records,
            n_vehicles=len(set(records.get("veh_id", []))),
            output_unit=self.output_unit,
        )

        if n_records:
            veh_arr = np.asarray(records["veh_id"], dtype=object)
            type_arr = np.asarray(records["vtype"], dtype=object)
            lane_arr = np.asarray(records["lane"], dtype=object)
            edge_arr = np.asarray(records["edge"], dtype=object)
            observation_lane_arr = np.asarray(records["observation_lane"], dtype=object)
            analysis_lane_arr = np.where(
                observation_lane_arr != "",
                observation_lane_arr,
                lane_arr,
            )
            signal_sensitive_arr = np.asarray(records["signal_sensitive"], dtype=bool)
            speed_arr = np.asarray(records["speed"], dtype=np.float64)
            accel_arr = np.asarray(records["accel"], dtype=np.float64)
            dt_arr = np.asarray(records["dt"], dtype=np.float64)

            speed_rounded = np.rint(speed_arr * 3.6).astype(np.int64)
            speed_used = speed_rounded.copy()
            speed_used = np.clip(speed_used, self.lookup.min_speed_kmh, self.lookup.max_speed_kmh)
            speed_idx = (speed_used - int(self.lookup.min_speed_kmh)).astype(np.int64)

            accel_rounded = np.rint(accel_arr * 10.0) / 10.0
            accel_used = accel_rounded.copy()
            accel_used = np.clip(accel_used, self.lookup.min_accel_ms2, self.lookup.max_accel_ms2)
            accel_idx = np.rint((accel_used - self.lookup.min_accel_ms2) / 0.1).astype(np.int64)
            if self.lookup._factor_grid.shape[3] > 0:
                accel_idx = np.clip(accel_idx, 0, self.lookup._factor_grid.shape[3] - 1)

            default_type_idx = self.lookup._type_to_idx.get(
                self.lookup.default_vehicle_type,
                self.lookup._type_to_idx.get(self.lookup.VEHICLE_TYPE_SEDAN, 0),
            )
            type_idx = np.fromiter(
                (self.lookup._type_to_idx.get(str(vt), default_type_idx) for vt in type_arr),
                dtype=np.int64,
                count=n_records,
            )

            unique_lanes, lane_inverse = np.unique(lane_arr, return_inverse=True)
            unique_edges, edge_inverse = np.unique(edge_arr, return_inverse=True)
            unique_types, type_inverse = np.unique(type_arr, return_inverse=True)
            unique_vehicles, vehicle_inverse = np.unique(veh_arr, return_inverse=True)

            for pollutant in dict.fromkeys(self.pollutants):
                pol_canon = self.lookup._normalize_pollutant(pollutant)
                pol_idx_scalar = self.lookup._pol_to_idx[pol_canon]
                factors = self.lookup._factor_grid[
                    type_idx, pol_idx_scalar, speed_idx, accel_idx
                ]
                mass = np.asarray(factors, dtype=np.float64) * dt_arr * float(self._mass_factor_scalar)

                if pol_canon == self.lookup._normalize_pollutant("NOx"):
                    exact = self.lookup._factor_exact_grid[
                        type_idx, pol_idx_scalar, speed_idx, accel_idx
                    ]
                    self.last_detail_batch = VehicleEmissionDetailBatch(
                        decision_step=decision_step,
                        row_ids=np.asarray(records["row_id"], dtype=np.int64),
                        nox_mg=np.asarray(mass, dtype=np.float64),
                        speed_kmh_used=np.asarray(speed_used, dtype=np.int16),
                        accel_ms2_used=np.asarray(accel_used, dtype=np.float32),
                        lookup_clipped=np.asarray(
                            (speed_rounded != speed_used) | (accel_rounded != accel_used),
                            dtype=np.bool_,
                        ),
                        fallback_used=np.asarray(~exact, dtype=np.bool_),
                    )

                summary.total_by_pollutant[pollutant] = float(np.sum(mass, dtype=np.float64))

                lane_buckets = np.zeros(len(unique_lanes), dtype=np.float64)
                np.add.at(lane_buckets, lane_inverse, mass)
                for lane_id, value in zip(unique_lanes.tolist(), lane_buckets.tolist()):
                    summary.by_lane[(str(lane_id), pollutant)] = float(value)

                analysis_lanes, analysis_inverse = np.unique(
                    analysis_lane_arr,
                    return_inverse=True,
                )
                analysis_buckets = np.zeros(len(analysis_lanes), dtype=np.float64)
                np.add.at(analysis_buckets, analysis_inverse, mass)
                for lane_id, value in zip(
                    analysis_lanes.tolist(),
                    analysis_buckets.tolist(),
                ):
                    summary.by_lane_analysis_full[(str(lane_id), pollutant)] = float(value)

                observation_mask = observation_lane_arr != ""
                if np.any(observation_mask):
                    observation_lanes, observation_inverse = np.unique(
                        observation_lane_arr[observation_mask], return_inverse=True
                    )
                    observation_buckets = np.zeros(len(observation_lanes), dtype=np.float64)
                    np.add.at(observation_buckets, observation_inverse, mass[observation_mask])
                    for lane_id, value in zip(
                        observation_lanes.tolist(), observation_buckets.tolist()
                    ):
                        summary.by_lane_observation_zone[(str(lane_id), pollutant)] = float(value)

                sensitive_observation_mask = observation_mask & signal_sensitive_arr
                if np.any(sensitive_observation_mask):
                    sensitive_lanes, sensitive_inverse = np.unique(
                        observation_lane_arr[sensitive_observation_mask],
                        return_inverse=True,
                    )
                    sensitive_buckets = np.zeros(len(sensitive_lanes), dtype=np.float64)
                    np.add.at(
                        sensitive_buckets,
                        sensitive_inverse,
                        mass[sensitive_observation_mask],
                    )
                    for lane_id, value in zip(
                        sensitive_lanes.tolist(), sensitive_buckets.tolist()
                    ):
                        summary.by_lane_observation_zone_signal_sensitive[
                            (str(lane_id), pollutant)
                        ] = float(value)

                edge_buckets = np.zeros(len(unique_edges), dtype=np.float64)
                np.add.at(edge_buckets, edge_inverse, mass)
                for edge_id, value in zip(unique_edges.tolist(), edge_buckets.tolist()):
                    summary.by_edge[(str(edge_id), pollutant)] = float(value)

                type_buckets = np.zeros(len(unique_types), dtype=np.float64)
                np.add.at(type_buckets, type_inverse, mass)
                for vehicle_type, value in zip(unique_types.tolist(), type_buckets.tolist()):
                    summary.by_vehicle_type[(str(vehicle_type), pollutant)] = float(value)

                vehicle_buckets = np.zeros(len(unique_vehicles), dtype=np.float64)
                np.add.at(vehicle_buckets, vehicle_inverse, mass)
                for veh_id, value in zip(unique_vehicles.tolist(), vehicle_buckets.tolist()):
                    value = float(value)
                    key = (str(veh_id), pollutant)
                    summary.by_vehicle[key] = value
                    summary.by_vehicle_delta[key] = value
                    self.vehicle_totals[key] = self.vehicle_totals.get(key, 0.0) + value

                if self.record_vehicle_step:
                    for idx in range(n_records):
                        self._vehicle_step_rows.append({
                            "episode": self.episode_id,
                            "decision_step": decision_step,
                            "sim_time": float(records["sim_time"][idx]),
                            "veh_id": str(veh_arr[idx]),
                            "vehicle_type": str(type_arr[idx]),
                            "lane_id": str(lane_arr[idx]),
                            "edge_id": str(edge_arr[idx]),
                            "observation_lane_id": str(observation_lane_arr[idx]),
                            "pollutant": pollutant,
                            f"emission_{self.output_unit}": float(mass[idx]),
                        })

        for pol, total in summary.total_by_pollutant.items():
            lane_total = sum(v for (_lane_id, p), v in summary.by_lane.items() if p == pol)
            edge_total = sum(v for (_edge_id, p), v in summary.by_edge.items() if p == pol)
            summary.consistency_diffs[f"lane_minus_total_{pol}"] = float(lane_total - total)
            summary.consistency_diffs[f"edge_minus_total_{pol}"] = float(edge_total - total)

        self.step_summaries.append(summary)
        self._finalized_step_keys.add(decision_step)

        return summary

    def finalize_all_steps(self) -> List[DecisionStepEmission]:
        steps = sorted(self.records_by_step.keys())
        for step in steps:
            self.finalize_decision_step(step, allow_repeat=False)
        return list(self.step_summaries)

    def finalize_episode(self) -> EpisodeEmissionResult:
        self.finalize_all_steps()

        step_rows: List[Dict[str, object]] = []
        edge_rows: List[Dict[str, object]] = []
        lane_rows: List[Dict[str, object]] = []

        for s in self.step_summaries:
            step_rows.extend(s.to_flat_rows())

            for (edge_id, pol), val in sorted(s.by_edge.items()):
                edge_rows.append({
                    "episode": s.episode_id,
                    "decision_step": s.decision_step,
                    "sim_time_start": s.sim_time_start,
                    "sim_time_end": s.sim_time_end,
                    "edge_id": edge_id,
                    "pollutant": pol,
                    f"emission_{self.output_unit}": float(val),
                    "n_records": s.n_records,
                    "n_vehicles": s.n_vehicles,
                })

            for (lane_id, pol), val in sorted(s.by_lane.items()):
                lane_rows.append({
                    "episode": s.episode_id,
                    "decision_step": s.decision_step,
                    "sim_time_start": s.sim_time_start,
                    "sim_time_end": s.sim_time_end,
                    "lane_id": lane_id,
                    "pollutant": pol,
                    f"emission_{self.output_unit}": float(val),
                    "n_records": s.n_records,
                    "n_vehicles": s.n_vehicles,
                })

        vehicle_rows: List[Dict[str, object]] = []
        veh_ids = sorted({veh_id for veh_id, _ in self.vehicle_totals.keys()})
        for veh_id in veh_ids:
            vt = self.vehicle_types.get(veh_id, self.lookup.VEHICLE_TYPE_SEDAN)
            vt_raw = self.vehicle_types_raw.get(veh_id, vt)
            dist_m = float(self.vehicle_distance_m.get(veh_id, 0.0))
            for pol in self.pollutants:
                total = float(self.vehicle_totals.get((veh_id, pol), 0.0))
                per_km = total / (dist_m / 1000.0) if dist_m > 1e-9 else 0.0
                vehicle_rows.append({
                    "episode": self.episode_id,
                    "veh_id": veh_id,
                    "vehicle_type": vt,
                    "vehicle_type_raw": vt_raw,
                    "vehicle_type_used": vt,
                    "pollutant": pol,
                    f"total_emission_{self.output_unit}": total,
                    f"total_{pol}_{self.output_unit}": total,
                    "distance_m": dist_m,
                    f"emission_{self.output_unit}_per_km": per_km,
                    f"{pol}_{self.output_unit}_per_km": per_km,
                })

        step_df = pd.DataFrame(step_rows)
        vehicle_df = pd.DataFrame(vehicle_rows)
        edge_step_df = pd.DataFrame(edge_rows)
        lane_step_df = pd.DataFrame(lane_rows)
        vehicle_step_df = pd.DataFrame(self._vehicle_step_rows)

        return EpisodeEmissionResult(
            episode_id=self.episode_id,
            output_unit=self.output_unit,
            step_summaries=list(self.step_summaries),
            vehicle_totals=dict(self.vehicle_totals),
            vehicle_types=dict(self.vehicle_types),
            vehicle_distance_m=dict(self.vehicle_distance_m),
            step_df=step_df,
            vehicle_df=vehicle_df,
            edge_step_df=edge_step_df,
            lane_step_df=lane_step_df,
            vehicle_step_df=vehicle_step_df,
        )

    def get_step_total(
        self,
        decision_step: int,
        pollutant: str = "NOx",
        allow_finalize: bool = True,
    ) -> float:
        for s in self.step_summaries:
            if s.decision_step == int(decision_step):
                return float(s.total_by_pollutant.get(pollutant, 0.0))

        if allow_finalize:
            s = self.finalize_decision_step(decision_step)
            return float(s.total_by_pollutant.get(pollutant, 0.0))

        return 0.0

def build_emission_lookup(
    csv_path: str | Path,
    default_pollutant: str = "NOx",
    default_vehicle_type: str = "sedan",
    enable_cache: bool = True,
    strict_two_types: bool = True,
) -> EmissionFactorLookup:
    return EmissionFactorLookup(
        csv_path=csv_path,
        default_pollutant=default_pollutant,
        default_vehicle_type=default_vehicle_type,
        enable_cache=enable_cache,
        strict_two_types=strict_two_types,
    )

def build_episode_recorder(
    csv_path: str | Path,
    pollutants: Sequence[str] | None = None,
    default_pollutant: str = "NOx",
    output_unit: str = "mg",
    enable_cache: bool = True,
) -> EmissionEpisodeRecorder:
    lookup = build_emission_lookup(
        csv_path=csv_path,
        default_pollutant=default_pollutant,
        enable_cache=enable_cache,
        strict_two_types=True,
    )
    return EmissionEpisodeRecorder(
        lookup=lookup,
        pollutants=pollutants or [default_pollutant],
        output_unit=output_unit,
    )
