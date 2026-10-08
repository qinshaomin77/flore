
from __future__ import annotations

import csv
import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

STEP_FIELDS = [
    "episode",
    "step",
    "global_step",
    "stage",
    "seconds",
    "replay_size",
    "update_due",
    "n_update_groups",
    "n_agents",
]

EPISODE_FIELDS = [
    "episode",
    "wall_time_s",
    "total_profiled_s",
    "unaccounted_s",
    "act_s",
    "act_pct",
    "env_step_s",
    "env_step_pct",
    "compute_rewards_s",
    "compute_rewards_pct",
    "build_next_observations_s",
    "build_next_observations_pct",
    "store_transition_s",
    "store_transition_pct",
    "update_from_replay_s",
    "update_from_replay_pct",
    "logger_s",
    "logger_pct",
    "env_close_s",
    "env_close_pct",
    "flush_s",
    "flush_pct",
    "save_checkpoint_s",
    "save_checkpoint_pct",
    "write_eval_outputs_s",
    "write_eval_outputs_pct",
    "steps",
    "n_agents",
    "replay_size_end",
]

LOGGER_STAGES = {"log_step_emission", "log_update", "log_step", "log_episode"}
FLUSH_STAGES = {"flush_episode_physical", "flush_episode_emission"}
PROFILE_TOTAL_EXCLUDE = {"episode_total"}

def _to_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def _write_csv(path: str, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _to_csv_value(row.get(name, "")) for name in fieldnames})

def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * float(q) / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)

class StageProfiler:

    def __init__(self, enabled: bool, sync_cuda: bool = False):
        self.enabled = bool(enabled)
        self.sync_cuda = bool(sync_cuda)
        self._records: list[dict[str, Any]] = []
        self._episode_rows: list[dict[str, Any]] = []
        self._step_meta: dict[tuple[int, int], dict[str, Any]] = {}
        self._torch: Any = None
        self._cuda_checked = False
        self._cuda_available = False

    @contextmanager
    def timeit(self, stage: str, **meta: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        self._maybe_sync_cuda()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._maybe_sync_cuda()
            self.add_value(stage, time.perf_counter() - start, **meta)

    def add_value(self, stage: str, seconds: float, **meta: Any) -> None:
        if not self.enabled:
            return
        row = {
            "episode": meta.pop("episode", ""),
            "step": meta.pop("step", ""),
            "global_step": meta.pop("global_step", ""),
            "stage": str(stage),
            "seconds": float(seconds),
        }
        row.update(meta)
        self._records.append(row)

    def add_step_meta(
        self,
        episode: int,
        step: int,
        global_step: Optional[int] = None,
        replay_size: Any = "",
        update_due: Any = "",
        n_update_groups: Any = "",
        n_agents: Any = "",
    ) -> None:
        if not self.enabled:
            return
        self._step_meta[(int(episode), int(step))] = {
            "global_step": "" if global_step is None else int(global_step),
            "replay_size": replay_size,
            "update_due": update_due,
            "n_update_groups": n_update_groups,
            "n_agents": n_agents,
        }

    def step_summary(
        self,
        episode: int,
        step: int,
        global_step: Optional[int] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        records = [
            r for r in self._records
            if r.get("episode") == episode and r.get("step") == step
        ]
        by_stage: dict[str, float] = {}
        for row in records:
            stage = str(row.get("stage", ""))
            by_stage[stage] = by_stage.get(stage, 0.0) + float(row.get("seconds", 0.0) or 0.0)
        out: dict[str, Any] = {
            "episode": int(episode),
            "step": int(step),
            "global_step": "" if global_step is None else int(global_step),
            "total_s": float(sum(by_stage.values())),
        }
        out.update({f"{stage}_s": value for stage, value in sorted(by_stage.items())})
        if extra:
            out.update(extra)
        return out

    def episode_summary(
        self,
        episode: int,
        wall_time_s: float,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        records = [r for r in self._records if r.get("episode") == episode]
        stage_totals: dict[str, float] = {}
        for row in records:
            stage = str(row.get("stage", ""))
            stage_totals[stage] = stage_totals.get(stage, 0.0) + float(row.get("seconds", 0.0) or 0.0)

        total_profiled = sum(v for k, v in stage_totals.items() if k not in PROFILE_TOTAL_EXCLUDE)

        def seconds(stage: str) -> float:
            return float(stage_totals.get(stage, 0.0))

        def pct(value: float) -> float:
            return float(value / total_profiled * 100.0) if total_profiled > 0 else 0.0

        logger_s = sum(seconds(stage) for stage in LOGGER_STAGES)
        flush_s = sum(seconds(stage) for stage in FLUSH_STAGES)
        row: dict[str, Any] = {
            "episode": int(episode),
            "wall_time_s": float(wall_time_s),
            "total_profiled_s": float(total_profiled),
            "unaccounted_s": float(wall_time_s - total_profiled),
            "act_s": seconds("act"),
            "act_pct": pct(seconds("act")),
            "env_step_s": seconds("env_step"),
            "env_step_pct": pct(seconds("env_step")),
            "compute_rewards_s": seconds("compute_rewards"),
            "compute_rewards_pct": pct(seconds("compute_rewards")),
            "build_next_observations_s": seconds("build_next_observations"),
            "build_next_observations_pct": pct(seconds("build_next_observations")),
            "store_transition_s": seconds("store_transition"),
            "store_transition_pct": pct(seconds("store_transition")),
            "update_from_replay_s": seconds("update_from_replay"),
            "update_from_replay_pct": pct(seconds("update_from_replay")),
            "logger_s": float(logger_s),
            "logger_pct": pct(logger_s),
            "env_close_s": seconds("env_close"),
            "env_close_pct": pct(seconds("env_close")),
            "flush_s": float(flush_s),
            "flush_pct": pct(flush_s),
            "save_checkpoint_s": seconds("save_checkpoint"),
            "save_checkpoint_pct": pct(seconds("save_checkpoint")),
            "write_eval_outputs_s": seconds("write_eval_outputs"),
            "write_eval_outputs_pct": pct(seconds("write_eval_outputs")),
        }
        if extra:
            row.update(extra)
        return row

    def finish_episode(
        self,
        episode: int,
        wall_time_s: float,
        steps: int,
        n_agents: int,
        replay_size_end: Any = "",
    ) -> None:
        if not self.enabled:
            return
        self.add_value("episode_total", float(wall_time_s), episode=episode, step=-1)
        row = self.episode_summary(
            episode=episode,
            wall_time_s=float(wall_time_s),
            extra={
                "steps": int(steps),
                "n_agents": int(n_agents),
                "replay_size_end": replay_size_end,
            },
        )
        self._episode_rows = [r for r in self._episode_rows if r.get("episode") != int(episode)]
        self._episode_rows.append(row)

    def flush_step_csv(self, path: str) -> None:
        if not self.enabled:
            return
        rows = [self._step_row(row) for row in self._records]
        fieldnames = self._fieldnames(STEP_FIELDS, rows)
        _write_csv(path, rows, fieldnames)

    def flush_episode_csv(self, path: str) -> None:
        if not self.enabled:
            return
        rows = sorted(self._episode_rows, key=lambda r: int(r.get("episode", 0) or 0))
        fieldnames = self._fieldnames(EPISODE_FIELDS, rows)
        _write_csv(path, rows, fieldnames)

    def save_summary_json(self, path: str) -> None:
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        stage_values: dict[str, list[float]] = {}
        for row in self._records:
            stage = str(row.get("stage", ""))
            stage_values.setdefault(stage, []).append(float(row.get("seconds", 0.0) or 0.0))
        total_profiled = sum(
            sum(values)
            for stage, values in stage_values.items()
            if stage not in PROFILE_TOTAL_EXCLUDE
        )
        stages: dict[str, dict[str, Any]] = {}
        for stage in sorted(stage_values):
            values = stage_values[stage]
            total_s = float(sum(values))
            stages[stage] = {
                "total_s": total_s,
                "mean_s": float(total_s / len(values)) if values else 0.0,
                "p50_s": _percentile(values, 50),
                "p90_s": _percentile(values, 90),
                "max_s": float(max(values)) if values else 0.0,
                "count": int(len(values)),
                "pct_of_profiled": float(total_s / total_profiled * 100.0) if total_profiled > 0 else 0.0,
            }
        payload = {
            "enabled": bool(self.enabled),
            "sync_cuda": bool(self.sync_cuda),
            "total_profiled_s": float(total_profiled),
            "stages": stages,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _step_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(row)
        try:
            key = (int(out.get("episode", 0)), int(out.get("step", 0)))
        except Exception:
            key = (0, 0)
        meta = self._step_meta.get(key, {})
        for name, value in meta.items():
            if out.get(name, "") in ("", None):
                out[name] = value
        return out

    def _fieldnames(self, base: list[str], rows: Iterable[Mapping[str, Any]]) -> list[str]:
        fieldnames = list(base)
        existing = set(fieldnames)
        for row in rows:
            for key in row.keys():
                if key not in existing:
                    fieldnames.append(str(key))
                    existing.add(str(key))
        return fieldnames

    def _maybe_sync_cuda(self) -> None:
        if not self.sync_cuda:
            return
        if not self._cuda_checked:
            self._cuda_checked = True
            try:
                import torch  # type: ignore

                self._torch = torch
                self._cuda_available = bool(torch.cuda.is_available())
            except Exception:
                self._torch = None
                self._cuda_available = False
        if self._cuda_available and self._torch is not None:
            self._torch.cuda.synchronize()
