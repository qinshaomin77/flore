"""E2-based movement Max-Pressure controller (no learning dependencies)."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Mapping
import numpy as np
from network_parser import Movement, build_maxpressure_movements, build_phase_movement_map, build_turn_lookup

@dataclass
class MovementPressure:
    movement_id: str
    upstream_halting: float
    downstream_halting_by_lane: Dict[str, float]
    downstream_weights: Dict[str, float]
    effective_downstream_halting: float
    pressure: float

@dataclass
class MaxPressureActionInfo:
    tl_id: str; current_phase: int; selected_phase: int
    phase_pressures: list; masked_phase_pressures: list; selected_pressure: float
    action_mask: list; tie_count: int; tie_kept_current: bool

class MaxPressureController:
    def __init__(self, net_info, detector_map: Mapping, config):
        self.net_info, self.detector_map, self.config = net_info, detector_map, config
        self.movements = build_maxpressure_movements(net_info)
        self.phase_movements = build_phase_movement_map(net_info, self.movements)
        self.turn_lookup = build_turn_lookup(net_info, self.movements)
        self.by_tls, self.by_lane = defaultdict(list), defaultdict(list)
        for m in self.movements.values(): self.by_tls[m.tl_id].append(m); self.by_lane[m.from_lane].append(m)
        self.reset()
    def reset(self): self.last_diagnostics = {}
    def collect_e2_snapshot(self, env):
        ids=set()
        for role in ("upstream","downstream"):
            for lane_map in self.detector_map[role].values(): ids.update(lane_map.values())
        return env.get_lanearea_snapshot(sorted(ids))
    def classify_shared_lane_queues(self, env, snapshots):
        queues, rows, threshold = defaultdict(float), [], float(self.config.halting_speed_threshold_mps)
        for lane, movements in self.by_lane.items():
            if len(movements)<=1: continue
            tl_id=movements[0].tl_id; det=self.detector_map["upstream"][tl_id][lane]
            unknown=0; counts=defaultdict(int)
            for vid in snapshots[det]["vehicle_ids"]:
                c=env.get_vehicle_route_context(vid)
                if c["speed_mps"]>=threshold: continue
                route,idx=c["route"],c["route_index"]
                if c["lane_id"]!=lane or idx<0 or idx>=len(route)-1 or route[idx]!=c["road_id"]: unknown+=1; continue
                mid=self.turn_lookup.get((lane,route[idx+1]))
                if mid is None: unknown+=1; continue
                queues[mid]+=1.0; counts[self.movements[mid].direction]+=1
            reported=int(snapshots[det]["halting_number"]); classified=sum(counts.values())
            rows.append({"tl_id":tl_id,"from_lane":lane,"detector_id":det,"e2_vehicle_count":snapshots[det]["vehicle_number"],
                         "e2_halting_count":reported,"straight_halting":counts["s"],"left_halting":counts["l"],
                         "right_halting":counts["r"],"uturn_halting":counts["t"],"unknown_halting":unknown,
                         "classification_difference":reported-classified-unknown})
        return queues,rows
    def build_movement_queues(self, env, snapshots):
        shared,rows=self.classify_shared_lane_queues(env,snapshots); queues={}
        for m in self.movements.values():
            if len(self.by_lane[m.from_lane])==1:
                det=self.detector_map["upstream"][m.tl_id][m.from_lane]; queues[m.movement_id]=float(snapshots[det]["halting_number"])
            else: queues[m.movement_id]=float(shared[m.movement_id])
        return queues,rows
    def compute_movement_pressures(self, queues, snapshots):
        result={}
        for m in self.movements.values():
            downstream={lane:float(snapshots[self.detector_map["downstream"][m.tl_id][lane]]["halting_number"]) for lane in m.to_lanes}
            weights={lane:1.0/len(downstream) for lane in downstream}; effective=sum(weights[k]*v for k,v in downstream.items()); upstream=float(queues[m.movement_id])
            result[m.movement_id]=MovementPressure(m.movement_id,upstream,downstream,weights,effective,upstream-effective)
        return result
    def compute_phase_pressures(self, pressures):
        return {tl:[sum(pressures[mid].pressure for mid in mids) for mids in self.phase_movements[tl]] for tl in self.net_info.intersection_ids}
    def select_actions(self, env, raw_obs, phase_pressures):
        actions,infos={},{}
        for tl,values in phase_pressures.items():
            mask=np.asarray(env.compute_action_mask(tl),dtype=bool)
            if not mask.any(): raise RuntimeError(f"No legal Max-Pressure phase for {tl}")
            masked=np.asarray(values,dtype=float); masked[~mask]=-np.inf; best=float(np.max(masked)); candidates=np.flatnonzero(masked==best).tolist(); current=int(raw_obs[tl].current_phase); selected=current if current in candidates else int(min(candidates))
            actions[tl]=selected; infos[tl]=MaxPressureActionInfo(tl,current,selected,list(map(float,values)),list(map(float,masked)),best,mask.tolist(),len(candidates),current in candidates)
        return actions,infos
    def act(self,env,raw_obs,episode=0,step=0):
        snapshots=self.collect_e2_snapshot(env); queues,rows=self.build_movement_queues(env,snapshots); pressures=self.compute_movement_pressures(queues,snapshots); phase=self.compute_phase_pressures(pressures); actions,infos=self.select_actions(env,raw_obs,phase)
        diagnostics={"snapshots":snapshots,"movement_pressures":pressures,"shared_lane_classifications":rows,"metadata_note":"Reward and NOx components are diagnostics only and do not affect Max-Pressure action selection."}; self.last_diagnostics=diagnostics
        return actions,infos,diagnostics
