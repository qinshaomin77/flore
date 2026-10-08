"""Generate Kunshan base (K0) or freight-enhanced (K1) demand."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
SCENARIO = 'K1_freight_enhanced'
DEFAULT_DEMAND_SCALE = 1.15
TIME_BINS = [(0, 300, 1854), (300, 600, 3694), (600, 900, 5361), (900, 1200, 6431), (1200, 1500, 6865), (1500, 1800, 6282), (1800, 2100, 5392), (2100, 2400, 4284), (2400, 2700, 3640), (2700, 3000, 4831), (3000, 3300, 5644), (3300, 3400, 1623)]
TRUCK_SHARE_TARGETS = [0.1, 0.12, 0.18, 0.23, 0.3, 0.33, 0.35, 0.37, 0.37, 0.34, 0.3, 0.2]
EXPECTED_TRUCK_VPH = [185, 443, 965, 1479, 2060, 2073, 1887, 1585, 1347, 1643, 1693, 325]
CORE_ROUTES = {'Freight_Horizontal_EB', 'Freight_Horizontal_WB', 'Freight_Vertical_NB', 'Freight_Vertical_SB'}
GROUP_ORDER = {'sedan': 0, 'truck_main': 1, 'truck_dominant': 2}
CSV_FIELDS = ['stage', 'begin_s', 'end_s', 'duration_s', 'target_total_vph', 'actual_total_vph', 'target_truck_share', 'expected_truck_vph', 'actual_expected_truck_vph', 'actual_expected_truck_share', 'sedan_flow_vph', 'truck_main_flow_vph', 'truck_dominant_flow_vph', 'expected_vehicle_count', 'expected_truck_count', 'n_sedan_flows', 'n_truck_main_flows', 'n_truck_dominant_flows']

class ValidationError(RuntimeError):
    """Raised when a critical generation or validation rule fails."""

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def unique(items: Iterable[ET.Element], label: str) -> None:
    ids = [item.get('id') for item in items]
    duplicates = sorted((key for key, count in Counter(ids).items() if count > 1))
    if duplicates:
        raise ValidationError(f'Duplicate {label} id(s): {duplicates}')

def largest_remainder(total: int, weights: list[float], tie_keys: list[str] | None=None) -> list[int]:
    """Allocate an integer total using Hamilton's largest-remainder method."""
    if total < 0:
        raise ValidationError(f'Cannot allocate negative total: {total}')
    if not weights or any((weight < 0 for weight in weights)):
        raise ValidationError('Weights must be a non-empty list of non-negative values')
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValidationError('Cannot allocate from a group whose original total is zero')
    exact = [total * weight / weight_sum for weight in weights]
    result = [math.floor(value) for value in exact]
    keys = tie_keys or [str(index) for index in range(len(weights))]
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - result[i]), keys[i]))
    for index in order[:total - sum(result)]:
        result[index] += 1
    if sum(result) != total:
        raise ValidationError('Largest-remainder allocation failed to preserve total')
    return result

def classify(flow: ET.Element) -> str:
    flow_id = flow.get('id', '')
    flow_type = flow.get('type')
    if flow_id.startswith('sedan_') and flow_type == 'sedan':
        return 'sedan'
    if flow_id.startswith('diag_') and flow_type == 'truck_main':
        return 'truck_main'
    if flow_id.startswith('freight_') and flow_type == 'truck_dominant':
        return 'truck_dominant'
    raise ValidationError(f'Unrecognized dynamic flow group: id={flow_id!r}, type={flow_type!r}')

def scaled_targets(demand_scale: float) -> list[int]:
    factor = demand_scale / DEFAULT_DEMAND_SCALE
    return [round(total * factor) for _, _, total in TIME_BINS]

def group_vehicle_totals(total: int, truck_share: float) -> dict[str, int]:
    expected_truck = total * truck_share
    exact = [total - 0.3 * expected_truck / 0.8 - 0.7 * expected_truck / 0.95, 0.3 * expected_truck / 0.8, 0.7 * expected_truck / 0.95]
    if exact[0] <= 0:
        raise ValidationError(f'Calculated sedan demand is not positive: {exact[0]}')
    allocated = largest_remainder(total, exact, ['sedan', 'truck_main', 'truck_dominant'])
    return dict(zip(('sedan', 'truck_main', 'truck_dominant'), allocated))

def source_dynamic_groups(root: ET.Element) -> dict[tuple[int, int], dict[str, list[ET.Element]]]:
    grouped: dict[tuple[int, int], dict[str, list[ET.Element]]] = defaultdict(lambda: defaultdict(list))
    for item in root.findall('flow'):
        if item.get('id', '').startswith('init_'):
            continue
        begin, end = (int(float(item.get('begin'))), int(float(item.get('end'))))
        grouped[begin, end][classify(item)].append(item)
    expected_bins = {(begin, end) for begin, end, _ in TIME_BINS}
    if set(grouped) != expected_bins:
        raise ValidationError(f'Source time bins differ from specification: {sorted(grouped)}')
    for key, groups in grouped.items():
        counts = {name: len(groups.get(name, [])) for name in GROUP_ORDER}
        if counts != {'sedan': 14, 'truck_main': 4, 'truck_dominant': 4}:
            raise ValidationError(f'Source bin {key} has invalid group counts: {counts}')
    return grouped

def build_enhanced(source_root: ET.Element, demand_scale: float) -> tuple[ET.ElementTree, list[dict[str, object]]]:
    source_groups = source_dynamic_groups(source_root)
    output_root = ET.Element('routes')
    for tag in ('vType', 'vTypeDistribution'):
        for item in source_root.findall(tag):
            output_root.append(copy.deepcopy(item))
    routes = source_root.findall('route')
    route_ids = {item.get('id') for item in routes}
    if not CORE_ROUTES.issubset(route_ids):
        raise ValidationError(f'Missing core route(s): {sorted(CORE_ROUTES - route_ids)}')
    for item in routes:
        output_root.append(copy.deepcopy(item))
    init_flows = [item for item in source_root.findall('flow') if item.get('id', '').startswith('init_')]
    if len(init_flows) != 252 or sum((int(item.get('number', '0')) for item in init_flows)) != 252:
        raise ValidationError('Source must contain exactly 252 one-vehicle init flows')
    for item in init_flows:
        output_root.append(copy.deepcopy(item))
    rows: list[dict[str, object]] = []
    targets = scaled_targets(demand_scale)
    for stage, ((begin, end, _), target_total, truck_share) in enumerate(zip(TIME_BINS, targets, TRUCK_SHARE_TARGETS)):
        groups = source_groups[begin, end]
        totals = group_vehicle_totals(target_total, truck_share)
        output_flows: list[ET.Element] = []
        for group in ('sedan', 'truck_main', 'truck_dominant'):
            originals = sorted(groups[group], key=lambda item: item.get('id', ''))
            weights = [int(item.get('vehsPerHour', '0')) for item in originals]
            values = largest_remainder(totals[group], weights, [item.get('id', '') for item in originals])
            for original, value in zip(originals, values):
                item = copy.deepcopy(original)
                item.set('vehsPerHour', str(value))
                output_flows.append(item)
        output_flows.sort(key=lambda item: (GROUP_ORDER[classify(item)], item.get('id', '')))
        output_root.extend(output_flows)
        actual = {name: sum((int(item.get('vehsPerHour')) for item in output_flows if classify(item) == name)) for name in GROUP_ORDER}
        actual_total = sum(actual.values())
        actual_truck = 0.8 * actual['truck_main'] + 0.95 * actual['truck_dominant']
        duration = end - begin
        rows.append({'stage': stage, 'begin_s': begin, 'end_s': end, 'duration_s': duration, 'target_total_vph': target_total, 'actual_total_vph': actual_total, 'target_truck_share': truck_share, 'expected_truck_vph': EXPECTED_TRUCK_VPH[stage] * demand_scale / DEFAULT_DEMAND_SCALE, 'actual_expected_truck_vph': actual_truck, 'actual_expected_truck_share': actual_truck / actual_total, 'sedan_flow_vph': actual['sedan'], 'truck_main_flow_vph': actual['truck_main'], 'truck_dominant_flow_vph': actual['truck_dominant'], 'expected_vehicle_count': actual_total * duration / 3600, 'expected_truck_count': actual_truck * duration / 3600, 'n_sedan_flows': 14, 'n_truck_main_flows': 4, 'n_truck_dominant_flows': 4})
    ET.indent(output_root, space='    ')
    return (ET.ElementTree(output_root), rows)

def static_validate(tree: ET.ElementTree, net_file: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    root = tree.getroot()
    if root.tag != 'routes':
        raise ValidationError(f'Root element must be routes, found {root.tag}')
    unique(root.findall('vType'), 'vType')
    unique(root.findall('vTypeDistribution'), 'vTypeDistribution')
    unique(root.findall('route'), 'route')
    unique(root.findall('flow'), 'flow')
    net_root = ET.parse(net_file).getroot()
    edges = {item.get('id') for item in net_root.findall('edge')}
    routes = {item.get('id'): item for item in root.findall('route')}
    for route_id in CORE_ROUTES:
        missing = [edge for edge in routes[route_id].get('edges', '').split() if edge not in edges]
        if missing:
            raise ValidationError(f'Route {route_id} contains unknown edge(s): {missing}')
    init_flows, dynamic = ([], [])
    for item in root.findall('flow'):
        if item.get('id', '').startswith('init_'):
            init_flows.append(item)
        else:
            dynamic.append(item)
            if int(float(item.get('begin'))) >= 3400:
                raise ValidationError(f"Dynamic flow starts at/after 3400 s: {item.get('id')}")
            if int(item.get('vehsPerHour', '-1')) < 0:
                raise ValidationError(f"Negative flow rate: {item.get('id')}")
        for attr in ('from', 'to'):
            if item.get(attr) and item.get(attr) not in edges:
                raise ValidationError(f"Flow {item.get('id')} references unknown {attr} edge {item.get(attr)}")
        if item.get('route') and item.get('route') not in routes:
            raise ValidationError(f"Flow {item.get('id')} references unknown route {item.get('route')}")
    if len(init_flows) != 252 or sum((int(item.get('number', '0')) for item in init_flows)) != 252:
        raise ValidationError('Output init vehicle count is not 252')
    by_bin: dict[tuple[int, int], list[ET.Element]] = defaultdict(list)
    for item in dynamic:
        by_bin[int(float(item.get('begin'))), int(float(item.get('end')))].append(item)
    targets = scaled_targets(rows and float(rows[0]['target_total_vph']) / TIME_BINS[0][2] * DEFAULT_DEMAND_SCALE or DEFAULT_DEMAND_SCALE)
    for index, (begin, end, _) in enumerate(TIME_BINS):
        items = by_bin[begin, end]
        counts = Counter((classify(item) for item in items))
        expected_counts = Counter({'sedan': 14, 'truck_main': 4, 'truck_dominant': 4})
        if len(items) != 22 or counts != expected_counts:
            raise ValidationError(f'Output bin {(begin, end)} has invalid flow counts: {dict(counts)}')
        total = sum((int(item.get('vehsPerHour')) for item in items))
        target = int(rows[index]['target_total_vph'])
        if total != target:
            raise ValidationError(f'Output bin {(begin, end)} total {total} != target {target}')
        share = float(rows[index]['actual_expected_truck_share'])
        if abs(share - TRUCK_SHARE_TARGETS[index]) > 0.005:
            raise ValidationError(f'Output bin {(begin, end)} truck-share error exceeds 0.5 percentage points')
    dynamic_count = sum((float(row['expected_vehicle_count']) for row in rows))
    truck_count = sum((float(row['expected_truck_count']) for row in rows))
    overall_share = truck_count / dynamic_count
    if not 4500 <= dynamic_count <= 4650:
        raise ValidationError(f'Dynamic expected vehicle count {dynamic_count:.3f} is not approximately 4568')
    if not 0.28 <= overall_share <= 0.285:
        raise ValidationError(f'Overall expected truck share {overall_share:.6f} is outside 28.0%-28.5%')
    return {'status': 'passed', 'root_routes': True, 'unique_ids': True, 'init_flow_count': len(init_flows), 'init_vehicle_count': 252, 'dynamic_flow_count': len(dynamic), 'flows_per_time_bin': 22, 'group_counts_per_bin': {'sedan': 14, 'truck_main': 4, 'truck_dominant': 4}, 'target_totals_exact': True, 'truck_share_tolerance_passed': True, 'no_dynamic_flow_after_3400': True, 'edge_references_valid': True, 'route_references_valid': True, 'core_routes_preserved': True, 'dynamic_expected_vehicle_count': dynamic_count, 'overall_expected_truck_share': overall_share}

def sumo_validate(net_file: Path, route_file: Path) -> tuple[str, str]:
    executable = shutil.which('sumo')
    if not executable:
        return ('skipped_no_sumo', 'SUMO executable not found')
    command = [executable, '--net-file', str(net_file), '--route-files', str(route_file), '--route-steps', '3600', '--begin', '0', '--end', '1', '--no-step-log', 'true', '--duration-log.statistics', 'true']
    completed = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise ValidationError(f'SUMO validation failed with exit code {completed.returncode}:\n{output}')
    return ('passed', output)

def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def summary_json(args: argparse.Namespace, rows: list[dict[str, object]], validation: dict[str, object], sumo_status: str, source_hash: str) -> dict[str, object]:
    dynamic_count = sum((float(row['expected_vehicle_count']) for row in rows))
    truck_count = sum((float(row['expected_truck_count']) for row in rows))
    return {'scenario': SCENARIO, 'source_route_file': args.source_rou.name, 'output_route_file': args.output_rou.name, 'demand_scale': args.demand_scale, 'simulation_begin_s': 0, 'demand_end_s': 3400, 'simulation_end_s': 3600, 'clearance_duration_s': 200, 'decision_interval_s': 10, 'init_vehicle_count': 252, 'dynamic_expected_vehicle_count': dynamic_count, 'overall_expected_truck_count': truck_count, 'overall_expected_truck_share': truck_count / dynamic_count, 'peak_total_vph': max((int(row['actual_total_vph']) for row in rows)), 'peak_truck_vph': max((float(row['actual_expected_truck_vph']) for row in rows)), 'high_exposure_begin_s': 1200, 'high_exposure_end_s': 3300, 'high_exposure_duration_s': 2100, 'source_route_sha256': source_hash, 'sumo_validation': sumo_status, 'validation': validation, 'time_bins': rows}

def print_summary(args: argparse.Namespace, rows: list[dict[str, object]], sumo_status: str) -> None:
    dynamic = sum((float(row['expected_vehicle_count']) for row in rows))
    trucks = sum((float(row['expected_truck_count']) for row in rows))
    print('=== Kunshan Freight Enhanced Demand ===')
    print(f'Source ROU: {args.source_rou}')
    print(f'Output ROU: {args.output_rou}')
    print('Init vehicles: 252')
    print(f'Dynamic expected vehicles: {dynamic:.3f}')
    print(f'Overall expected truck count: {trucks:.3f}')
    print(f'Overall expected truck share: {trucks / dynamic:.4%}')
    print(f"Peak total flow: {max((int(row['actual_total_vph']) for row in rows))} veh/h")
    print(f"Peak expected truck flow: {max((float(row['actual_expected_truck_vph']) for row in rows)):.1f} veh/h")
    print('High-exposure interval: 1200-3300 s')
    print('Clearance interval: 3400-3600 s')
    print('Static validation: passed')
    print(f'SUMO validation: {sumo_status}')
    print(f'Summary CSV: {args.summary_csv}')
    print(f'Summary JSON: {args.summary_json}')
    for row in rows:
        print(f"{row['begin_s']}-{row['end_s']} | {row['actual_total_vph']} | {row['sedan_flow_vph']} | {row['truck_main_flow_vph']} | {row['truck_dominant_flow_vph']} | {row['actual_expected_truck_vph']:.1f} | {row['actual_expected_truck_share']:.4%}")
import argparse
import collections
import xml.etree.ElementTree as ET
from pathlib import Path
base_BINS = [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500), (1500, 1800), (1800, 2100), (2100, 2400), (2400, 2700), (2700, 3000), (3000, 3300), (3300, 3400)]
base_BASE_MAIN = [144, 324, 510, 600, 570, 433, 273, 149, 95, 115, 126, 45]
base_BASE_MINOR = [95, 163, 192, 226, 271, 269, 264, 203, 206, 423, 600, 149]
base_BACKGROUND_ODS = [('H1', 'npW1_1_nt10', 'nt05_npE0', 'npE0_nt05', 'nt10_npW1_1', 0.7), ('H2', 'npW4_nt40', 'nt55_npE5', 'npE5_nt55', 'nt40_npW4', 0.95), ('H3', 'npW6_nt61', 'nt65_npE6', 'npE6_nt65', 'nt61_npW6', 1.0), ('H4', 'npW7_nt71', 'nt73_npE7', 'npE7_nt73', 'nt71_npW7', 0.7), ('V1', 'npS1_nt71', 'nt11_npN1', 'npN1_nt11', 'nt71_npS1', 0.8), ('V2', 'npS2_nt72', 'nt12_npN2', 'npN2_nt12', 'nt72_npS2', 1.0), ('V3', 'npS5_nt65', 'nt05_npN5', 'npN5_nt05', 'nt65_npS5', 0.9)]
base_DIAGONAL_ODS = [('D1', 'npW0_nt05', 'nt71_npS1', 'npS1_nt71', 'nt05_npW0', [13, 19, 29, 42, 59, 79, 100, 125, 150, 104, 50, 17], [11, 17, 29, 50, 100, 150, 125, 88, 63, 42, 25, 13]), ('D2', 'npS5_nt65', 'nt11_npN1', 'npN1_nt11', 'nt65_npS5', [10, 15, 24, 34, 47, 64, 80, 100, 120, 84, 40, 14], [9, 14, 24, 40, 80, 120, 100, 70, 50, 34, 20, 10])]
base_FREIGHT = {'FH_EB': [54, 80, 120, 160, 220, 300, 385, 340, 260, 180, 100, 40], 'FH_WB': [24, 35, 50, 70, 100, 140, 190, 306, 250, 180, 100, 50], 'FV_NB': [32, 60, 100, 140, 190, 240, 289, 260, 200, 140, 80, 40], 'FV_SB': [14, 25, 40, 60, 85, 120, 170, 304, 250, 180, 100, 50]}
base_FREIGHT_ROUTES = {'Freight_Horizontal_EB': 'npW3_nt30 nt30_nt31 nt31_nt32 nt32_nt33 nt33_nt34 nt34_nt35 nt35_npE3', 'Freight_Horizontal_WB': 'npE3_nt35 nt35_nt34 nt34_nt33 nt33_nt32 nt32_nt31 nt31_nt30 nt30_npW3', 'Freight_Vertical_NB': 'npS4_nt64 nt64_nt54 nt54_nt34 nt34_nt04 nt04_npN4', 'Freight_Vertical_SB': 'npN4_nt04 nt04_nt34 nt34_nt54 nt54_nt64 nt64_npS4'}

def base_add_comment(root: ET.Element, text: str) -> None:
    root.append(ET.Comment(f' {text} '))

def base_initial_edge_destinations(net_root: ET.Element) -> list[tuple[str, int, list[str]]]:
    edges = [e for e in net_root.findall('edge') if not e.get('id', '').startswith(':')]
    degree: collections.Counter[str] = collections.Counter()
    for edge in edges:
        degree.update((edge.get('from'), edge.get('to')))
    boundary = {node for node, count in degree.items() if count == 2}
    internal = [(edge.get('id'), len(edge.findall('lane'))) for edge in edges if edge.get('from') not in boundary and edge.get('to') not in boundary]
    exits = {edge.get('id') for edge in edges if edge.get('from') not in boundary and edge.get('to') in boundary}
    adjacency: dict[str, set[str]] = collections.defaultdict(set)
    for connection in net_root.findall('connection'):
        source, target = (connection.get('from'), connection.get('to'))
        if source and target and (not source.startswith(':')) and (not target.startswith(':')):
            adjacency[source].add(target)
    result, cursor = ([], 0)
    for edge_id, lane_count in internal:
        seen, pending = ({edge_id}, [edge_id])
        while pending:
            new_edges = adjacency[pending.pop()] - seen
            seen.update(new_edges)
            pending.extend(new_edges)
        reachable = sorted(exits & seen)
        if not reachable:
            raise ValueError(f'No reachable boundary exit from internal edge {edge_id}')
        picks = []
        for _ in range(lane_count):
            picks.append(reachable[cursor % len(reachable)])
            cursor += 1
        result.append((edge_id, lane_count, picks))
    return result

def base_flow(root: ET.Element, **attrs: object) -> None:
    ET.SubElement(root, 'flow', {key: str(value) for key, value in attrs.items()})

def base_build(net_path: Path) -> ET.ElementTree:
    net_root = ET.parse(net_path).getroot()
    root = ET.Element('routes')
    sedan = ET.SubElement(root, 'vType', {'id': 'sedan', 'length': '5', 'maxSpeed': '20', 'accel': '5', 'decel': '10', 'color': '255,255,0'})
    ET.SubElement(sedan, 'param', {'key': 'has.rerouting.device', 'value': 'true'})
    ET.SubElement(sedan, 'param', {'key': 'device.rerouting.period', 'value': '90'})
    ET.SubElement(root, 'vType', {'id': 'truck', 'length': '12', 'maxSpeed': '15', 'accel': '2', 'decel': '6', 'color': '255,0,0'})
    ET.SubElement(root, 'vTypeDistribution', {'id': 'truck_dominant', 'vTypes': 'sedan truck', 'probabilities': '0.05 0.95'})
    ET.SubElement(root, 'vTypeDistribution', {'id': 'truck_main', 'vTypes': 'sedan truck', 'probabilities': '0.20 0.80'})
    for route_id, edges in base_FREIGHT_ROUTES.items():
        ET.SubElement(root, 'route', {'id': route_id, 'edges': edges})
    base_add_comment(root, 'Initial state: one sedan per lane on every internal directed edge')
    init_id = 1
    for edge_id, lane_count, destinations in base_initial_edge_destinations(net_root):
        for lane in range(lane_count):
            base_flow(root, id=f'init_{init_id}', **{'from': edge_id, 'to': destinations[lane]}, begin=0, end=1, number=1, type='sedan', departLane=lane, departPos='random_free', departSpeed=0)
            init_id += 1
    for index, (begin, end) in enumerate(base_BINS):
        base_add_comment(root, f'T{index}: {begin}-{end} s')
        for name, fm, tm, fr, tr, scale in base_BACKGROUND_ODS:
            base_flow(root, id=f'sedan_{name}_main_T{index}', **{'from': fm, 'to': tm}, begin=begin, end=end, vehsPerHour=round(base_BASE_MAIN[index] * scale), type='sedan', departLane='best', departPos='random_free', departSpeed=0)
            base_flow(root, id=f'sedan_{name}_minor_T{index}', **{'from': fr, 'to': tr}, begin=begin, end=end, vehsPerHour=round(base_BASE_MINOR[index] * scale), type='sedan', departLane='best', departPos='random_free', departSpeed=0)
        for name, fm, tm, fr, tr, main_rates, minor_rates in base_DIAGONAL_ODS:
            base_flow(root, id=f'diag_{name}_main_T{index}', **{'from': fm, 'to': tm}, begin=begin, end=end, vehsPerHour=main_rates[index], type='truck_main', departLane='best', departPos='random_free', departSpeed=0)
            base_flow(root, id=f'diag_{name}_minor_T{index}', **{'from': fr, 'to': tr}, begin=begin, end=end, vehsPerHour=minor_rates[index], type='truck_main', departLane='best', departPos='random_free', departSpeed=0)
        for key, route_id in (('FH_EB', 'Freight_Horizontal_EB'), ('FH_WB', 'Freight_Horizontal_WB'), ('FV_NB', 'Freight_Vertical_NB'), ('FV_SB', 'Freight_Vertical_SB')):
            base_flow(root, id=f'freight_{key}_T{index}', route=route_id, begin=begin, end=end, vehsPerHour=base_FREIGHT[key][index], type='truck_dominant', departLane='best', departPos='random_free', departSpeed=0)
    ET.indent(root, space='    ')
    return ET.ElementTree(root)

def base_validate(tree: ET.ElementTree) -> None:
    root = tree.getroot()
    flows = root.findall('flow')
    begins = [float(item.get('begin')) for item in flows]
    assert len(flows) == 516, f'Expected 516 flows, found {len(flows)}'
    assert sum((int(item.get('number', '0')) for item in flows)) == 252
    assert all((left <= right for left, right in zip(begins, begins[1:])))
    assert max((int(item.get('vehsPerHour')) for item in flows if item.get('id', '').startswith('diag_D1'))) == 150
    assert max((int(item.get('vehsPerHour')) for item in flows if item.get('id', '').startswith('diag_D2'))) == 120


def main():
    here = Path(__file__).resolve().parents[1]/'data/kunshan'
    default_output = here.parents[1]/'results/generated/kunshan'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['base','enhanced'], default='enhanced')
    parser.add_argument('--source-rou', type=Path, default=here/'kunshan.rou.xml')
    parser.add_argument('--net-file', type=Path, default=here/'kunshan.net.xml')
    parser.add_argument('--output-dir', type=Path, default=default_output)
    parser.add_argument('--demand-scale', type=float, default=DEFAULT_DEMAND_SCALE)
    parser.add_argument('--check', action='store_true', help='Validate without writing')
    args=parser.parse_args()
    if args.demand_scale <= 0:
        parser.error('--demand-scale must be positive')
    output=args.output_dir.resolve()
    if output == here.resolve() or output.is_relative_to(here.resolve()) or here.resolve().is_relative_to(output):
        parser.error('--output-dir must be separate from data/kunshan')
    if args.mode=='base':
        tree=base_build(args.net_file)
        # Preserve the finalized per-OD rounding table and explicit emission classes.
        corrections = {'sedan_V3_minor_T0': '85', 'sedan_H2_main_T2': '485', 'sedan_V1_minor_T2': '153', 'sedan_H2_minor_T3': '214', 'sedan_H1_minor_T5': '189', 'sedan_H2_main_T5': '412', 'sedan_H4_minor_T5': '189', 'sedan_V1_main_T5': '347', 'sedan_H2_main_T6': '260', 'sedan_V1_main_T6': '219', 'sedan_V3_minor_T6': '237', 'sedan_H1_main_T7': '105', 'sedan_H4_main_T7': '105', 'sedan_V1_minor_T7': '163', 'sedan_H1_main_T8': '67', 'sedan_H2_main_T8': '91', 'sedan_H2_minor_T8': '195', 'sedan_H4_main_T8': '67', 'sedan_H1_main_T9': '81', 'sedan_H2_main_T9': '110', 'sedan_H4_main_T9': '81', 'sedan_V3_main_T10': '114', 'sedan_H1_main_T11': '32', 'sedan_H1_minor_T11': '105', 'sedan_H4_main_T11': '32', 'sedan_H4_minor_T11': '105', 'sedan_V3_main_T11': '41'}
        for flow in tree.getroot().findall('flow'):
            if flow.get('id') in corrections:
                flow.set('vehsPerHour', corrections[flow.get('id')])
        emissions = {'sedan': 'HBEFA4/PC_petrol_Euro-6ab', 'truck': 'HBEFA4/RT_gt14-20t_Euro-VI_D-E'}
        for vehicle in tree.getroot().findall('vType'):
            if vehicle.get('id') in emissions:
                vehicle.set('emissionClass', emissions[vehicle.get('id')])
        base_validate(tree)
        if not args.check:
            output.mkdir(parents=True,exist_ok=True)
            tree.write(output/'kunshan.rou.xml',encoding='UTF-8',xml_declaration=True)
        print('Kunshan base demand validation passed')
        return
    args.output_rou=output/'kunshan_freight_enhanced.rou.xml'
    args.summary_csv=output/'kunshan_freight_enhanced_summary.csv'
    args.summary_json=output/'kunshan_freight_enhanced_summary.json'
    try:
        if args.output_rou.resolve()==args.source_rou.resolve():
            raise ValidationError('Output ROU must not overwrite source K0 ROU')
        source_hash=sha256(args.source_rou)
        tree,rows=build_enhanced(ET.parse(args.source_rou).getroot(),args.demand_scale)
        validation=static_validate(tree,args.net_file,rows)
        if args.check:
            print('Kunshan enhanced demand static validation passed')
            return
        output.mkdir(parents=True,exist_ok=True)
        tree.write(args.output_rou,encoding='UTF-8',xml_declaration=True)
        status,message=sumo_validate(args.net_file,args.output_rou)
        validation.update(source_route_unchanged=sha256(args.source_rou)==source_hash,sumo_validation=status,sumo_output=message)
        if not validation['source_route_unchanged']:
            raise ValidationError('Source K0 ROU changed during generation')
        write_csv(args.summary_csv,rows)
        args.summary_json.write_text(json.dumps(summary_json(args,rows,validation,status,source_hash),ensure_ascii=False,indent=2),encoding='utf-8')
        print_summary(args,rows,status)
    except (ValidationError,ET.ParseError,OSError,ValueError) as error:
        parser.exit(1,f'ERROR: {error}\n')

if __name__=='__main__':
    main()
