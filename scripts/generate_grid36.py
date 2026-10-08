"""Generate all six Grid36 demand levels with the distributed scenario layout."""
from __future__ import annotations
import argparse
import hashlib
import os
import itertools
import json
import math
import random
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / 'data' / 'grid36'
SOURCE_ROUTE = ROOT / 'demand_scenarios_20' / 'grid36_D60_P20.rou.xml'
SCENARIO_SEED = 20260802
P_LEVELS = ('P05', 'P10', 'P20', 'P30', 'P40')
DEMANDS = {'D20': 1835, 'D40': 3670, 'D50': 4587.5, 'D60': 5505, 'D70': 6422.5, 'D80': 7340}
OD_ROWS = (('Horizontal_0_main', 'background', 'npE0_nt05', 'nt00_npW0', None, 239.805), ('Horizontal_0_minor', 'background', 'npW0_nt00', 'nt05_npE0', None, 147.241), ('Horizontal_1_main', 'background', 'npW2_nt20', 'nt25_npE2', None, 239.805), ('Horizontal_1_minor', 'background', 'npE2_nt25', 'nt20_npW2', None, 147.241), ('Horizontal_2_main', 'background', 'npE4_nt45', 'nt40_npW4', None, 227.828), ('Horizontal_2_minor', 'background', 'npW4_nt40', 'nt45_npE4', None, 139.924), ('Diag_1_main', 'main', 'npS0_nt50', 'nt05_npN5', None, 64.199), ('Diag_1_minor', 'main', 'npN5_nt05', 'nt50_npS0', None, 42.949), ('Diag_2_main', 'main', 'npN0_nt00', 'nt55_npS5', None, 64.199), ('Diag_2_minor', 'main', 'npS5_nt55', 'nt00_npN0', None, 42.949), ('Vertical_1_main', 'main', 'npS1_nt51', 'nt01_npN1', None, 43.05), ('Vertical_1_minor', 'main', 'npN1_nt01', 'nt51_npS1', None, 35.633), ('Vertical_2_main', 'main', 'npS3_nt53', 'nt03_npN3', None, 43.05), ('Vertical_2_minor', 'main', 'npN3_nt03', 'nt53_npS3', None, 35.633), ('Blue_Horizontal_main', 'dominant', None, None, 'Blue_Horizontal_main', 104.943), ('Blue_Horizontal_minor', 'dominant', None, None, 'Blue_Horizontal_minor', 74.072), ('Blue_Vertical_main', 'dominant', None, None, 'Blue_Vertical_main', 81.188), ('Blue_Vertical_minor', 'dominant', None, None, 'Blue_Vertical_minor', 61.292))
WARMUP_RATES = (240, 105, 240, 105, 228, 100, 28, 12, 28, 12, 24, 10, 24, 10, 54, 24, 32, 14)
RETREAT_RATES = (75, 165, 75, 165, 71, 157, 22, 48, 22, 48, 26, 58, 26, 58, 39, 87, 40, 88)
TRUCK_PROB = {'P05': (0.012798, 0.102381, 0.121577), 'P10': (0.025595, 0.204761, 0.243154), 'P20': (0.05119, 0.409522, 0.486308), 'P30': (0.076785, 0.614283, 0.729461), 'P40': (0.102381, 0.819044, 0.972615)}

def indent(root: ET.Element) -> None:
    ET.indent(root, space='    ')

def load_static_elements() -> tuple[list[ET.Element], list[ET.Element]]:
    root = ET.parse(SOURCE_ROUTE).getroot()
    initial = [deepcopy(x) for x in root.findall('flow') if x.get('id', '').startswith('init_')]
    routes = [deepcopy(x) for x in root.findall('route')]
    if len(initial) != 360 or len(routes) != 4:
        raise RuntimeError(f'Unexpected source content: {len(initial)} initial flows, {len(routes)} routes')
    for flow in initial:
        for param in list(flow.findall('param')):
            flow.remove(param)
    return (initial, routes)

def episode_schedule() -> list[tuple[str, ...]]:
    permutations = list(itertools.permutations(P_LEVELS))
    random.Random(SCENARIO_SEED).shuffle(permutations)
    schedule = list(permutations)
    for block in range(8):
        base = list(P_LEVELS)
        random.Random(SCENARIO_SEED + block + 1).shuffle(base)
        schedule.extend((tuple((base[(i + shift) % 5] for i in range(5))) for shift in range(5)))
    if len(schedule) != 160:
        raise AssertionError('Schedule must contain 160 episodes')
    return schedule

def add_types(root: ET.Element) -> None:
    for vehicle_id, length, speed, accel, decel, color, reroute in (('sedan', '5', '20', '5', '10', '255,255,0', True), ('truck', '12', '15', '2', '6', '255,0,0', True), ('sedan_fixed', '5', '20', '5', '10', '255,255,0', False), ('truck_fixed', '12', '15', '2', '6', '255,0,0', False)):
        node = ET.SubElement(root, 'vType', id=vehicle_id, length=length, maxSpeed=speed, accel=accel, decel=decel, color=color)
        if reroute:
            ET.SubElement(node, 'param', key='has.rerouting.device', value='true')
            ET.SubElement(node, 'param', key='device.rerouting.period', value='90')
    for p in P_LEVELS:
        for category, probability, fixed in zip(('background', 'main', 'dominant'), TRUCK_PROB[p], (False, False, True)):
            sedan = 1.0 - probability
            ET.SubElement(root, 'vTypeDistribution', id=f'{category}_{p}' + ('_fixed' if fixed else ''), vTypes='sedan_fixed truck_fixed' if fixed else 'sedan truck', probabilities=f'{sedan:.6f} {probability:.6f}')

def rounded_core_rates(demand):
    # Published D50/D70 are interpolated between adjacent integer OD tables.
    total = sum(row[-1] for row in OD_ROWS)
    def base_rates(value):
        return [math.ceil(value * 1.01 * row[-1] / total) for row in OD_ROWS]
    if demand == DEMANDS['D50']:
        return [(a+b)/2 for a,b in zip(base_rates(DEMANDS['D40']),base_rates(DEMANDS['D60']))]
    if demand == DEMANDS['D70']:
        return [(a+b)/2 for a,b in zip(base_rates(DEMANDS['D60']),base_rates(DEMANDS['D80']))]
    return base_rates(demand)


def add_flow(root: ET.Element, prefix: str, row: tuple, begin: int, end: int, rate: int, p: str) -> None:
    name, category, source, destination, route, _ = row
    attrs = {'id': f'{prefix}_{name}', 'begin': str(begin), 'end': str(end), 'vehsPerHour': format(rate, 'g'), 'type': f'{category}_{p}' + ('_fixed' if category == 'dominant' else ''), 'departLane': 'best', 'departPos': 'random_free', 'departSpeed': '0'}
    if route:
        attrs['route'] = route
    else:
        attrs['from'], attrs['to'] = (source, destination)
    ET.SubElement(root, 'flow', attrs)

def build_route(initial: list[ET.Element], routes: list[ET.Element], demand: int, slot_ps: tuple[str, ...]) -> ET.ElementTree:
    root = ET.Element('routes')
    add_types(root)
    for route in routes:
        root.append(deepcopy(route))
    for flow in initial:
        root.append(deepcopy(flow))
    for row, rate in zip(OD_ROWS, WARMUP_RATES):
        add_flow(root, 'warmup', row, 0, 600, math.ceil(rate), 'P20')
    rates = rounded_core_rates(demand)
    for slot, p in enumerate(slot_ps, 1):
        begin = 600 + (slot - 1) * 600
        for row, rate in zip(OD_ROWS, rates):
            add_flow(root, f'core_s{slot}', row, begin, begin + 600, rate, p)
    for row, rate in zip(OD_ROWS, RETREAT_RATES):
        add_flow(root, 'retreat', row, 3600, 3800, math.ceil(rate), 'P20')
    indent(root)
    return ET.ElementTree(root)

def write_xml(tree: ET.ElementTree, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding='UTF-8', xml_declaration=True, short_empty_elements=True)


def add_sedan_types(root: ET.Element) -> None:
    sedan = ET.SubElement(root, 'vType', id='sedan', length='5', maxSpeed='20', accel='5', decel='10', color='255,255,0')
    ET.SubElement(sedan, 'param', key='has.rerouting.device', value='true')
    ET.SubElement(sedan, 'param', key='device.rerouting.period', value='90')
    ET.SubElement(root, 'vType', id='sedan_fixed', length='5', maxSpeed='20', accel='5', decel='10', color='255,255,0')

def add_calibration_flow(root: ET.Element, prefix: str, row: tuple, begin: int, end: int, rate: int) -> None:
    name, category, source, destination, route, _ = row
    attrs = {'id': f'{prefix}_{name}', 'begin': str(begin), 'end': str(end), 'vehsPerHour': format(rate, 'g'), 'type': 'sedan_fixed' if category == 'dominant' else 'sedan', 'departLane': 'best', 'departPos': 'random_free', 'departSpeed': '0'}
    if route:
        attrs['route'] = route
    else:
        attrs['from'], attrs['to'] = (source, destination)
    ET.SubElement(root, 'flow', attrs)

def build_calibration_route(initial: list[ET.Element], routes: list[ET.Element], demand_name: str, demand: int) -> ET.ElementTree:
    root = ET.Element('routes')
    add_sedan_types(root)
    for route in routes:
        root.append(deepcopy(route))
    for flow in initial:
        copied = deepcopy(flow)
        copied.set('type', 'sedan')
        root.append(copied)
    for row, rate in zip(OD_ROWS, WARMUP_RATES):
        add_calibration_flow(root, f'cal_{demand_name}_warmup', row, 0, 600, rate)
    core_rates = rounded_core_rates(demand)
    for slot in range(1, 6):
        begin = 600 + (slot - 1) * 600
        for row, rate in zip(OD_ROWS, core_rates):
            add_calibration_flow(root, f'cal_{demand_name}_core{slot}', row, begin, begin + 600, rate)
    for row, rate in zip(OD_ROWS, RETREAT_RATES):
        add_calibration_flow(root, f'cal_{demand_name}_retreat', row, 3600, 3800, rate)
    indent(root)
    return ET.ElementTree(root)


def write_scenario_config(path, route_path, calibration=False):
    config = ET.Element('configuration')
    inputs = ET.SubElement(config, 'input')
    ET.SubElement(inputs, 'net-file', value='../../truck_sensitive_grid36.net.xml')
    ET.SubElement(inputs, 'route-files', value=os.path.relpath(route_path, path.parent).replace('\\', '/'))
    additional = '../../truck_sensitive_grid36.add.xml'
    if calibration:
        additional += ',../../truck_sensitive_grid36_actuated.tll.xml'
    ET.SubElement(inputs, 'additional-files', value=additional)
    times = ET.SubElement(config, 'time')
    ET.SubElement(times, 'begin', value='0')
    ET.SubElement(times, 'end', value='4200')
    processing = ET.SubElement(config, 'processing')
    ET.SubElement(processing, 'time-to-teleport', value='-1')
    ET.indent(config, space='  ')
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(config).write(path, encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['all', 'train', 'evaluation', 'calibration'], default='all')
    parser.add_argument('--output-dir', type=Path, default=ROOT.parents[1]/'results/generated/grid36')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    # A separate output tree makes generation reviewable and preserves distributed inputs.
    if output == ROOT.resolve() or output.is_relative_to(ROOT.resolve()) or ROOT.resolve().is_relative_to(output):
        parser.error('--output-dir must be separate from data/grid36')
    initial, routes = load_static_elements()
    output.mkdir(parents=True, exist_ok=True)
    for name in ('truck_sensitive_grid36.net.xml', 'truck_sensitive_grid36.add.xml', 'truck_sensitive_grid36_actuated.tll.xml'):
        shutil.copy2(ROOT/name, output/name)
    report = {'kind':args.kind, 'demands':list(DEMANDS), 'scenarios':[], 'unique_training_routes':0}
    unique = {}
    for demand_name, demand in DEMANDS.items():
        if args.kind in ('all', 'train'):
            for episode, slots in enumerate(episode_schedule(), 1):
                name = f'train_grid36_{demand_name}_episode_{episode:03d}'
                route_path = output/'train_rou_xml'/demand_name/(name+'.rou.xml')
                tree = build_route(initial, routes, demand, slots)
                payload = ET.tostring(tree.getroot(), encoding='utf-8')
                digest = hashlib.sha256(payload).hexdigest()
                if digest in unique:
                    route_path = unique[digest]
                else:
                    unique[digest] = route_path
                    write_xml(tree, route_path)
                config_path = output/'train_sumocfg'/demand_name/(name+'.sumocfg')
                write_scenario_config(config_path, route_path)
                report['scenarios'].append({'config':str(config_path.relative_to(output)), 'route':str(route_path.relative_to(output))})
        if args.kind in ('all', 'evaluation'):
            for p in P_LEVELS:
                name = f'eval_{demand_name}_{p}'
                route_path = output/'evaluation_rou_xml'/demand_name/(name+'.rou.xml')
                tree = build_route(initial, routes, demand, (p,)*5)
                # Evaluation routes explicitly use the published HBEFA4 classes.
                for vehicle in tree.getroot().findall('vType'):
                    vehicle.set('emissionClass', 'HBEFA4/RT_gt14-20t_Euro-VI_D-E' if vehicle.get('id').startswith('truck') else 'HBEFA4/PC_petrol_Euro-6ab')
                write_xml(tree, route_path)
                config_path = output/'evaluation_sumocfg'/demand_name/(name+'.sumocfg')
                write_scenario_config(config_path, route_path)
                report['scenarios'].append({'config':str(config_path.relative_to(output)), 'route':str(route_path.relative_to(output))})
        if args.kind in ('all', 'calibration'):
            name = f'calibration_grid36_{demand_name}_sedan'
            route_path = output/'calibration_rou_xml'/demand_name/(name+'.rou.xml')
            write_xml(build_calibration_route(initial, routes, demand_name, demand), route_path)
            config_path = output/'calibration_sumocfg'/demand_name/(name+'_actuated.sumocfg')
            write_scenario_config(config_path, route_path, calibration=True)
            report['scenarios'].append({'config':str(config_path.relative_to(output)), 'route':str(route_path.relative_to(output))})
    report['unique_training_routes'] = len(unique)
    (output/'generation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'scenarios':len(report['scenarios']),'unique_training_routes':len(unique)},indent=2))

if __name__ == '__main__':
    main()
