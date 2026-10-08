"""Portable entry point. Every child process runs from this repository root."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

def main():
    p = argparse.ArgumentParser(description='FLORE training, evaluation and calibration')
    p.add_argument('command', choices=['train','evaluate','window','baseline','calibrate',
                                     'presslight-train','presslight-evaluate','doctor','quickstart'])
    p.add_argument('--network', choices=['grid36','kunshan'], default='grid36')
    args, extra = p.parse_known_args()
    def call(script, flags):
        subprocess.run([sys.executable, str(ROOT / script), *flags], cwd=ROOT, check=True)
    def default(flags, name, value):
        if name not in flags:
            flags.extend([name, value])
    flags = list(extra)
    engine = 'code' if args.network == 'grid36' else 'code/kunshan'
    config = f'configs/{args.network}/base.yaml'
    if args.command == 'doctor':
        call('scripts/check_project.py', flags)
    elif args.command == 'quickstart':
        if flags: p.error('quickstart does not accept extra arguments')
        call('scripts/quickstart.py', ['--network', args.network])
    elif args.command == 'train':
        default(flags, '--config', config)
        call(f'{engine}/train.py', flags)
    elif args.command == 'evaluate':
        default(flags, '--config', config)
        if '--checkpoint' not in flags:
            p.error('evaluate requires --checkpoint; this prevents accidental use of the wrong model')
        call(f'{engine}/evaluate.py', flags)
    elif args.command == 'window':
        if args.network != 'grid36': p.error('window metrics are implemented for Grid36; use evaluate for Kunshan')
        default(flags, '--config', config)
        default(flags, '--controller', 'rl')
        default(flags, '--case-name', 'flore')
        call('code/evaluate_windowed.py', flags)
    elif args.command == 'baseline':
        b = argparse.ArgumentParser()
        b.add_argument('--controller', choices=['actuated','max_pressure','truck_weighted_max_pressure'], required=True)
        ctrl, flags = b.parse_known_args(flags)
        if args.network == 'kunshan':
            if ctrl.controller == 'actuated':
                default(flags, '--config', 'configs/kunshan/baseline/kunshan_actuated_eval.yaml')
                call('code/kunshan/evaluate_actuated.py', flags)
            else:
                default(flags, '--config', config)
                default(flags, '--controller', ctrl.controller)
                call('code/kunshan/evaluate_max_pressure.py', flags)
        else:
            default(flags, '--controller', ctrl.controller)
            default(flags, '--config', config)
            default(flags, '--case-name', ctrl.controller)
            if ctrl.controller != 'actuated':
                default(flags, '--detector-map', 'data/grid36/maxpressure_detector_map.json')
                default(flags, '--sumocfg', 'data/grid36/truck_sensitive_grid36_maxpressure.sumocfg')
            call('code/evaluate_windowed.py', flags)
    elif args.command == 'calibrate':
        script = 'calibrate_nox_max.py' if args.network == 'grid36' else 'calibrate_spatial_nox_max.py'
        call('scripts/' + script, flags)
    else:
        default(flags, '--base-config', config)
        default(flags, '--presslight-config', 'configs/presslight/presslight_' +
                ('grid36' if args.network == 'grid36' else 'kunshan_freight_enhanced') + '.yaml')
        script = 'train_presslight.py' if args.command == 'presslight-train' else 'evaluate_presslight.py'
        call('code/presslight/' + script, flags)

if __name__ == '__main__':
    main()
