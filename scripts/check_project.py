"""Check distributed data/config inputs without reading private source folders."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hashes', action='store_true', help='Verify every distributed asset SHA256')
    p.add_argument('--static-only', action='store_true', help='Do not require installed scientific dependencies')
    p.add_argument('--models', action='store_true', help='Check public RL weight compatibility')
    p.add_argument('--network', choices=['grid36','kunshan'])
    p.add_argument('--update-manifest', action='store_true', help='Refresh the manifest after intentional input changes')
    args = p.parse_args()
    if args.update_manifest:
        update_manifest()
    errors = []
    for file in ROOT.rglob('*.py'):
        if any(part in file.parts for part in ('.venv','results','.git')): continue
        try: ast.parse(file.read_text(encoding='utf-8-sig'), filename=str(file))
        except SyntaxError as exc: errors.append(str(exc))
    configs = list((ROOT/'data').rglob('*.sumocfg'))
    for file in configs:
        try:
            tree = ET.parse(file)
            for node in tree.findall('./input/*'):
                if 'file' not in node.tag: continue
                for name in node.get('value','').split(','):
                    if name.strip() and not (file.parent / name.strip()).is_file():
                        errors.append(f'{file.relative_to(ROOT)} -> missing {name}')
        except Exception as exc: errors.append(str(exc))
    if args.hashes:
        manifest = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
        for entry in manifest['assets']:
            file = ROOT / entry['path']
            if not file.is_file(): errors.append('Missing asset: ' + entry['path']); continue
            with file.open('rb') as handle: actual = hashlib.file_digest(handle,'sha256').hexdigest()
            if actual != entry['sha256']: errors.append('SHA256 mismatch: ' + entry['path'])
    versions = {'python':sys.version.split()[0]}
    if not args.static_only:
        for package in ('torch','numpy','pandas','PyYAML','pyarrow','traci','sumolib'):
            try: versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: errors.append('Install dependency: '+package)
        sumo=shutil.which('sumo')
        if sumo:
            versions['sumo']=subprocess.check_output([sumo,'--version'],text=True).splitlines()[0]
        else: errors.append('sumo executable not on PATH')
        if not errors:
            import yaml
            for file in (ROOT/'configs').rglob('*.yaml'):
                payload=yaml.safe_load(file.read_text(encoding='utf-8')) or {}
                env=payload.get('env',{})
                for key in ('sumo_cfg','net_xml','add_xml','intersection_groups_json','emission_factor_csv','baseline_sumo_cfg','baseline_net_xml'):
                    value=env.get(key)
                    if value and not (ROOT/value).is_file(): errors.append(f'{file.relative_to(ROOT)} {key}: {value}')
                value=payload.get('emission_risk',{}).get('thresholds_json')
                if value and not (ROOT/value).is_file(): errors.append(f'{file.relative_to(ROOT)} threshold: {value}')
                for key,value in payload.get('paths',{}).items():
                    if key in ('sumo_cfg','net_xml','base_add_xml','intersection_groups_json','presslight_add_xml','detector_map_json','emission_thresholds_json') and value and not (ROOT/value).is_file():
                        errors.append(f'{file.relative_to(ROOT)} {key}: {value}')
    print(json.dumps({'sumocfg_count':len(configs),'versions':versions,'errors':errors},indent=2,ensure_ascii=False))
    if errors: raise SystemExit(1)
    if args.models:
        check_models(args.network)



def update_manifest():
    path=ROOT/'data/manifest.json'
    old=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    assets=[]
    for base in ('data','models','configs'):
        for file in sorted((ROOT/base).rglob('*')):
            if not file.is_file() or file==path or file.suffix=='.md': continue
            with file.open('rb') as handle: digest=hashlib.file_digest(handle,'sha256').hexdigest()
            assets.append({'path':file.relative_to(ROOT).as_posix(),'bytes':file.stat().st_size,'sha256':digest})
    old.update(schema_version=1,assets=assets)
    old['origins']=[entry for entry in old.get('origins',[]) if (ROOT/entry['path']).is_file()]
    path.write_text(json.dumps(old,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'{len(assets)} assets, {sum(row["bytes"] for row in assets):,} bytes')

def check_models(network):
    import yaml
    args = argparse.Namespace(network=network)
    if network is None:
        for selected in ('grid36','kunshan'):
            subprocess.run([sys.executable,__file__,'--models','--network',selected],cwd=ROOT,check=True)
        return
    sys.path.insert(0,str(ROOT/('code' if args.network=='grid36' else 'code/kunshan')))
    from config import get_config
    from agent import MGMQAgentManager
    from network_parser import parse_network
    from obs_reward import ObsRewardBuilder
    import torch
    torch.set_num_threads(1)
    suites={}
    for path in sorted((ROOT/'configs').glob('*/experiments.yaml')):
        suites.update(yaml.safe_load(path.read_text(encoding='utf-8'))['suites'])
    cases={}
    for tasks in suites.values():
        for task in tasks:
            flags=task['args']
            if flags[0] not in ('evaluate','window'): continue
            if '--checkpoint' not in flags: continue
            network=flags[flags.index('--network')+1] if '--network' in flags else 'grid36'
            if network!=args.network: continue
            checkpoint=flags[flags.index('--checkpoint')+1]
            config=flags[flags.index('--config')+1]
            threshold=flags[flags.index('--thresholds-json')+1] if '--thresholds-json' in flags else None
            cases[checkpoint]=(config,threshold)
    if args.network=='grid36':
        cases['models/grid36/nox_only.pt']=('configs/grid36/objective/flore_nox_only.yaml',None)
        cases['models/grid36/nox_dominant.pt']=('configs/grid36/objective/flore_emission_dominant.yaml',None)
        for number in ('10','50','100','200'):
            checkpoint=f'models/grid36_lambda_sensitive/full_lambda_{number}p0.pt'
            if (ROOT/checkpoint).is_file():
                cases[checkpoint]=('configs/grid36/objective/flore_emission_dominant.yaml' if number=='200' else f'configs/grid36/sensitivity/flore_lambda_{number}p0.yaml',None)
    errors=[]
    reports=[]
    for checkpoint,(config,threshold) in cases.items():
        cfg=get_config(str(ROOT/config))
        for field in ('sumo_cfg','net_xml','add_xml','intersection_groups_json','emission_factor_csv'):
            setattr(cfg.env,field,str(ROOT/getattr(cfg.env,field)))
        cfg.emission_risk.thresholds_json=str(ROOT/(threshold or cfg.emission_risk.thresholds_json))
        cfg.device='cpu'
        cfg.validate()
        net=parse_network(cfg.env.net_xml,cfg.env.add_xml,cfg.env.intersection_groups_json)
        ObsRewardBuilder(cfg,net)
        agent=MGMQAgentManager(cfg,net,device='cpu')
        try:
            payload=torch.load(ROOT/checkpoint,map_location='cpu',weights_only=False)
            agent._validate_checkpoint_compatibility(payload,checkpoint)
            agent.online_bank.load_state_dict(payload['online_state_dict'],strict=True)
            agent.target_bank.load_state_dict(payload['target_state_dict'],strict=True)
            report={'checkpoint':checkpoint,'config':config,'status':'passed'}
        except Exception as exc:
            report={'checkpoint':checkpoint,'config':config,'status':'failed','reason':str(exc)}
            errors.append(report)
        reports.append(report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    out=ROOT/'results'/f'model_preflight_{args.network}.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(reports,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'{args.network}: {len(cases)-len(errors)}/{len(cases)} RL weights passed')
    if errors: raise SystemExit(1)


if __name__ == '__main__': main()
