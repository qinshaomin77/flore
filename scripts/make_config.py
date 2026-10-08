"""Create a complete experiment config using dot.key=value overrides."""
import argparse
import json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--set',action='append',default=[])
    args=p.parse_args()
    payload=yaml.safe_load((ROOT/args.base).read_text(encoding='utf-8'))
    for setting in args.set:
        key,separator,value=setting.partition('=')
        if not separator: p.error('--set must be key=value')
        target=payload;parts=key.split('.')
        for part in parts[:-1]: target=target[part]
        if parts[-1] not in target: p.error('Unknown field: '+key)
        target[parts[-1]]=yaml.safe_load(value)
    path=ROOT/args.output;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(payload,sort_keys=False),encoding='utf-8')
    print(path)
if __name__=='__main__': main()
