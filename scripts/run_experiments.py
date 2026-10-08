"""Run named, explicit experiment suites. Never silently replace existing results."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--suite',required=True)
    p.add_argument('--manifest', help='Optional custom experiment suite file')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',help='Existing batch directory; skip only exactly matching successful tasks')
    args=p.parse_args()
    manifests=[ROOT/args.manifest] if args.manifest else sorted((ROOT/'configs').glob('*/experiments.yaml'))
    manifest={'suites':{}}
    for path in manifests:
        payload=yaml.safe_load(path.read_text(encoding='utf-8'))
        overlap=set(manifest['suites']) & set(payload['suites'])
        if overlap: p.error('Duplicate suites: '+str(sorted(overlap)))
        manifest['suites'].update(payload['suites'])
    if args.suite not in manifest['suites']: p.error('Unknown suite: '+args.suite)
    tasks=manifest['suites'][args.suite]
    batch=Path(args.resume).resolve() if args.resume else ROOT/'results/batches'/f'{args.suite}_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}'
    source_hash=hashlib.sha256()
    for folder in ('code','scripts'):
        for file in sorted((ROOT/folder).rglob('*.py')):
            source_hash.update(file.relative_to(ROOT).as_posix().encode())
            source_hash.update(file.read_bytes())
    source_hash.update((ROOT/'data/manifest.json').read_bytes())
    completed={}
    for index,task in enumerate(tasks):
        flags=list(map(str,task['args']))
        signature=hashlib.sha256((source_hash.hexdigest()+json.dumps(task,sort_keys=True)).encode())
        for option in ('--config','--base-config','--presslight-config'):
            if option in flags: signature.update((ROOT/flags[flags.index(option)+1]).read_bytes())
        dependency=task.get('checkpoint_from')
        if dependency:
            if args.dry_run:
                flags+=['--checkpoint',f'<best checkpoint from {dependency}>']
            else:
                if dependency not in completed: raise ValueError('Uncompleted dependency: '+dependency)
                candidates=list(Path(completed[dependency]['output_dir']).rglob('checkpoint_best.pt'))
                if len(candidates)!=1: raise ValueError(f'Expected one best checkpoint for {dependency}: {candidates}')
                with candidates[0].open('rb') as handle: signature.update(hashlib.file_digest(handle,'sha256').digest())
                flags+=['--checkpoint',str(candidates[0])]
        fingerprint=signature.hexdigest()
        if args.dry_run:
            print(json.dumps({'task':task['name'],'args':flags,'checkpoint_from':dependency},ensure_ascii=False),flush=True)
            continue
        batch.mkdir(parents=True,exist_ok=True)
        record=batch/f'{index:04d}_{task["name"]}.json'
        if record.exists():
            old=json.loads(record.read_text(encoding='utf-8'))
            if old.get('fingerprint')==fingerprint and old.get('returncode')==0:
                completed[task['name']]=old
                continue
        output=batch/task['name']/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        option='--log-root' if flags[0] in ('train','evaluate') else '--output-root'
        if option in flags:
            at=flags.index(option);flags[at+1]=str(output)
        else: flags.extend([option,str(output)])
        cmd=[sys.executable,str(ROOT/'run.py'),*flags]
        print(json.dumps({'task':task['name'],'command':cmd},ensure_ascii=False),flush=True)
        result=subprocess.run(cmd,cwd=ROOT,check=False)
        info={'task':task['name'],'command':cmd,'returncode':result.returncode,
              'fingerprint':fingerprint,'output_dir':str(output)}
        record.write_text(json.dumps(info,indent=2),encoding='utf-8')
        if result.returncode: raise SystemExit(result.returncode)
        completed[task['name']]=info

if __name__=='__main__': main()
