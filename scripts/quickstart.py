"""Short, real training (including optimizer updates), then evaluate its weight."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--network',choices=['grid36','kunshan'],default='grid36')
    args=p.parse_args()
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_id=f'{args.network}_{stamp}'
    output=ROOT/'results/quickstart'/run_id
    output.mkdir(parents=True)
    cfg=yaml.safe_load((ROOT/('configs/grid36/flore_quickstart.yaml' if args.network=='grid36' else 'configs/kunshan/flore.yaml')).read_text(encoding='utf-8'))
    cfg['device']='cpu'
    cfg['env'].update(episode_duration=300,save_sumo_aux_outputs=False,save_vehicle_state_output=False)
    cfg['ddqn'].update(total_episodes=2,batch_size=4,min_replay_size=8,replay_memory_size=256,epsilon_decay_steps=60,target_update_interval=10)
    cfg['log'].update(log_root=str(output/'training'),run_id='model',console_mode='compact')
    config=output/'config.yaml'
    config.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    def call(arguments): subprocess.run([sys.executable,str(ROOT/'run.py'),*arguments],cwd=ROOT,check=True)
    call(['train','--network',args.network,'--config',str(config)])
    weights=list((output/'training').rglob('checkpoint_latest.pt'))
    if len(weights)!=1: raise RuntimeError(f'Expected one latest checkpoint, found {weights}')
    # Learning must happen: do not mistake an empty rollout for a training smoke test.
    import torch
    checkpoint=torch.load(weights[0],map_location='cpu',weights_only=False)
    if checkpoint.get('update_count',0)<=0: raise RuntimeError('Quickstart performed no optimizer updates')
    flags=['evaluate','--network',args.network,'--config',str(config),'--checkpoint',str(weights[0]),
           '--eval-episodes','1','--log-root',str(output/'evaluation'),'--run-id','trained_model']
    flags += ['--workers','1'] if args.network=='grid36' else ['--serial']
    call(flags)
    print(json.dumps({'status':'passed','output':str(output),'optimizer_updates':checkpoint['update_count']},indent=2))

if __name__=='__main__': main()
