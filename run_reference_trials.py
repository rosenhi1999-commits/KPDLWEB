"""Run configured full-finetuning ablations; never open blind data."""
from pathlib import Path
import train_only_guard
import argparse,json,subprocess,sys
import torch

ROOT=Path(__file__).resolve().parent
NAMES=[f'PhoBERT_lan{i}' for i in range(3,11)]
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',choices=NAMES,default=NAMES)
    ap.add_argument('--train',type=Path,default=ROOT/'train.csv');ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
    if a.train.name!='train.csv':raise ValueError('Only train.csv accepted')
    for name in a.runs:
        p=ROOT/name/'config.json';cfg=json.loads(p.read_text())
        assert cfg['seed']==2026 and cfg['max_length']==256 and cfg['unfreeze_layers']==-1
        print(name,json.dumps({'changes':cfg['changed_parameters'],'status':cfg['status'],
            'effective_review_batch_size':cfg['batch_size']*cfg['accumulation']},ensure_ascii=False),flush=True)
    if a.dry_run:raise SystemExit(0)
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable: configurations inspected; full fine-tuning has not been run')
    for name in a.runs:
        if (ROOT/name/'metrics.json').exists():
            print('Already measured; skipping',name,flush=True);continue
        subprocess.run([sys.executable,str(ROOT/'neural_train.py'),'--config',str(ROOT/name/'config.json'),
            '--train',str(a.train.resolve())],check=True)
