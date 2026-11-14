"""Execute one real full-finetuning trial on a standard GitHub CPU runner."""
from pathlib import Path
import train_only_guard
import argparse,hashlib,json,os,re,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parent
PUBLIC_FILES=['config.json','metrics.json','history.csv','run_info.json','progress.json',
              'tune_scores.npz','validation_scores.npz','ci_provenance.json']

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--target',required=True);a=ap.parse_args()
    if not re.fullmatch(r'PhoBERT_lan\d+',a.source) or not re.fullmatch(r'PhoBERT_lan\d+',a.target):raise ValueError('Invalid experiment name')
    folder=ROOT/'private_ci_runs'/a.target;folder.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/a.source/'config.json').read_text())
    flags=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    cfg.update(allow_cpu=True,cpu_bf16='avx512_bf16' in flags or 'amx_bf16' in flags,
        cpu_threads=min(4,os.cpu_count() or 1),max_train_seconds=18000,log_every_batches=25,
        source_configuration=a.source,experiment=a.target,status='running',production_refit=False)
    (folder/'config.json').write_text(json.dumps(cfg,indent=2))
    provenance={'experiment':a.target,'source_configuration':a.source,
        'training_code_sha':os.environ.get('GITHUB_SHA'),'github_run_id':os.environ.get('GITHUB_RUN_ID'),
        'train_sha256':hashlib.sha256((ROOT/'train.csv').read_bytes()).hexdigest(),
        'neural_train_sha256':hashlib.sha256((ROOT/'neural_train.py').read_bytes()).hexdigest(),
        'full_encoder_trainable':cfg['unfreeze_layers']==-1,'blind_file_read':False}
    (folder/'ci_provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps({'experiment':a.target,'allow_cpu':True,'cpu_threads':cfg['cpu_threads'],
        'cpu_bf16':cfg['cpu_bf16'],'epochs_requested':cfg['epochs'],'full_encoder_trainable':True}),flush=True)
    result=None
    try:
        result=subprocess.run([sys.executable,str(ROOT/'neural_train.py'),'--config',str(folder/'config.json'),
            '--train',str(ROOT/'train.csv')],check=False)
    finally:
        public=ROOT/'public_results'/a.target;public.mkdir(parents=True,exist_ok=True)
        for name in PUBLIC_FILES:
            if (folder/name).is_file():shutil.copy2(folder/name,public/name)
    if result is None or result.returncode:raise SystemExit(result.returncode if result else 1)
    metrics=json.loads((folder/'metrics.json').read_text())
    assert metrics['blind_file_read'] is False and (folder/'model.pt').exists()
    summary=os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary,'a') as f:
            f.write(f"### {a.target}\n\nABSA-F1 validation: **{metrics['ABSA_F1']:.6f}**\n\n")
            f.write(f"Completed epochs: {metrics['epochs_completed']}; selected epoch: {metrics['best_epoch']}. Train-only.\n")
