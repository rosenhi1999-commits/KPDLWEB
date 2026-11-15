"""Package one selected checkpoint compactly, then score the rounded weights."""
from pathlib import Path
import train_only_guard
import argparse,hashlib,json,shutil
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer
from neural_train import ReviewModel,Reviews,batches,infer,seed_all
from absa_core import LABELS,reference_splits,binary_predictions
from score_absa import report

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--metadata',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--out',type=Path,default=ROOT/'best_model');a=ap.parse_args()
    cfg=json.loads((a.metadata/'config.json').read_text());seed_all(cfg.get('training_seed',2026));torch.set_num_threads(cfg.get('cpu_threads',2))
    original=torch.load(a.model,map_location='cpu',weights_only=True)
    compact={k:v.half() if v.is_floating_point() else v for k,v in original.items()};del original
    a.out.mkdir(parents=True,exist_ok=False);torch.save(compact,a.out/'model.fp16.pt')
    if (a.out/'model.fp16.pt').stat().st_size>350_000_000:raise ValueError('Compact checkpoint exceeds artifact budget')
    model=ReviewModel(cfg['model_name'],cfg['revision'],cfg.get('head','joint'),cfg.get('unfreeze_layers',-1),cfg.get('dropout',.2))
    model.load_state_dict(compact);del compact
    flags=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    cfg['training_cpu_bf16']=cfg.get('cpu_bf16',False)
    cfg['cpu_bf16']=cfg['training_cpu_bf16'] and ('avx512_bf16' in flags or 'amx_bf16' in flags)
    model.autocast_cpu_bf16=cfg['cpu_bf16']
    tok=AutoTokenizer.from_pretrained(cfg['model_name'],revision=cfg['revision'],use_fast=cfg.get('fast_tokenizer',False))
    train=pd.read_csv(ROOT/'train.csv');_,_,valid=reference_splits(train,cfg.get('split_seed',2026))
    y=train[LABELS].to_numpy(int);ds=Reviews(train.text.iloc[valid],y[valid],tok,cfg['segment'],cfg['max_length'],cfg['stride'])
    logits=infer(model,batches(ds,tok,cfg['batch_size']),torch.device('cpu'))
    cut=np.asarray(cfg['thresholds']);logit_cut=np.log(cut/(1-cut))
    pred=binary_predictions(logits,logit_cut);gt=train.iloc[valid][['id']+LABELS].reset_index(drop=True)
    sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
    metrics=report(gt,sb,n_boot=1000);metrics.update(blind_file_read=False,checkpoint_storage_dtype='float16',compute_weight_dtype='float32',inference_cpu_bf16=cfg['cpu_bf16'])
    (a.out/'metrics_after_storage_rounding.json').write_text(json.dumps(metrics,indent=2))
    cfg.update(checkpoint_file='model.fp16.pt',checkpoint_storage_dtype='float16',compute_weight_dtype='float32')
    (a.out/'config.json').write_text(json.dumps(cfg,indent=2))
    for name in ['run_info.json','ci_provenance.json','selection_provenance.json']:
        if (a.metadata/name).exists():shutil.copy2(a.metadata/name,a.out/name)
    (a.out/'README.txt').write_text('Train-only selected encoder checkpoint. Load into ReviewModel as float32; thresholds in config are probabilities. See metrics_after_storage_rounding.json. No blind predictions or original data included. Model is selection fit, not an all-train production refit.\n')
    print(json.dumps({'packaged_ABSA_F1':metrics['ABSA_F1'],'checkpoint_bytes':(a.out/'model.fp16.pt').stat().st_size,'blind_file_read':False}),flush=True)
