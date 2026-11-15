"""Replay a trusted trained checkpoint on train-derived validation, never blind."""
from pathlib import Path
import os
os.environ.setdefault('HF_HOME',str(Path(__file__).resolve().parent/'private_hf_cache'))
import train_only_guard
import argparse,hashlib,json,time
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer
from neural_train import ReviewModel,Reviews,batches,infer,seed_all
from absa_core import LABELS,reference_splits,binary_predictions
from score_absa import report

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True)
    ap.add_argument('--calibration-dir',type=Path,required=True);ap.add_argument('--source-tune-scores',type=Path,required=True)
    ap.add_argument('--expected-experiment',required=True);ap.add_argument('--train',type=Path,default=ROOT/'train.csv');a=ap.parse_args()
    if a.train.name!='train.csv':raise ValueError('Only train.csv accepted')
    cfg=json.loads((a.model_dir/'config.json').read_text());provenance=json.loads((a.model_dir/'ci_provenance.json').read_text())
    calibration=json.loads((a.calibration_dir/'calibration_report.json').read_text())
    assert provenance['experiment']==a.expected_experiment
    sha=hashlib.sha256(a.train.read_bytes()).hexdigest()
    assert provenance['train_sha256']==calibration['train_sha256']==sha
    assert hashlib.sha256(a.source_tune_scores.read_bytes()).hexdigest()==calibration['tune_scores_sha256']
    seed_all(cfg.get('training_seed',2026));torch.set_num_threads(2)
    model=ReviewModel(cfg['model_name'],cfg['revision'],cfg['head'],cfg['unfreeze_layers'],cfg.get('dropout',.2))
    checkpoint=a.model_dir/cfg.get('checkpoint_file','model.fp16.pt')
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    model.autocast_cpu_bf16=cfg.get('cpu_bf16',False)
    tok=AutoTokenizer.from_pretrained(cfg['model_name'],revision=cfg['revision'],use_fast=cfg.get('fast_tokenizer',False),local_files_only=True)
    train=pd.read_csv(a.train);_,_,valid=reference_splits(train,cfg.get('split_seed',2026))
    y=train[LABELS].to_numpy(int);ds=Reviews(train.text.iloc[valid],y[valid],tok,cfg['segment'],cfg['max_length'],cfg['stride'])
    started=time.perf_counter();logits=infer(model,batches(ds,tok,cfg['batch_size']),torch.device('cpu'))
    p=1/(1+np.exp(-np.clip(logits,-30,30)))
    base=np.asarray(cfg['thresholds']);cut=np.asarray([calibration['thresholds'][c] for c in LABELS])
    gt=train.iloc[valid][['id']+LABELS].reset_index(drop=True)
    def score(thresholds):
        pred=binary_predictions(p,thresholds);sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
        return report(gt,sb,n_boot=1000)
    baseline=score(base);calibrated=score(cut)
    expected=json.loads((a.model_dir/'metrics_after_storage_rounding.json').read_text())
    result={'experiment':a.expected_experiment,'source_run_id':provenance['github_run_id'],
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'checkpoint_storage_dtype':cfg['checkpoint_storage_dtype'],'compute_weight_dtype':'float32',
        'cpu_bf16':model.autocast_cpu_bf16,'cpu_threads':2,'rows':len(valid),'seconds':time.perf_counter()-started,
        'original_threshold_ABSA_F1':baseline['ABSA_F1'],'original_packaged_ABSA_F1':expected['ABSA_F1'],
        'original_threshold_replay_matches':abs(baseline['ABSA_F1']-expected['ABSA_F1'])<1e-12,
        'calibrated_checkpoint_ABSA_F1':calibrated['ABSA_F1'],'calibrated_CI95':calibrated['ABSA_F1_CI95'],
        'delta_ABSA_F1':calibrated['ABSA_F1']-baseline['ABSA_F1'],'thresholds_selected_on':'tune OOF only',
        'outer_validation_used_for_threshold_selection':False,'blind_file_read':False}
    (a.calibration_dir/'checkpoint_replay.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
