"""Refit a recipe selected before refit on outer80; never calibrate in-sample."""
from pathlib import Path
import train_only_guard
import argparse,hashlib,json,math,shutil,time
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer,get_cosine_schedule_with_warmup
from neural_train import ReviewModel,Reviews,batches,infer,seed_all
from reference_optimization import optimizer_for,positive_weights,supervised_loss
from absa_core import LABELS,reference_splits,binary_predictions
from calibrate_reference import load_tune
from score_absa import report

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--selection-dir',type=Path,required=True);ap.add_argument('--train',type=Path,required=True)
    ap.add_argument('--check-only',action='store_true');a=ap.parse_args()
    if a.train.name!='train.csv':raise ValueError('Train only')
    cfg=json.loads(a.config.read_text());out=a.config.parent
    selection=json.loads((a.selection_dir/'config.json').read_text());sm=json.loads((a.selection_dir/'metrics.json').read_text())
    train=pd.read_csv(a.train);sha=hashlib.sha256(a.train.read_bytes()).hexdigest()
    assert sm['train_sha256']==sha and selection['production_refit'] is False
    assert cfg['model_name']==selection['model_name'] and cfg['revision']==selection['revision']
    _,tune_prob=load_tune(a.selection_dir/'tune_scores.npz',train)
    epochs=int(sm['best_epoch']);assert cfg['epochs']==epochs
    cuts=np.asarray(selection['thresholds']);assert len(cuts)==21 and selection['threshold_space']=='probability'
    fit,tune,valid=reference_splits(train,cfg.get('split_seed',2026));outer=np.r_[fit,tune]
    if a.check_only:
        print(json.dumps({'selection_verified':True,'epochs_fixed':epochs,'fit_rows':len(outer),'validation_rows':len(valid),
            'tune_score_rows_from_original_model':len(tune_prob),'refit_started':False,'blind_file_read':False}));raise SystemExit(0)
    y=train[LABELS].to_numpy(int);seed_all(cfg.get('training_seed',2026));torch.set_num_threads(cfg['cpu_threads'])
    tok=AutoTokenizer.from_pretrained(cfg['model_name'],revision=cfg['revision'],use_fast=cfg.get('fast_tokenizer',False))
    training=Reviews(train.text.iloc[outer],y[outer],tok,cfg['segment'],cfg['max_length'],cfg['stride'])
    validation=Reviews(train.text.iloc[valid],y[valid],tok,cfg['segment'],cfg['max_length'],cfg['stride'])
    loader=batches(training,tok,cfg['batch_size'],True,cfg.get('training_seed',2026),True)
    model=ReviewModel(cfg['model_name'],cfg['revision'],cfg['head'],-1,cfg.get('dropout',.2))
    model.autocast_cpu_bf16=cfg.get('cpu_bf16',False);opt=optimizer_for(model,cfg)
    acc=cfg['accumulation'];updates=math.ceil(len(loader)/acc)*epochs
    scheduler=get_cosine_schedule_with_warmup(opt,int(updates*cfg['warmup_ratio']),updates)
    pw=positive_weights(y[outer],cfg,'cpu');history=[];started=time.perf_counter()
    for ep in range(1,epochs+1):
        model.train();opt.zero_grad(set_to_none=True);total=0
        for step,(inputs,counts,labels) in enumerate(loader):
            with torch.autocast('cpu',dtype=torch.bfloat16,enabled=model.autocast_cpu_bf16):
                logits=model(inputs,counts);loss=supervised_loss(logits,labels,pw,cfg)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite refit loss')
            divisor=min(acc,len(loader)-(step//acc)*acc);(loss/divisor).backward();total+=float(loss.detach())
            if (step+1)%acc==0 or step+1==len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();scheduler.step();opt.zero_grad(set_to_none=True)
            if (step+1)%25==0:
                progress={'epoch':ep,'batch':step+1,'batches':len(loader),'mean_loss':total/(step+1),'elapsed_seconds':time.perf_counter()-started,'tune_evaluated_during_refit':False,'blind_file_read':False}
                (out/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
        history.append({'epoch':ep,'loss':total/len(loader),'tune_evaluated':False})
        pd.DataFrame(history).to_csv(out/'history.csv',index=False);print(history[-1],flush=True)
    seconds=time.perf_counter()-started;scores=infer(model,batches(validation,tok,cfg['batch_size']),torch.device('cpu'))
    pred=binary_predictions(scores,np.log(cuts/(1-cuts)));gt=train.iloc[valid][['id']+LABELS].reset_index(drop=True)
    sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
    result=report(gt,sb,n_boot=1000);result.update(best_epoch=epochs,best_tune_F1=None,selection_tune_F1=sm['best_tune_F1'],
        epochs_completed=epochs,epochs_requested=epochs,train_seconds=seconds,train_sha256=sha,fit_rows=len(outer),
        tune_rows=0,selection_tune_rows=len(tune),validation_rows=len(valid),refit_scope='outer80',
        selection_source=cfg['selection_source'],split_policy='reference_group_2026',tune_scores_origin='pre_refit_selection_encoder_out_of_fit',
        probability_shift_after_refit_possible=True,blind_file_read=False,validation_previously_seen=True,production_refit=False)
    (out/'metrics.json').write_text(json.dumps(result,indent=2));torch.save(model.state_dict(),out/'model.pt')
    np.savez_compressed(out/'validation_scores.npz',scores=1/(1+np.exp(-np.clip(scores,-30,30))),ids=gt.id.to_numpy(dtype=str))
    # Copy only OLD out-of-fit tune probabilities; never use predictions on the refitted model's training rows.
    shutil.copy2(a.selection_dir/'tune_scores.npz',out/'tune_scores.npz')
    (out/'selection_provenance.json').write_text(json.dumps({'selection_run_id':cfg['selection_run_id'],
        'selection_source':cfg['selection_source'],'selection_config_sha256':hashlib.sha256((a.selection_dir/'config.json').read_bytes()).hexdigest(),
        'tune_scores_sha256':hashlib.sha256((a.selection_dir/'tune_scores.npz').read_bytes()).hexdigest(),
        'tune_scores_are_from_pre_refit_model':True,'new_model_tune_scores_generated':False},indent=2))
    cfg.update(status='trained_validation_only',refit_scope='outer80',thresholds=cuts.tolist(),threshold_space='probability',columns=LABELS)
    a.config.write_text(json.dumps(cfg,indent=2))
    (out/'run_info.json').write_text(json.dumps({'train_sha256':sha,'fit_rows':len(outer),'refit_scope':'outer80','blind_file_read':False},indent=2))
    print(json.dumps({'experiment':cfg['experiment'],'ABSA_F1':result['ABSA_F1'],'fit_rows':len(outer),'blind_file_read':False}),flush=True)
