"""Low-complexity blending selected on tune OOF, with one outer evaluation."""
from pathlib import Path
import train_only_guard
import hashlib,json
import numpy as np,pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from absa_core import LABELS,reference_splits,binary_predictions,absa_f1
from calibrate_reference import load_partition,fit_thresholds
from score_absa import report,paired,to_arrays

ROOT=Path(__file__).resolve().parent

def blend(trials,train,out_root=None,selection_rule='one_se'):
    out_root=Path(out_root or ROOT)
    rank=[];cache={};_,tu,va=reference_splits(train)
    sha=hashlib.sha256((ROOT/'train.csv').read_bytes()).hexdigest()
    for name,path,m in trials:
        if m['train_sha256']!=sha:raise ValueError('Source training data mismatch')
        _,t=load_partition(path/'tune_scores.npz',train,'tune');_,v=load_partition(path/'validation_scores.npz',train,'validation')
        if m.get('refit_scope')=='outer80':score=float(m['selection_tune_F1'])
        else:
            h=pd.read_csv(path/'history.csv');score=float(h.loc[h.epoch==m['best_epoch'],'tune_F1_tuned_threshold'].iloc[0])
        rank.append((score,m.get('fit_rows',2812),name));cache[name]=(t,v)
    rank.sort(reverse=True);names=[];seen=set()
    # An old selection encoder and its refit can share identical tune predictions.
    for _,_,name in rank:
        key=hashlib.sha256(np.ascontiguousarray(cache[name][0]).tobytes()).hexdigest()
        if key not in seen:names.append(name);seen.add(key)
    names=names[:5];assert names
    candidates=[('anchor',{names[0]:1.})]
    for k in [2,3,5]:
        if len(names)>=k:candidates.append((f'mean_top{k}',{n:1/k for n in names[:k]}))
    if len(names)>=2:
        candidates += [('anchor75',{names[0]:.75,names[1]:.25}),('anchor90',{names[0]:.9,names[1]:.1})]
    y=train[LABELS].iloc[tu].to_numpy(int)
    folds=list(MultilabelStratifiedKFold(5,shuffle=True,random_state=2026).split(np.zeros(len(tu)),y))
    cv=[];predictions={}
    for label,weights in candidates:
        p=sum(cache[n][0]*w for n,w in weights.items());pred=np.zeros_like(y)
        for fit,held in folds:
            thresholds=fit_thresholds(y[fit],p[fit],'reference_grid','encoder')
            pred[held]=binary_predictions(p[held],thresholds)
        predictions[label]=pred;cv.append({'candidate':label,'tune_OOF_ABSA_F1':absa_f1(y,pred),'sources':len(weights)})
    best=max(cv,key=lambda x:(x['tune_OOF_ABSA_F1'],-x['sources']))
    rng=np.random.default_rng(2026);boot=[]
    for _ in range(300):
        idx=rng.integers(0,len(y),len(y));boot.append(absa_f1(y[idx],predictions[best['candidate']][idx]))
    standard_error=float(np.std(boot,ddof=1))
    if selection_rule=='one_se':
        eligible=[row for row in cv if row['tune_OOF_ABSA_F1']>=best['tune_OOF_ABSA_F1']-standard_error]
        selected=min(eligible,key=lambda row:(row['sources'],-row['tune_OOF_ABSA_F1']))['candidate']
    elif selection_rule=='max_oof':selected=best['candidate']
    else:raise ValueError('Invalid blend selection rule')
    weights=dict(candidates)[selected];p=sum(cache[n][0]*w for n,w in weights.items())
    thresholds=fit_thresholds(y,p,'reference_grid','encoder')
    # Only after candidate and thresholds are fixed, evaluate outer validation once.
    v=sum(cache[n][1]*w for n,w in weights.items());pred=binary_predictions(v,thresholds)
    gt=train.iloc[va][['id']+LABELS].reset_index(drop=True);sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
    result=report(gt,sb,n_boot=1000)
    number=max([0]+[int(p.name.removeprefix('Blending_lan')) for p in out_root.glob('Blending_lan*') if p.name.removeprefix('Blending_lan').isdigit()])+1
    folder=out_root/f'Blending_lan{number}';folder.mkdir(exist_ok=False)
    result.update(experiment=folder.name,selected_candidate=selected,split_policy='reference_group_2026',
        train_sha256=sha,blind_file_read=False,validation_previously_seen=True,fully_nested_encoder_cv=False)
    cfg={'weights':weights,'thresholds':dict(zip(LABELS,thresholds.tolist())),
        'selection':'global weights selected by 5-fold tune OOF only, outer validation never used',
        'selection_rule':selection_rule,'bootstrap_tune_OOF_standard_error':standard_error,
        'ranked_unique_sources':names,'candidate_scores':cv,'blind_file_read':False}
    (folder/'config.json').write_text(json.dumps(cfg,indent=2));(folder/'metrics.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(cv).to_csv(folder/'tune_oof_comparison.csv',index=False)
    private=out_root/'private_blending';private.mkdir(exist_ok=True)
    np.savez_compressed(private/(folder.name+'_validation_scores.npz'),scores=v,ids=gt.id.to_numpy(dtype=str))
    return result,[str((folder/n).relative_to(out_root)) for n in ['config.json','metrics.json','tune_oof_comparison.csv']]

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--artifacts',type=Path,required=True)
    ap.add_argument('--selection-rule',choices=['one_se','max_oof'],default='one_se');a=ap.parse_args()
    train=pd.read_csv(ROOT/'train.csv');trials=[]
    for path in sorted(a.artifacts.glob('absa-*-metrics')):
        if (path/'metrics.json').exists():trials.append((path.name.removeprefix('absa-').removesuffix('-metrics'),path,json.loads((path/'metrics.json').read_text())))
    result,_=blend(trials,train,selection_rule=a.selection_rule);print(json.dumps({'experiment':result['experiment'],'selected_candidate':result['selected_candidate'],'ABSA_F1':result['ABSA_F1']}))
