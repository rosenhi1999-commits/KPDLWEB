"""Calibrate saved TUNE probabilities, refusing validation IDs and blind CSVs.

Cross-validation estimates calibration stability on tune, not a final held-out
score. Encoder checkpoint selection on the same tune is disclosed explicitly.
"""
from pathlib import Path
import train_only_guard
import argparse,json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import precision_recall_curve
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from absa_core import LABELS,reference_splits,binary_predictions,absa_f1

def best_cut(y,p,grid=None):
    y=np.asarray(y,dtype=bool);p=np.asarray(p,dtype=float)
    if not y.any():return float(np.nextafter(p.max(),np.inf))
    if y.all():return float(p.min())
    if grid is None:
        precision,recall,candidates=precision_recall_curve(y,p)
        fs=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
    else:
        candidates=np.asarray(grid);pred=p[:,None]>=candidates[None,:]
        tp=(pred&y[:,None]).sum(0);fp=(pred&~y[:,None]).sum(0);fn=(~pred&y[:,None]).sum(0)
        fs=2*tp/np.maximum(2*tp+fp+fn,1)
    tied=np.flatnonzero(np.isclose(fs,fs.max(),rtol=0,atol=1e-14))
    return float(candidates[tied[np.argmin(abs(candidates[tied]-.5))]])

def fit_thresholds(y,p,method,stage='encoder'):
    if method=='fixed':return np.full(21,.5)
    grid=np.linspace(.15,.75,13) if stage=='encoder' else np.linspace(.1,.9,17)
    if method=='reference_grid':return np.asarray([best_cut(y[:,j],p[:,j],grid) for j in range(21)])
    result=np.asarray([best_cut(y[:,j],p[:,j]) for j in range(21)])
    if method=='exact_pr':return result
    strength={'shrink25':25,'shrink75':75}[method]
    for offset in range(3):
        idx=list(range(offset,21,3));anchor=best_cut(y[:,idx].ravel(),p[:,idx].ravel())
        for j in idx:
            if y[:,j].min()==y[:,j].max():continue
            positives=y[:,j].sum();weight=positives/(positives+strength)
            result[j]=weight*result[j]+(1-weight)*anchor
    return result

def load_partition(path,train,partition):
    _,tune,valid=reference_splits(train)
    selected=tune if partition=='tune' else valid
    with np.load(path,allow_pickle=False) as obj:
        if 'ids' not in obj:raise ValueError('IDs required to verify tune membership')
        ids=obj['ids'].astype(str);scores=obj['scores'].copy()
    expected=train.id.iloc[selected].astype(str).to_numpy()
    if len(set(ids))!=len(ids) or set(ids)!=set(expected):
        raise ValueError('Scores are not exactly the reference '+partition.upper()+' IDs')
    if scores.shape!=(len(selected),21) or not np.isfinite(scores).all() or not ((scores>=0)&(scores<=1)).all():
        raise ValueError('Expected finite n_partition x 21 probabilities in [0,1]')
    order=pd.Index(ids).get_indexer(expected)
    return train[LABELS].iloc[selected].to_numpy(int),scores[order]

def load_tune(path,train):
    return load_partition(path,train,'tune')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scores',type=Path,required=True)
    ap.add_argument('--train',type=Path,default=Path(__file__).parent/'train.csv')
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stage',choices=['encoder','meta'],default='encoder')
    ap.add_argument('--validation-scores',type=Path,help='Optional train-derived outer validation scores, evaluated only after calibration selection')
    a=ap.parse_args()
    if a.train.name!='train.csv':raise ValueError('Only train.csv accepted')
    train=pd.read_csv(a.train);y,p=load_tune(a.scores,train)
    folds=list(MultilabelStratifiedKFold(5,shuffle=True,random_state=2026).split(p,y))
    methods=['fixed','reference_grid','exact_pr','shrink25','shrink75'];rows=[]
    for method in methods:
        out=np.zeros_like(y)
        for fit,held in folds:
            thresholds=fit_thresholds(y[fit],p[fit],method,a.stage)
            out[held]=binary_predictions(p[held],thresholds)
        rows.append({'method':method,'tune_OOF_ABSA_F1':absa_f1(y,out)})
    chosen=max(rows,key=lambda r:r['tune_OOF_ABSA_F1'])['method']
    thresholds=fit_thresholds(y,p,chosen,a.stage)
    validation=None
    if a.validation_scores:
        if a.validation_scores.parent.resolve()!=a.scores.parent.resolve():raise ValueError('Tune and validation scores must come from the same model directory')
        validation=load_partition(a.validation_scores,train,'validation')
    a.out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_csv(a.out/'calibration_comparison.csv',index=False)
    result={'selected_method':chosen,'stage':a.stage,'thresholds':dict(zip(LABELS,thresholds.tolist())),
        'selection':'5-fold held-out calibration on train-derived tune only','rows':len(y),
        'not_an_outer_validation_or_public_score':True,'encoder_may_have_selected_checkpoint_on_tune':True,
        'blind_file_read':False,'candidate_results':rows,
        'train_sha256':hashlib.sha256(a.train.read_bytes()).hexdigest(),
        'tune_scores_sha256':hashlib.sha256(a.scores.read_bytes()).hexdigest()}
    (a.out/'calibration_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if validation is not None:
        from score_absa import report
        yy,pp=validation;_,_,valid=reference_splits(train)
        gt=train.iloc[valid][['id']+LABELS].reset_index(drop=True)
        pred=binary_predictions(pp,thresholds);sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
        metrics=report(gt,sb,n_boot=1000)
        baseline=fit_thresholds(y,p,'reference_grid',a.stage)
        base_f1=absa_f1(yy,binary_predictions(pp,baseline))
        metrics.update(baseline_reference_grid_ABSA_F1=base_f1,delta_ABSA_F1=metrics['ABSA_F1']-base_f1,
            selected_on='tune OOF only',selected_method=chosen,split_policy='reference_group_2026',
            outer_validation_used_for_threshold_selection=False,blind_file_read=False)
        (a.out/'metrics.json').write_text(json.dumps(metrics,indent=2))
        sb.to_csv(a.out/'validation_predictions.csv',index=False)
        print(json.dumps({'outer_validation_ABSA_F1':metrics['ABSA_F1'],'delta_to_reference_grid':metrics['delta_ABSA_F1']}))
