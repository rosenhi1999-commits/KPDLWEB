"""Search on an inner tune split; evaluate fixed candidates on outer validation."""
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from absa_core import LABELS, prepare_texts, splits, reference_splits, features, SparseMultiLabel, tune_thresholds, binary_predictions, absa_f1
from score_absa import report

ROOT=Path(__file__).resolve().parent
CONFIGS=[
 ('TFIDF_lan3',dict(kind='logistic',C=1)),
 ('TFIDF_lan4',dict(kind='logistic',C=4)),
 ('TFIDF_lan5',dict(kind='logistic',C=16)),
 ('TFIDF_lan6',dict(kind='svm',C=.25)),
 ('TFIDF_lan7',dict(kind='svm',C=1)),
 ('TFIDF_lan8',dict(kind='svm',C=4)),
 ('TFIDF_lan9',dict(kind='svm',C=1,char=False,word_max=3)),
 ('TFIDF_lan10',dict(kind='svm',C=1,word=False,char_analyzer='char_wb')),
 ('TFIDF_lan11',dict(kind='svm',C=1,normalization='lexical')),
 ('TFIDF_lan12',dict(kind='svm',C=1,normalization='segmented')),
 ('NBSVM_lan1',dict(kind='nbsvm',C=1)),
 ('NBSVM_lan2',dict(kind='nbsvm',C=4,normalization='lexical')),
 ('ComplementNB_lan1',dict(kind='complementnb',alpha=.5)),
 ('SGD_lan1',dict(kind='sgd',alpha=1e-4)),
 ('Ridge_lan1',dict(kind='ridge',alpha=1)),
]

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')

def evaluate(df,y,idx,scores,thresholds,folder):
    pred=binary_predictions(scores,thresholds)
    gt=df.iloc[idx][['id']+LABELS].reset_index(drop=True)
    sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
    r=report(gt,sb,n_boot=500)
    sb.to_csv(folder/'validation_predictions.csv',index=False,encoding='utf-8')
    errors=[]
    for i in np.flatnonzero((pred!=y[idx]).any(axis=1))[:20]:
        errors.append({'id':str(gt.id.iloc[i]),'text':str(df.text.iloc[idx[i]]),
            'false_positive':[LABELS[j] for j in np.flatnonzero((pred[i]==1)&(y[idx[i]]==0))],
            'false_negative':[LABELS[j] for j in np.flatnonzero((pred[i]==0)&(y[idx[i]]==1))]})
    dump(folder/'private_validation_errors.json',errors)
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=Path,default=ROOT/'train.csv')
    ap.add_argument('--suite',choices=['legacy','reference'],default='legacy')
    ap.add_argument('--results-root',type=Path,default=ROOT/'private_runs')
    a=ap.parse_args()
    results_root=a.results_root;results_root.mkdir(parents=True,exist_ok=True)
    if a.train.name!='train.csv':raise ValueError('This runner accepts only train.csv')
    df=pd.read_csv(a.train);assert list(df.columns)==['id','text']+LABELS
    y=df[LABELS].to_numpy(int);assert np.isin(y,[0,1]).all()
    fit,tune,valid=reference_splits(df) if a.suite=='reference' else splits(len(df),y)
    outer=np.r_[fit,tune]
    suffix='_reference' if a.suite=='reference' else ''
    configs=CONFIGS if a.suite=='legacy' else [
        ('TFIDF_lan13',dict(kind='svm',C=1)),
        ('TFIDF_lan14',dict(kind='svm',C=1,normalization='lexical')),
        ('TFIDF_lan15',dict(kind='svm',C=1,normalization='segmented')),
        ('NBSVM_lan3',dict(kind='nbsvm',C=4,normalization='lexical')),
        ('TFIDF_lan16',dict(kind='logistic',C=16))]
    assert not set(outer)&set(valid)
    split={'fit':fit.tolist(),'tune':tune.tolist(),'validation':valid.tolist()}
    dump(results_root/('private_split_indices'+suffix+'.json'),split)
    rows=[];completed=[];text_cache={}
    # No validation scores are consulted until ALL inner experiments are fixed.
    for name,cfg in configs:
        folder=results_root/name;folder.mkdir(exist_ok=False)
        cfg={**cfg,'normalization':cfg.get('normalization','base'),
             'split_policy':'reference_group_2026' if a.suite=='reference' else 'legacy_random42'}
        dump(folder/'config.json',cfg)
        mode=cfg['normalization'];start=time.perf_counter()
        if mode not in text_cache:text_cache[mode]=prepare_texts(df.text,mode)
        texts=np.asarray(text_cache[mode],dtype=object)
        vec=features(cfg);X=vec.fit_transform(texts[fit]);T=vec.transform(texts[tune])
        model=SparseMultiLabel(cfg).fit(X,y[fit]);scores=model.decision_function(T)
        thresholds=tune_thresholds(y[tune],scores)
        tune_f1=absa_f1(y[tune],binary_predictions(scores,thresholds))
        inner_seconds=time.perf_counter()-start
        dump(folder/'inner_tuning.json',{'ABSA_F1':tune_f1,'seconds':inner_seconds,'thresholds':thresholds.tolist()})
        print(name,'inner_tune_ABSA_F1',round(tune_f1,6),flush=True)
        completed.append((name,cfg,thresholds,tune_f1,inner_seconds))
    selected=sorted(completed,key=lambda r:r[3],reverse=True)[:3]
    dump(results_root/('ensemble_selection'+suffix+'.json'),{'selected_by':'inner_tune_ABSA_F1_only','members':[r[0] for r in selected]})
    # Refit on all outer80; keep threshold selection independent of outer20.
    all_scores={};all_tune_scores={}
    for name,cfg,thresholds,tune_f1,inner_seconds in completed:
        folder=results_root/name;texts=np.asarray(text_cache[cfg['normalization']],dtype=object)
        start=time.perf_counter();vec=features(cfg)
        X=vec.fit_transform(texts[outer]);V=vec.transform(texts[valid])
        model=SparseMultiLabel(cfg).fit(X,y[outer]);fit_seconds=time.perf_counter()-start
        start=time.perf_counter();scores=model.decision_function(V);inference=time.perf_counter()-start
        r=evaluate(df,y,valid,scores,thresholds,folder)
        r.update({'experiment':name,'inner_tune_ABSA_F1':tune_f1,'split_seed':2026 if a.suite=='reference' else 42,'model_seed':42,'split_policy':cfg['split_policy'],'fit_rows':len(fit),
            'tune_rows':len(tune),'refit_rows':len(outer),'validation_rows':len(valid),
            'inner_search_seconds':inner_seconds,'refit_seconds':fit_seconds,'validation_inference_seconds':inference,
            'train_sha256':hashlib.sha256(a.train.read_bytes()).hexdigest(),
            'validation_previously_seen':True,'blind_file_read':False,
            'python':sys.version,'platform':platform.platform(),'processor':platform.processor()})
        dump(folder/'metrics.json',r)
        joblib.dump({'config':cfg,'features':vec,'model':model,'thresholds':thresholds,'labels':LABELS},folder/'validation_model.joblib')
        np.save(folder/'private_validation_scores.npy',scores)
        all_scores[name]=scores-thresholds
        rows.append({'experiment':name,'inner_tune_ABSA_F1':tune_f1,'ABSA_F1':r['ABSA_F1'],
            'CI95_low':r['ABSA_F1_CI95'][0],'CI95_high':r['ABSA_F1_CI95'][1],
            'precision':r['precision_TB'],'recall':r['recall_TB'],'accuracy':r['accuracy_TB'],
            'kappa':r['kappa_TB'],'MCC':r['mcc_TB'],'balanced_accuracy':r['bal_acc_TB'],
            'refit_seconds':fit_seconds,'validation_inference_seconds':inference})
        print(name,'outer_validation_ABSA_F1',round(r['ABSA_F1'],6),flush=True)
    folder=results_root/('Ensemble_lan2' if a.suite=='reference' else 'Ensemble_lan1');folder.mkdir(exist_ok=False)
    members=[r[0] for r in selected]
    # Equal votes; no weights learned from outer validation.
    votes=np.mean([all_scores[n]>=0 for n in members],axis=0)
    r=evaluate(df,y,valid,votes,np.full(21,.5),folder)
    r.update({'members':members,'selection':'inner tune only','validation_previously_seen':True,'blind_file_read':False})
    dump(folder/'metrics.json',r);dump(folder/'config.json',{'members':members,'vote_threshold':.5})
    rows.append({'experiment':folder.name,'ABSA_F1':r['ABSA_F1'],
        'CI95_low':r['ABSA_F1_CI95'][0],'CI95_high':r['ABSA_F1_CI95'][1]})
    comparison=pd.DataFrame(rows).sort_values('ABSA_F1',ascending=False)
    comparison.to_csv(results_root/('experiment_comparison'+suffix+'.csv'),index=False)
    print(comparison[['experiment','ABSA_F1','CI95_low','CI95_high']].to_string(index=False),flush=True)

if __name__=='__main__':main()
