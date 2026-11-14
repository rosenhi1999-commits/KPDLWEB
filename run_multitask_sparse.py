"""Shared sentiment coefficients plus aspect-specific residuals, train only."""
from pathlib import Path
import train_only_guard
import json,time
import numpy as np,pandas as pd,joblib
from scipy import sparse
from absa_core import ASPECTS,LABELS,features,prepare_texts,reference_splits,SparseMultiLabel,tune_thresholds,binary_predictions,absa_f1
from aspect_scope import scoped_texts
from score_absa import report

ROOT=Path(__file__).resolve().parent

def matrices(texts,scopes,fit,predict,cfg):
    global_vec=features({'char':False,'word_max':2})
    G=global_vec.fit_transform(texts[fit]);V=global_vec.transform(texts[predict])
    scope_vec=features({'word_max':2,'char_analyzer':'char_wb'})
    a=np.concatenate([np.asarray(scopes[k],dtype=object)[fit] for k in ASPECTS])
    b=np.concatenate([np.asarray(scopes[k],dtype=object)[predict] for k in ASPECTS])
    S=scope_vec.fit_transform(a);T=scope_vec.transform(b)
    n=len(fit);m=len(predict)
    specific=sparse.block_diag([S[i*n:(i+1)*n] for i in range(7)],format='csr')
    specific_t=sparse.block_diag([T[i*m:(i+1)*m] for i in range(7)],format='csr')
    # The same shared lexical coefficient can transfer between aspects.
    X=sparse.hstack([S,specific*cfg['residual_weight'],sparse.block_diag([G]*7)*cfg['global_weight'],
        sparse.kron(sparse.eye(7),np.ones((n,1)))],format='csr')
    P=sparse.hstack([T,specific_t*cfg['residual_weight'],sparse.block_diag([V]*7)*cfg['global_weight'],
        sparse.kron(sparse.eye(7),np.ones((m,1)))],format='csr')
    return X,P,{'scope_vectorizer':scope_vec,'global_vectorizer':global_vec,'config':cfg}

def fit_predict(texts,scopes,y,fit,predict,cfg):
    X,P,bundle=matrices(texts,scopes,fit,predict,cfg)
    target=y[fit].reshape(-1,7,3).transpose(1,0,2).reshape(-1,3)
    model=SparseMultiLabel(cfg).fit(X,target)
    scores=model.decision_function(P).reshape(7,len(predict),3).transpose(1,0,2).reshape(len(predict),21)
    bundle['model']=model;return scores,bundle

if __name__=='__main__':
    df=pd.read_csv(ROOT/'train.csv');assert list(df.columns)==['id','text']+LABELS
    y=df[LABELS].to_numpy(int);fit,tune,valid=reference_splits(df);outer=np.r_[fit,tune]
    texts=np.asarray(prepare_texts(df.text,'lexical'),dtype=object);scopes=scoped_texts(df.text)
    configs=[dict(kind='svm',C=1,residual_weight=.5,global_weight=.5),
        dict(kind='logistic',C=4,residual_weight=.5,global_weight=.5),
        dict(kind='svm',C=1,residual_weight=1,global_weight=.5),
        dict(kind='logistic',C=4,residual_weight=0,global_weight=.5)]
    selected=[]
    for i,cfg in enumerate(configs,1):
        name='MultiTaskTFIDF_lan'+str(i);folder=ROOT/name;folder.mkdir(exist_ok=False)
        cfg.update(split_policy='reference_group_2026',blind_file_read=False)
        (folder/'config.json').write_text(json.dumps(cfg,indent=2))
        start=time.perf_counter();scores,_=fit_predict(texts,scopes,y,fit,tune,cfg)
        thresholds=tune_thresholds(y[tune],scores);f1=absa_f1(y[tune],binary_predictions(scores,thresholds))
        selected.append((name,cfg,thresholds,f1,time.perf_counter()-start))
        print(name,'inner_tune_ABSA_F1',f1,flush=True)
    rows=[]
    for name,cfg,thresholds,tune_f1,inner_seconds in selected:
        start=time.perf_counter();scores,bundle=fit_predict(texts,scopes,y,outer,valid,cfg)
        seconds=time.perf_counter()-start;pred=binary_predictions(scores,thresholds)
        gt=df.iloc[valid][['id']+LABELS].reset_index(drop=True);sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
        r=report(gt,sb,n_boot=1000);r.update(experiment=name,inner_tune_ABSA_F1=tune_f1,
            inner_seconds=inner_seconds,outer_fit_and_predict_seconds=seconds,split_policy='reference_group_2026',
            validation_previously_seen=True,blind_file_read=False)
        folder=ROOT/name;(folder/'metrics.json').write_text(json.dumps(r,indent=2))
        bundle.update(labels=LABELS,thresholds=thresholds);joblib.dump(bundle,folder/'validation_model.joblib')
        np.save(folder/'private_validation_scores.npy',scores);sb.to_csv(folder/'validation_predictions.csv',index=False)
        rows.append({'experiment':name,'inner_tune_ABSA_F1':tune_f1,'ABSA_F1':r['ABSA_F1'],
            'CI95_low':r['ABSA_F1_CI95'][0],'CI95_high':r['ABSA_F1_CI95'][1]})
        print(name,'outer_validation_ABSA_F1',r['ABSA_F1'],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'multitask_sparse_comparison.csv',index=False)
