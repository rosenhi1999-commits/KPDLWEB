"""Train-only scoped TF-IDF; fixed reference split; thresholds learned on tune."""
import json,time,hashlib
from pathlib import Path
import train_only_guard
import numpy as np,pandas as pd,joblib
from scipy import sparse
from absa_core import ASPECTS,LABELS,prepare_texts,reference_splits,features,SparseMultiLabel,tune_thresholds,binary_predictions,absa_f1
from aspect_scope import scoped_texts
from score_absa import report

ROOT=Path(__file__).resolve().parent

def fit_predict(texts,scopes,y,fit,predict,cfg):
    vec=features({'word_max':2});G=vec.fit_transform(texts[fit]);P=vec.transform(texts[predict])
    models=[];local_vecs=[];scores=[]
    for a,aspect in enumerate(ASPECTS):
        local=features({'word_max':3,'char_analyzer':'char_wb'})
        L=local.fit_transform(np.asarray(scopes[aspect],dtype=object)[fit])
        V=local.transform(np.asarray(scopes[aspect],dtype=object)[predict])
        X=sparse.hstack([G,L*cfg['scope_weight']],format='csr')
        T=sparse.hstack([P,V*cfg['scope_weight']],format='csr')
        model=SparseMultiLabel(cfg).fit(X,y[fit,a*3:a*3+3]);scores.append(model.decision_function(T))
        models.append(model);local_vecs.append(local)
    return np.hstack(scores),{'global_vectorizer':vec,'local_vectorizers':local_vecs,'models':models,'config':cfg}

if __name__=='__main__':
    df=pd.read_csv(ROOT/'train.csv');assert list(df.columns)==['id','text']+LABELS
    y=df[LABELS].to_numpy(int);fit,tune,valid=reference_splits(df);outer=np.r_[fit,tune]
    texts=np.asarray(prepare_texts(df.text,'lexical'),dtype=object);scopes=scoped_texts(df.text)
    configs=[('ScopedTFIDF_lan1',dict(kind='svm',C=1,scope_weight=.5)),
             ('ScopedTFIDF_lan2',dict(kind='svm',C=1,scope_weight=1)),
             ('ScopedTFIDF_lan3',dict(kind='svm',C=1,scope_weight=2)),
             ('ScopedTFIDF_lan4',dict(kind='logistic',C=4,scope_weight=1)),
             ('ScopedTFIDF_lan5',dict(kind='nbsvm',C=4,scope_weight=1))]
    fixed=[]
    for name,cfg in configs:
        folder=ROOT/name;folder.mkdir(exist_ok=False)
        cfg.update(split_policy='reference_group_2026',label_method='supervised',blind_file_read=False)
        (folder/'config.json').write_text(json.dumps(cfg,indent=2))
        start=time.perf_counter();scores,_=fit_predict(texts,scopes,y,fit,tune,cfg)
        thresholds=tune_thresholds(y[tune],scores);score=absa_f1(y[tune],binary_predictions(scores,thresholds))
        fixed.append((name,cfg,thresholds,score,time.perf_counter()-start))
        print(name,'inner_tune_ABSA_F1',score,flush=True)
    rows=[]
    for name,cfg,thresholds,tune_score,inner_seconds in fixed:
        start=time.perf_counter();scores,bundle=fit_predict(texts,scopes,y,outer,valid,cfg)
        seconds=time.perf_counter()-start;pred=binary_predictions(scores,thresholds)
        gt=df.iloc[valid][['id']+LABELS].reset_index(drop=True);sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
        r=report(gt,sb,n_boot=1000);r.update(experiment=name,inner_tune_ABSA_F1=tune_score,
            inner_seconds=inner_seconds,outer_fit_and_predict_seconds=seconds,validation_rows=len(valid),
            split_policy='reference_group_2026',train_sha256=hashlib.sha256((ROOT/'train.csv').read_bytes()).hexdigest(),
            validation_previously_seen=True,blind_file_read=False)
        folder=ROOT/name;(folder/'metrics.json').write_text(json.dumps(r,indent=2))
        bundle.update(thresholds=thresholds,labels=LABELS);joblib.dump(bundle,folder/'validation_model.joblib')
        np.save(folder/'private_validation_scores.npy',scores);sb.to_csv(folder/'validation_predictions.csv',index=False)
        rows.append({'experiment':name,'inner_tune_ABSA_F1':tune_score,'ABSA_F1':r['ABSA_F1'],
            'CI95_low':r['ABSA_F1_CI95'][0],'CI95_high':r['ABSA_F1_CI95'][1]})
        print(name,'outer_validation_ABSA_F1',r['ABSA_F1'],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'scoped_comparison.csv',index=False)
