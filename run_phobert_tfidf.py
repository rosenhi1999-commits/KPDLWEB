"""Train supervised classifiers on frozen PhoBERT features, optionally with TF-IDF."""
from pathlib import Path
import train_only_guard
import argparse,hashlib,json,time
import joblib,numpy as np,pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,normalize as l2normalize
from absa_core import LABELS,reference_splits,prepare_texts,features,SparseMultiLabel,tune_thresholds,binary_predictions,absa_f1
from score_absa import report

ROOT=Path(__file__).resolve().parent

def fit_predict(texts,emb,y,fit,predict,cfg):
    scaler=StandardScaler();D=scaler.fit_transform(emb[fit]);V=scaler.transform(emb[predict])
    pca=PCA(n_components=256,svd_solver='randomized',random_state=2026)
    D=l2normalize(pca.fit_transform(D));V=l2normalize(pca.transform(V))
    weight=cfg['embedding_weight'];vec=None
    X=sparse.csr_matrix(D*weight);P=sparse.csr_matrix(V*weight)
    if cfg['with_tfidf']:
        vec=features({});G=vec.fit_transform(texts[fit]);T=vec.transform(texts[predict])
        X=sparse.hstack([G,X],format='csr');P=sparse.hstack([T,P],format='csr')
    model=SparseMultiLabel(cfg).fit(X,y[fit]);scores=model.decision_function(P)
    return scores,dict(scaler=scaler,pca=pca,vectorizer=vec,model=model,config=cfg)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--feature-set',choices=['pyvi','rdr'],default='pyvi');args=ap.parse_args()
    stem='phobert' if args.feature_set=='pyvi' else 'phobert_rdr'
    df=pd.read_csv(ROOT/'train.csv');assert list(df.columns)==['id','text']+LABELS
    meta=json.loads((ROOT/(stem+'_feature_extraction.json')).read_text())
    assert hashlib.sha256((ROOT/'train.csv').read_bytes()).hexdigest()==meta['train_sha256']
    emb=np.load(ROOT/('private_'+stem+'_features.npy'));assert emb.shape[0]==len(df)
    y=df[LABELS].to_numpy(int);fit,tune,valid=reference_splits(df);outer=np.r_[fit,tune]
    texts=np.asarray(prepare_texts(df.text,'lexical'),dtype=object)
    configs=[dict(kind='logistic',C=4,embedding_weight=1,with_tfidf=False),
        dict(kind='logistic',C=4,embedding_weight=.5,with_tfidf=True),
        dict(kind='logistic',C=4,embedding_weight=2,with_tfidf=True),
        dict(kind='logistic',C=4,embedding_weight=4,with_tfidf=True),
        dict(kind='svm',C=1,embedding_weight=2,with_tfidf=True)]
    fixed=[]
    if args.feature_set=='rdr':configs=[configs[0],configs[1],dict(kind='svm',C=1,embedding_weight=.5,with_tfidf=True)]
    for i,cfg in enumerate(configs,1):
        name=('PhoBERT_TFIDF_lan' if args.feature_set=='pyvi' else 'PhoBERT_RDR_lan')+str(i);folder=ROOT/name;folder.mkdir(exist_ok=False)
        cfg.update(encoder_frozen=True,split_policy='reference_group_2026',normalization='lexical',
                   embedding_normalization=args.feature_set+', preserve case',blind_file_read=False)
        (folder/'config.json').write_text(json.dumps(cfg,indent=2))
        start=time.perf_counter();scores,_=fit_predict(texts,emb,y,fit,tune,cfg)
        thresholds=tune_thresholds(y[tune],scores);f1=absa_f1(y[tune],binary_predictions(scores,thresholds))
        fixed.append((name,cfg,thresholds,f1,time.perf_counter()-start))
        print(name,'inner_tune_ABSA_F1',f1,flush=True)
    rows=[]
    for name,cfg,cuts,tune_f1,inner_seconds in fixed:
        start=time.perf_counter();scores,bundle=fit_predict(texts,emb,y,outer,valid,cfg)
        seconds=time.perf_counter()-start;pred=binary_predictions(scores,cuts)
        gt=df.iloc[valid][['id']+LABELS].reset_index(drop=True);sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
        r=report(gt,sb,n_boot=1000);r.update(experiment=name,inner_tune_ABSA_F1=tune_f1,
            inner_seconds=inner_seconds,outer_fit_and_predict_seconds=seconds,feature_extraction=meta,
            split_policy='reference_group_2026',validation_previously_seen=True,blind_file_read=False)
        folder=ROOT/name;(folder/'metrics.json').write_text(json.dumps(r,indent=2))
        bundle.update(thresholds=cuts,labels=LABELS);joblib.dump(bundle,folder/'validation_model.joblib')
        np.save(folder/'private_validation_scores.npy',scores);sb.to_csv(folder/'validation_predictions.csv',index=False)
        rows.append({'experiment':name,'inner_tune_ABSA_F1':tune_f1,'ABSA_F1':r['ABSA_F1'],
            'CI95_low':r['ABSA_F1_CI95'][0],'CI95_high':r['ABSA_F1_CI95'][1]})
        print(name,'outer_validation_ABSA_F1',r['ABSA_F1'],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/(stem+'_tfidf_comparison.csv'),index=False)
