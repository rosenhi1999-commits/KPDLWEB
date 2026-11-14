"""Cross-fitted meta classifier from train-only supervised source predictions.

Base neural heads selected epochs on tune; therefore this is not fully nested
encoder cross-fitting. Outer validation is excluded from all meta fitting.
"""
from pathlib import Path
import train_only_guard
import hashlib,json,time
import numpy as np,pandas as pd,torch,joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from absa_core import LABELS,reference_splits,prepare_texts,features,SparseMultiLabel,tune_thresholds,binary_predictions
from aspect_scope import scoped_texts
from run_scoped import fit_predict as scoped_fit
from train_frozen_attention import prepare,Attention,load_batches,infer
from score_absa import report

ROOT=Path(__file__).resolve().parent

def estimator(c):
    return make_pipeline(StandardScaler(),LogisticRegression(C=c,solver='liblinear',max_iter=1500,random_state=2026))

def array(sources,label,mode):
    idx=list(range(21)) if mode=='all' else list(range((label//3)*3,(label//3+1)*3))
    return np.concatenate([s[:,idx] for s in sources],axis=1)

if __name__=='__main__':
    torch.set_num_threads(2);started=time.perf_counter();folder=ROOT/'Stacking_lan1';folder.mkdir(exist_ok=False)
    df=pd.read_csv(ROOT/'train.csv');y=df[LABELS].to_numpy(int);fit,tune,valid=reference_splits(df)
    texts=np.asarray(prepare_texts(df.text,'lexical'),dtype=object);scopes=scoped_texts(df.text)
    names=['ScopedTFIDF_lan1','ScopedTFIDF_lan4','ScopedTFIDF_lan5','NBSVM_lan3','PhoBERT_Attention_lan1','PhoBERT_Attention_lan2']
    cfg={'sources':names,'split_policy':'reference_group_2026','meta_cv_folds':5,
        'candidate_C':[.01,.1,1],'feature_modes':['own','all'],'blind_file_read':False,
        'neural_head_checkpoint_selected_on_tune':True,'fully_nested_encoder_cv':False}
    (folder/'config.json').write_text(json.dumps(cfg,indent=2))
    tune_sources=[];val_sources=[]
    for name in names[:3]:
        c=json.loads((ROOT/name/'config.json').read_text());scores,_=scoped_fit(texts,scopes,y,fit,tune,c)
        tune_sources.append(scores);val_sources.append(np.load(ROOT/name/'private_validation_scores.npy'))
        print('Train-only source ready',name,flush=True)
    name='NBSVM_lan3';c=json.loads((ROOT/name/'config.json').read_text())
    vec=features(c);X=vec.fit_transform(texts[fit]);T=vec.transform(texts[tune])
    scores=SparseMultiLabel(c).fit(X,y[fit]).decision_function(T)
    tune_sources.append(scores);val_sources.append(np.load(ROOT/name/'private_validation_scores.npy'))
    meta=json.loads((ROOT/'private_phobert_tokens.json').read_text())
    assert meta['train_sha256']==hashlib.sha256((ROOT/'train.csv').read_bytes()).hexdigest()
    raw=np.memmap(ROOT/'private_phobert_tokens.f16',dtype=np.float16,mode='r',shape=tuple(meta['shape']))
    spans=[[] for _ in range(len(df))]
    for i,start,length in meta['chunks']:spans[i].append((start,length))
    tokens,g,_=prepare(raw,spans,texts,fit)
    for name in names[-2:]:
        c=json.loads((ROOT/name/'config.json').read_text());model=Attention(g.shape[1],c['hybrid'],c['dropout'])
        model.load_state_dict(torch.load(ROOT/name/'inner_best.pt',map_location='cpu',weights_only=True))
        scores=infer(model,load_batches(tokens,spans,g,y.astype(np.float32),tune))
        tune_sources.append(scores);val_sources.append(np.load(ROOT/name/'private_validation_scores.npy'))
        print('Train-only source ready',name,flush=True)
    models=[];thresholds=[];label_cfg=[];validation_scores=np.zeros((len(valid),21));oof_all=np.zeros((len(tune),21))
    from sklearn.metrics import f1_score
    for j,label in enumerate(LABELS):
        target=y[tune,j];folds=list(StratifiedKFold(5,shuffle=True,random_state=2026).split(target,target));best=None
        for mode in cfg['feature_modes']:
            xt=array(tune_sources,j,mode)
            for c in cfg['candidate_C']:
                oof=np.zeros(len(tune))
                for a,b in folds:
                    m=estimator(c).fit(xt[a],target[a]);oof[b]=m.decision_function(xt[b])
                threshold=tune_thresholds(target[:,None],oof[:,None])[0]
                f1=f1_score(target,oof>=threshold,zero_division=0)
                if best is None or f1>best[0]:best=f1,c,mode,threshold,oof
        f1,c,mode,threshold,oof=best
        m=estimator(c).fit(array(tune_sources,j,mode),target)
        validation_scores[:,j]=m.decision_function(array(val_sources,j,mode));models.append(m);thresholds.append(threshold)
        oof_all[:,j]=oof;label_cfg.append({'label':label,'C':c,'mode':mode,'threshold':threshold,'meta_OOF_F1':f1})
        print(label,'meta_OOF_F1',round(f1,5),'C',c,'mode',mode,flush=True)
    pred=binary_predictions(validation_scores,np.asarray(thresholds))
    gt=df.iloc[valid][['id']+LABELS].reset_index(drop=True);sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
    r=report(gt,sb,n_boot=1000);r.update(experiment='Stacking_lan1',seconds=time.perf_counter()-started,
        split_policy='reference_group_2026',validation_previously_seen=True,blind_file_read=False,
        fully_nested_encoder_cv=False,train_sha256=meta['train_sha256'])
    cfg['labels']=label_cfg;(folder/'config.json').write_text(json.dumps(cfg,indent=2));(folder/'metrics.json').write_text(json.dumps(r,indent=2))
    joblib.dump({'models':models,'thresholds':thresholds,'sources':names,'labels':label_cfg},folder/'meta_models.joblib')
    np.save(folder/'private_validation_scores.npy',validation_scores);np.save(folder/'private_meta_oof.npy',oof_all)
    sb.to_csv(folder/'validation_predictions.csv',index=False)
    print('Stacking_lan1 outer_validation_ABSA_F1',r['ABSA_F1'],flush=True)
