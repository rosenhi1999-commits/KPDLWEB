"""Train-only augmentation ablation, same untouched reference outer validation."""
from pathlib import Path
import train_only_guard
import json,time
import numpy as np,pandas as pd,joblib
from absa_core import LABELS,reference_splits,prepare_texts,tune_thresholds,binary_predictions,absa_f1
from aspect_scope import scoped_texts
from run_scoped import fit_predict
from score_absa import report

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    train=pd.read_csv(ROOT/'train.csv')
    added_path=ROOT/'du_lieu_them.csv'
    if not added_path.exists():added_path=ROOT/'private_du_lieu_them.csv'
    aug=pd.read_csv(added_path)
    assert list(train.columns)==list(aug.columns)==['id','text']+LABELS
    assert set(train.id).isdisjoint(set(aug.id))
    # Remove exact training duplicates; no use of validation labels for augmentation.
    normalize=lambda s:' '.join(str(s).lower().split())
    aug=aug[~aug.text.map(normalize).isin(set(train.text.map(normalize)))].copy()
    fit,tune,valid=reference_splits(train);df=pd.concat([train,aug],ignore_index=True)
    added=np.arange(len(train),len(df));fit_aug=np.r_[fit,added];outer_aug=np.r_[fit,tune,added]
    y=df[LABELS].to_numpy(int);texts=np.asarray(prepare_texts(df.text,'lexical'),dtype=object);scopes=scoped_texts(df.text)
    configs=[('AugmentScoped_lan1',dict(kind='nbsvm',C=4,scope_weight=1),'ScopedTFIDF_lan5'),
        ('AugmentScoped_lan2',dict(kind='logistic',C=4,scope_weight=1),'ScopedTFIDF_lan4')]
    selected=[]
    for name,cfg,baseline in configs:
        folder=ROOT/name;folder.mkdir(exist_ok=False)
        cfg.update(split_policy='reference_group_2026',augmentation_rows=len(aug),augmentation_file='du_lieu_them.csv',
            provenance_file='augmentation_provenance.json',baseline=baseline,blind_file_read=False)
        (folder/'config.json').write_text(json.dumps(cfg,indent=2))
        start=time.perf_counter();scores,_=fit_predict(texts,scopes,y,fit_aug,tune,cfg)
        thresholds=tune_thresholds(y[tune],scores);f1=absa_f1(y[tune],binary_predictions(scores,thresholds))
        selected.append((name,cfg,baseline,thresholds,f1,time.perf_counter()-start))
        print(name,'inner_tune_ABSA_F1',f1,flush=True)
    rows=[]
    for name,cfg,baseline,thresholds,tune_f1,inner_seconds in selected:
        start=time.perf_counter();scores,bundle=fit_predict(texts,scopes,y,outer_aug,valid,cfg)
        seconds=time.perf_counter()-start;pred=binary_predictions(scores,thresholds)
        gt=df.iloc[valid][['id']+LABELS].reset_index(drop=True);sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
        r=report(gt,sb,n_boot=1000);base=json.loads((ROOT/baseline/'metrics.json').read_text())
        r.update(experiment=name,inner_tune_ABSA_F1=tune_f1,inner_seconds=inner_seconds,
            outer_fit_and_predict_seconds=seconds,augmentation_rows=len(aug),baseline=baseline,
            baseline_ABSA_F1=base['ABSA_F1'],delta_ABSA_F1=r['ABSA_F1']-base['ABSA_F1'],
            split_policy='reference_group_2026',validation_previously_seen=True,blind_file_read=False)
        folder=ROOT/name;(folder/'metrics.json').write_text(json.dumps(r,indent=2))
        bundle.update(labels=LABELS,thresholds=thresholds);joblib.dump(bundle,folder/'validation_model.joblib')
        np.save(folder/'private_validation_scores.npy',scores);sb.to_csv(folder/'validation_predictions.csv',index=False)
        rows.append({'experiment':name,'ABSA_F1':r['ABSA_F1'],'baseline':baseline,
            'baseline_ABSA_F1':base['ABSA_F1'],'delta_ABSA_F1':r['delta_ABSA_F1'],
            'CI95_low':r['ABSA_F1_CI95'][0],'CI95_high':r['ABSA_F1_CI95'][1]})
        print(name,'outer_validation_ABSA_F1',r['ABSA_F1'],'delta',r['delta_ABSA_F1'],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'augmentation_comparison.csv',index=False)
