"""Collect finished real trials and fit train-only stacking from saved scores."""
from pathlib import Path
import train_only_guard
import argparse,hashlib,json,os,re,shutil,subprocess
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from absa_core import LABELS,reference_splits,binary_predictions
from calibrate_reference import load_partition,best_cut
from score_absa import report

ROOT=Path(__file__).resolve().parent
SAFE_FILES=['config.json','metrics.json','history.csv','run_info.json','progress.json','ci_provenance.json']

def comparable_tune_score(path,metrics):
    # Older workers called the fixed-threshold epoch-selection score best_tune_F1.
    # Compare all candidates by tuned F1 at their already-selected checkpoint.
    history=pd.read_csv(path/'history.csv')
    row=history.loc[history.epoch==metrics['best_epoch']]
    if len(row)!=1:raise ValueError('Selected epoch missing from source history')
    return float(row['tune_F1_tuned_threshold'].iloc[0])

def design(sources,j,mode):
    idx=list(range(21)) if mode=='all' else list(range((j//3)*3,(j//3+1)*3))
    return np.concatenate([np.log(np.clip(s[:,idx],1e-5,1-1e-5)/(1-np.clip(s[:,idx],1e-5,1-1e-5))) for s in sources],axis=1)

def stack(trials,train):
    _,tu,va=reference_splits(train);y=train[LABELS].to_numpy(int);tune=[];validation=[]
    for _,path,_ in trials:
        _,t=load_partition(path/'tune_scores.npz',train,'tune');_,v=load_partition(path/'validation_scores.npz',train,'validation')
        tune.append(t);validation.append(v)
    scores=np.zeros((len(va),21));thresholds=[];models=[];labels=[]
    for j,label in enumerate(LABELS):
        target=y[tu,j];folds=list(StratifiedKFold(5,shuffle=True,random_state=2026).split(target,target));best=None
        for mode in ['own','all']:
            X=design(tune,j,mode)
            for c in [.003,.01,.03,.1,.3,1]:
                oof=np.zeros(len(tu))
                for a,b in folds:
                    m=make_pipeline(StandardScaler(),LogisticRegression(C=c,solver='liblinear',random_state=2026,max_iter=1500)).fit(X[a],target[a])
                    oof[b]=m.predict_proba(X[b])[:,1]
                cut=best_cut(target,oof,np.linspace(.1,.9,17));f1=f1_score(target,oof>=cut,zero_division=0)
                if best is None or f1>best[0]:best=(f1,mode,c,cut)
        f1,mode,c,cut=best
        model=make_pipeline(StandardScaler(),LogisticRegression(C=c,solver='liblinear',random_state=2026,max_iter=1500)).fit(design(tune,j,mode),target)
        scores[:,j]=model.predict_proba(design(validation,j,mode))[:,1]
        models.append(model);thresholds.append(cut);labels.append({'label':label,'mode':mode,'C':c,'threshold':cut,'meta_OOF_F1':f1})
    pred=binary_predictions(scores,np.asarray(thresholds));gt=train.iloc[va][['id']+LABELS].reset_index(drop=True)
    sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
    r=report(gt,sb,n_boot=1000);r.update(experiment='Stacking_lan2',split_policy='reference_group_2026',
        blind_file_read=False,validation_previously_seen=True,base_encoders_fine_tuned=True,fully_nested_encoder_cv=False)
    folder=ROOT/'Stacking_lan2';folder.mkdir(exist_ok=False)
    cfg={'sources':[n for n,_,_ in trials],'labels':labels,'seed':2026,'split_policy':'reference_group_2026',
        'selection':'meta C, feature mode and thresholds selected on tune OOF only','blind_file_read':False}
    (folder/'config.json').write_text(json.dumps(cfg,indent=2));(folder/'metrics.json').write_text(json.dumps(r,indent=2))
    private=ROOT/'ci_meta';private.mkdir(exist_ok=True)
    joblib.dump({'models':models,'config':cfg},private/'meta_models.joblib')
    np.savez_compressed(private/'validation_scores.npz',scores=scores,ids=gt.id.to_numpy(dtype=str))
    return r,[str((folder/n).relative_to(ROOT)) for n in ['config.json','metrics.json']]

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--artifacts',type=Path,required=True);ap.add_argument('--commit',action='store_true');a=ap.parse_args()
    train=pd.read_csv(ROOT/'train.csv');train_hash=hashlib.sha256((ROOT/'train.csv').read_bytes()).hexdigest()
    trials=[];files=[]
    for path in sorted(a.artifacts.glob('absa-*-metrics')):
        name=path.name.removeprefix('absa-').removesuffix('-metrics')
        if not re.fullmatch(r'PhoBERT_lan\d+',name) or not (path/'metrics.json').exists():continue
        m=json.loads((path/'metrics.json').read_text())
        if m['train_sha256']!=train_hash:raise ValueError('Training data changed: cannot combine source scores')
        load_partition(path/'tune_scores.npz',train,'tune');load_partition(path/'validation_scores.npz',train,'validation')
        folder=ROOT/name;folder.mkdir(exist_ok=True)
        for f in SAFE_FILES:
            if (path/f).exists():shutil.copy2(path/f,folder/f);files.append(f'{name}/{f}')
        trials.append((name,path,m))
    winner=max(trials,key=lambda t:comparable_tune_score(t[1],t[2])) if trials else None
    stacked=None
    if len(trials)>=2:stacked,extra=stack(trials,train);files+=extra
    rows=[{'experiment':n,'selected_checkpoint_tuned_F1':comparable_tune_score(path,m),'ABSA_F1':m['ABSA_F1'],'epochs_completed':m['epochs_completed']} for n,path,m in trials]
    if stacked:rows.append({'experiment':'Stacking_lan2','ABSA_F1':stacked['ABSA_F1']})
    pd.DataFrame(rows).to_csv(ROOT/'ci_training_comparison.csv',index=False);files.append('ci_training_comparison.csv')
    selection={'selected_encoder':winner[0] if winner else None,'selected_by':'tuned F1 at the selected checkpoint, never outer validation',
        'completed_encoder_trials':len(trials),'github_run_id':os.environ.get('GITHUB_RUN_ID'),
        'blind_file_read':False,'stacking_ABSA_F1':stacked['ABSA_F1'] if stacked else None}
    (ROOT/'ci_selected_run.json').write_text(json.dumps(selection,indent=2));files.append('ci_selected_run.json')
    text='# Automatic train-only fine-tuning results\n\n'+f'Completed encoder trials: {len(trials)} / 8.\n\n'
    for row in rows:text+=f"- {row['experiment']}: validation ABSA-F1 {row['ABSA_F1']:.6f}.\n"
    text+='\nValidation has been seen during development. No blind data was opened. Sources selected checkpoints on tune; meta CV is not fully nested encoder CV.\n'
    text+='\nOriginal FP32 checkpoints are cached by this workflow for retrieval; the selected encoder is packaged separately and re-evaluated after FP16 storage conversion.\n'
    (ROOT/'CI_TRAINING_REPORT.md').write_text(text);files.append('CI_TRAINING_REPORT.md')
    output=os.environ.get('GITHUB_OUTPUT')
    if output:
        with open(output,'a') as f:f.write('selected='+ (winner[0] if winner else '')+'\n')
    summary=os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary,'a') as f:f.write(text)
    if a.commit and trials:
        subprocess.run(['git','config','user.name','github-actions[bot]'],check=True)
        subprocess.run(['git','config','user.email','41898282+github-actions[bot]@users.noreply.github.com'],check=True)
        subprocess.run(['git','add','--',*files],check=True)
        if subprocess.run(['git','diff','--cached','--quiet']).returncode:
            subprocess.run(['git','commit','-m','Record real train-only fine-tuning and stacking results'],check=True)
            subprocess.run(['git','pull','--rebase','origin','main'],check=True)
            subprocess.run(['git','push','origin','HEAD:main'],check=True)
    print(json.dumps(selection),flush=True)
