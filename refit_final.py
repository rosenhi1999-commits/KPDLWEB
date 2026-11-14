"""Refit a selected sparse model on train.csv, without reading blind data."""
import argparse,json,time,hashlib,sys,platform
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from absa_core import LABELS, prepare_texts,features,SparseMultiLabel

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=Path,default=ROOT/'train.csv')
    ap.add_argument('--experiment',type=Path,required=True);a=ap.parse_args()
    if a.train.name!='train.csv':raise ValueError('Only train.csv accepted for training')
    folder=a.experiment;cfg=json.loads((folder/'config.json').read_text())
    tune=json.loads((folder/'inner_tuning.json').read_text())
    df=pd.read_csv(a.train);assert list(df.columns)==['id','text']+LABELS
    y=df[LABELS].to_numpy(int);assert np.isin(y,[0,1]).all()
    start=time.perf_counter();texts=prepare_texts(df.text,cfg['normalization'])
    vec=features(cfg);X=vec.fit_transform(texts);model=SparseMultiLabel(cfg).fit(X,y)
    bundle={'features':vec,'model':model,'config':cfg,'labels':LABELS,
        'thresholds':np.asarray(tune['thresholds'])}
    joblib.dump(bundle,folder/'final_model.joblib')
    meta={'fit_rows':len(df),'train_sha256':hashlib.sha256(a.train.read_bytes()).hexdigest(),
        'seconds':time.perf_counter()-start,'python':sys.version,'platform':platform.platform(),
        'model':'final_model.joblib','test_file_read':False,
        'thresholds_source':'inner tune; no blind labels or outer validation optimization'}
    (folder/'final_model_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))
