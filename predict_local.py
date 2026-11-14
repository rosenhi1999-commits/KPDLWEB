"""Local model inference. Never send input texts to an API/LLM or log them."""
import argparse,json,time
from pathlib import Path
import joblib
import pandas as pd
from absa_core import prepare_texts,binary_predictions,LABELS
from submission_io import write_submission

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True,type=Path)
    ap.add_argument('--local-test',required=True,type=Path)
    ap.add_argument('--template',required=True,type=Path)
    ap.add_argument('--output',default='submission.csv',type=Path)
    a=ap.parse_args()
    # Load only self-produced trusted joblib artifacts.
    bundle=joblib.load(a.model)
    if a.model.name!='final_model.joblib':raise ValueError('Use a final model refitted on all training rows')
    if bundle['labels']!=LABELS:raise ValueError('Unexpected model label order')
    df=pd.read_csv(a.local_test,dtype={'id':str},keep_default_na=False)
    if list(df.columns)!=['id','text']:raise ValueError('Test must have columns id,text')
    start=time.perf_counter()
    texts=prepare_texts(df.text,bundle['config']['normalization'])
    X=bundle['features'].transform(texts)
    scores=bundle['model'].decision_function(X)
    predictions=binary_predictions(scores,bundle['thresholds'])
    info=write_submission(df.id,predictions,a.template,a.output)
    info['inference_seconds']=time.perf_counter()-start
    print(json.dumps(info,ensure_ascii=False))
