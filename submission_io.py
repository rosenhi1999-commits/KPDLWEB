"""Strict schema checks and CSV writing. This module never reads review text."""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from absa_core import LABELS, binary_predictions

def read_template(path):
    df=pd.read_csv(path,dtype=str,keep_default_na=False)
    if list(df.columns)!=['id']+LABELS:raise ValueError('Template must have the exact ordered 22 columns')
    if len(df)!=732:raise ValueError('Expected 732 template rows')
    if df.id.eq('').any() or df.id.duplicated().any():raise ValueError('Missing or duplicate template IDs')
    return df

def validate_submission(path,template_path):
    template=read_template(template_path)
    with open(path,encoding='utf-8-sig',newline='') as f:
        rows=list(csv.reader(f))
    if not rows or rows[0]!=['id']+LABELS:raise ValueError('Wrong column names or order')
    if len(rows)-1!=732:raise ValueError('Expected exactly 732 rows')
    if any(len(r)!=22 for r in rows[1:]):raise ValueError('Every row must contain 22 fields')
    ids=[r[0] for r in rows[1:]]
    if len(set(ids))!=len(ids) or set(ids)!=set(template.id):raise ValueError('IDs do not match template')
    if any(v not in ('0','1') for r in rows[1:] for v in r[1:]):
        raise ValueError('Labels must be literal 0/1, not floats, booleans or probabilities')
    arr=np.array([r[1:] for r in rows[1:]],dtype=np.int8).reshape(732,7,3)
    if np.any((arr[:,:,1:].max(axis=2)==1)&(arr[:,:,0]==0)):
        raise ValueError('Positive khen/che requires nhac=1')
    return {'valid':True,'rows':732,'columns':22,'template_ids_match':True,
            'note':'Template check only; independently compare IDs to test_blind.csv before submission.'}

def write_submission(ids,predictions,template_path,output_path):
    template=read_template(template_path)
    ids=list(ids);predictions=np.asarray(predictions)
    if predictions.shape!=(len(ids),21):raise ValueError('Wrong prediction shape')
    if not np.isin(predictions,[0,1]).all():raise ValueError('Only binary model predictions can be written')
    if len(set(ids))!=len(ids) or set(ids)!=set(template.id):raise ValueError('Input IDs do not match template')
    out=pd.DataFrame(predictions.astype(np.int8),columns=LABELS,index=ids)
    out=out.loc[template.id].copy()
    for j in range(0,21,3):out.iloc[:,j]=out.iloc[:,j:j+3].max(axis=1)
    out.index.name='id';out.to_csv(output_path,index=True,encoding='utf-8')
    return validate_submission(output_path,template_path)

if __name__=='__main__':
    import argparse,json
    ap=argparse.ArgumentParser();ap.add_argument('submission');ap.add_argument('template')
    args=ap.parse_args();print(json.dumps(validate_submission(args.submission,args.template),ensure_ascii=False))
