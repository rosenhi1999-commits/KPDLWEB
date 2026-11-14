import json, re, time, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import joblib

ROOT=Path(__file__).resolve().parent

def normalize(text):
    # Preserve accents, negations, placeholders and punctuation.
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC',str(text))).strip().lower()

if __name__ == '__main__':
    df=pd.read_csv(ROOT.parent/'train.csv')
    labels=list(df.columns[2:]); y=df[labels].to_numpy(dtype=int)
    train,valid=train_test_split(np.arange(len(df)),test_size=.2,random_state=42)
    pd.DataFrame({'id':df.id,'split':np.where(np.isin(np.arange(len(df)),valid),'validation','train')}).to_csv(ROOT/'split.csv',index=False)
    features=FeatureUnion([
        ('word',TfidfVectorizer(preprocessor=normalize,ngram_range=(1,2),min_df=2,sublinear_tf=True)),
        ('char',TfidfVectorizer(preprocessor=normalize,analyzer='char',ngram_range=(3,5),min_df=3,sublinear_tf=True,max_features=150000))])
    started=time.perf_counter()
    X=features.fit_transform(df.text.iloc[train]); V=features.transform(df.text.iloc[valid])
    clf=OneVsRestClassifier(LinearSVC(C=1,class_weight='balanced',max_iter=5000,random_state=42),n_jobs=1)
    clf.fit(X,y[train]); duration=time.perf_counter()-started
    pred=clf.predict(V)
    for j,c in enumerate(labels):
        if c.endswith('_nhac'): pred[:,j]=np.maximum(pred[:,j],np.maximum(pred[:,j+1],pred[:,j+2]))
    scored=[i for i,c in enumerate(labels) if c.endswith(('_khen','_che'))]
    a=y[valid][:,scored]; b=pred[:,scored]
    result={'seed':42,'train_rows':len(train),'validation_rows':len(valid),'ABSA_F1':f1_score(a,b,average='macro',zero_division=0),'precision_macro':precision_score(a,b,average='macro',zero_division=0),'recall_macro':recall_score(a,b,average='macro',zero_division=0),'accuracy_mean_per_label':float((a==b).mean()),'train_seconds':duration,'per_label_f1':dict(zip([labels[i] for i in scored],f1_score(a,b,average=None,zero_division=0).tolist()))}
    (ROOT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    joblib.dump({'features':features,'model':clf,'labels':labels},ROOT/'validation_model.joblib')
    print(json.dumps(result,ensure_ascii=False,indent=2))
