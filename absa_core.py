"""Train-derived ABSA utilities. No test text or external labeling service is used."""
import re
import unicodedata
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

ASPECTS = ['nhan_vien','sach_se','vi_tri','phong','gia_tri','an_uong','tien_nghi']
LABELS = [f'{a}_{s}' for a in ASPECTS for s in ['nhac','khen','che']]
MAIN = [i for i,c in enumerate(LABELS) if c.endswith(('_khen','_che'))]

def normalize(text, mode='base'):
    text = unicodedata.normalize('NFC', str(text)).lower()
    text = re.sub('[\u200b\u200c\u200d\ufeff]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if mode in ('lexical','segmented'):
        replacements = {'ksan':'khách sạn','ks':'khách sạn','nv':'nhân viên',
                        'ko':'không','khong':'không','k0':'không','hok':'không',
                        'đc':'được','dc':'được','vs':'với','gđ':'gia đình',
                        'buffe':'buffet','buffee':'buffet'}
        text = re.sub(r'(?u)\b(?:'+ '|'.join(replacements)+r')\b', lambda m:replacements[m[0]], text)
    if mode == 'segmented':
        from pyvi import ViTokenizer
        text = ViTokenizer.tokenize(text)
    return text

def prepare_texts(texts, mode):
    return [normalize(t, mode) for t in texts]

def splits(n, y, seed=42):
    outer, valid = train_test_split(np.arange(n),test_size=.2,random_state=seed)
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    splitter=MultilabelStratifiedShuffleSplit(n_splits=1,test_size=.25,random_state=seed)
    fit,tune=next(splitter.split(outer, y[outer]))
    return outer[fit],outer[tune],valid

def reference_splits(df,seed=2026):
    """Exactly the clean/lowercase SHA256 GroupShuffleSplit supplied by the user."""
    import hashlib
    clean=lambda s:re.sub(r'\s+',' ',unicodedata.normalize('NFC',str(s))).strip()
    groups=df.text.map(lambda s:hashlib.sha256(clean(s).lower().encode()).hexdigest())
    outer,valid=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(df,groups=groups))
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1).split(df.iloc[outer],groups=groups.iloc[outer]))
    fit,tune=outer[a],outer[b]
    assert not set(groups.iloc[fit])&set(groups.iloc[tune])
    assert not set(groups.iloc[outer])&set(groups.iloc[valid])
    return fit,tune,valid

def features(cfg):
    vecs=[]
    if cfg.get('word',True):
        vecs.append(('word',TfidfVectorizer(ngram_range=(1,cfg.get('word_max',2)),
            min_df=2,sublinear_tf=True,strip_accents=None,max_features=120000,dtype=np.float64)))
    if cfg.get('char',True):
        vecs.append(('char',TfidfVectorizer(analyzer=cfg.get('char_analyzer','char'),ngram_range=(3,5),
            min_df=3,sublinear_tf=True,max_features=150000,dtype=np.float64)))
    return FeatureUnion(vecs)

class SparseMultiLabel:
    def __init__(self, cfg):
        self.cfg=cfg

    def fit(self, X, y):
        self.models=[];self.ratios=[]
        X=X.tocsr();X.sort_indices()
        kind=self.cfg['kind'];c=self.cfg.get('C',1)
        for j in range(y.shape[1]):
            yy=y[:,j];ratio=None
            if len(np.unique(yy))<2:
                model=float(yy[0])
            else:
                weight='balanced' if self.cfg.get('balanced',True) else None
                if kind in ('svm','nbsvm'):
                    model=LinearSVC(C=c,class_weight=weight,max_iter=6000,random_state=42)
                elif kind=='logistic':
                    model=LogisticRegression(C=c,class_weight=weight,solver='liblinear',max_iter=1500,random_state=42)
                elif kind=='sgd':
                    model=SGDClassifier(loss='modified_huber',alpha=self.cfg.get('alpha',1e-4),
                        class_weight=weight,max_iter=1500,tol=1e-4,random_state=42)
                elif kind=='ridge':
                    model=RidgeClassifier(alpha=self.cfg.get('alpha',1),class_weight=weight,solver='lsqr')
                elif kind=='complementnb':
                    model=ComplementNB(alpha=self.cfg.get('alpha',1))
                else:
                    raise ValueError(kind)
                if kind=='nbsvm':
                    p=np.asarray(X[yy==1].sum(axis=0)).ravel()+1
                    q=np.asarray(X[yy==0].sum(axis=0)).ravel()+1
                    ratio=np.log((p/p.sum())/(q/q.sum()))
                xx=X if ratio is None else X.multiply(ratio).tocsr()
                model.fit(xx,yy)
            self.models.append(model);self.ratios.append(ratio)
        return self

    def decision_function(self,X):
        result=[]
        for model,ratio in zip(self.models,self.ratios):
            if isinstance(model,float):
                score=np.full(X.shape[0],20 if model else -20,dtype=float)
            else:
                xx=X if ratio is None else X.multiply(ratio).tocsr()
                if hasattr(model,'decision_function'):
                    score=model.decision_function(xx)
                else:
                    p=np.clip(model.predict_proba(xx)[:,1],1e-8,1-1e-8)
                    score=np.log(p/(1-p))
            result.append(score)
        return np.asarray(result).T

def tune_thresholds(y,scores):
    thresholds=[]
    for j in range(y.shape[1]):
        candidates=np.unique(np.r_[0,np.quantile(scores[:,j],np.linspace(.05,.95,37))])
        # A deterministic tie-break prefers the default boundary.
        ranked=[(f1_score(y[:,j],scores[:,j]>=t,zero_division=0),-abs(t),t) for t in candidates]
        thresholds.append(max(ranked)[2])
    return np.asarray(thresholds)

def binary_predictions(scores,thresholds):
    out=(scores>=thresholds).astype(np.int8)
    for j in range(0,21,3):out[:,j]=out[:,j:j+3].max(axis=1)
    return out

def absa_f1(y,pred):
    return f1_score(y[:,MAIN],pred[:,MAIN],average='macro',zero_division=0)
