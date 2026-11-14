"""Learn aspect attention over frozen PhoBERT token features, TRAIN ONLY.

This trains a supervised neural head, not the quantized encoder. PCA/SVD are fit
on the training partition only. Epochs and thresholds are selected on tune.
"""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import hashlib,json,random,time
from pathlib import Path
import train_only_guard
import numpy as np,pandas as pd,torch,joblib
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.preprocessing import StandardScaler
from absa_core import LABELS,reference_splits,prepare_texts,features,absa_f1,binary_predictions
from neural_train import cuts
from score_absa import report

ROOT=Path(__file__).resolve().parent

def seed():
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
    torch.use_deterministic_algorithms(True,warn_only=True)

class Tokens(Dataset):
    def __init__(self,tokens,spans,global_x,y,indices):
        self.tokens=tokens;self.spans=spans;self.global_x=global_x;self.y=y;self.indices=indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,k):
        i=self.indices[k]
        pieces=[self.tokens[start+1:start+length-1] if length>2 else self.tokens[start:start+length] for start,length in self.spans[i]]
        return torch.from_numpy(np.concatenate(pieces)),torch.tensor(self.global_x[i]),torch.tensor(self.y[i],dtype=torch.float32)

def collate(items):
    tokens,g,y=zip(*items);lengths=torch.tensor([len(t) for t in tokens])
    x=pad_sequence(tokens,batch_first=True)
    mask=torch.arange(x.shape[1])[None,:]<lengths[:,None]
    return x,mask,torch.stack(g),torch.stack(y)

class Attention(nn.Module):
    def __init__(self,global_dim,hybrid,dropout):
        super().__init__();self.hybrid=hybrid
        self.project=nn.Sequential(nn.Linear(96,64),nn.Tanh(),nn.Dropout(dropout))
        self.queries=nn.Parameter(torch.randn(7,64)*.1)
        self.weights=nn.Parameter(torch.randn(7,64,3)*.02);self.bias=nn.Parameter(torch.zeros(7,3))
        self.drop=nn.Dropout(dropout)
        if hybrid:self.global_head=nn.Sequential(nn.Dropout(dropout),nn.Linear(global_dim,21))
    def forward(self,x,mask,g):
        h=self.project(x)
        att=torch.einsum('bth,ah->bat',h,self.queries).masked_fill(~mask[:,None,:],-1e4).softmax(-1)
        pooled=self.drop(torch.einsum('bat,bth->bah',att,h))
        logits=(torch.einsum('bah,ahc->bac',pooled,self.weights)+self.bias).reshape(-1,21)
        if self.hybrid:logits=logits+self.global_head(g)
        return logits

def load_batches(tokens,spans,g,y,idx,shuffle=False):
    return DataLoader(Tokens(tokens,spans,g,y,idx),batch_size=32,shuffle=shuffle,
        generator=torch.Generator().manual_seed(2026),collate_fn=collate,num_workers=0)

@torch.no_grad()
def infer(model,loader):
    model.eval();return np.concatenate([model(x,mask,g).numpy() for x,mask,g,_ in loader])

def prepare(raw,spans,texts,fit):
    rng=np.random.default_rng(2026)
    candidates=np.concatenate([np.arange(start,start+length) for i in fit for start,length in spans[i]])
    sampled=rng.choice(candidates,size=min(20000,len(candidates)),replace=False)
    pca=PCA(n_components=96,svd_solver='randomized',random_state=2026)
    pca.fit(np.asarray(raw[sampled],dtype=np.float32))
    tokens=np.empty((len(raw),96),dtype=np.float32)
    for start in range(0,len(raw),8192):tokens[start:start+8192]=pca.transform(np.asarray(raw[start:start+8192],dtype=np.float32))
    # Fixed scale fitted on training token samples only, not validation.
    scale=np.std(tokens[sampled],axis=0).clip(.01);tokens/=scale
    vec=features({'char':False,'word_max':3});X=vec.fit_transform(texts[fit]);V=vec.transform(texts)
    svd=TruncatedSVD(n_components=128,random_state=2026);svd.fit(X)
    global_x=svd.transform(V).astype(np.float32)
    scaler=StandardScaler().fit(global_x[fit]);global_x=scaler.transform(global_x).astype(np.float32)
    return tokens,global_x,{'pca':pca,'token_scale':scale,'vectorizer':vec,'svd':svd,'global_scaler':scaler}

def epoch(model,loader,opt,pw):
    model.train();total=0
    for x,mask,g,y in loader:
        opt.zero_grad(set_to_none=True);logits=model(x,mask,g)
        loss=nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=pw)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=float(loss.detach())
    return total/len(loader)

if __name__=='__main__':
    torch.set_num_threads(2);seed();df=pd.read_csv(ROOT/'train.csv')
    assert list(df.columns)==['id','text']+LABELS
    meta=json.loads((ROOT/'private_phobert_tokens.json').read_text())
    assert meta['train_sha256']==hashlib.sha256((ROOT/'train.csv').read_bytes()).hexdigest()
    raw=np.memmap(ROOT/'private_phobert_tokens.f16',dtype=np.float16,mode='r',shape=tuple(meta['shape']))
    spans=[[] for _ in range(len(df))]
    for i,start,length in meta['chunks']:spans[i].append((start,length))
    y=df[LABELS].to_numpy(np.float32);fit,tune,valid=reference_splits(df);outer=np.r_[fit,tune]
    texts=np.asarray(prepare_texts(df.text,'lexical'),dtype=object)
    tokens,g,preprocessing=prepare(raw,spans,texts,fit)
    configs=[('PhoBERT_Attention_lan1',dict(hybrid=False,dropout=.2,lr=.002)),
        ('PhoBERT_Attention_lan2',dict(hybrid=True,dropout=.3,lr=.001))]
    selected=[]
    for name,cfg in configs:
        folder=ROOT/name;folder.mkdir(exist_ok=False)
        cfg.update(encoder_frozen=True,max_epochs=24,patience=5,split_policy='reference_group_2026',blind_file_read=False)
        (folder/'config.json').write_text(json.dumps(cfg,indent=2))
        seed();model=Attention(g.shape[1],cfg['hybrid'],cfg['dropout'])
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.01)
        pw=torch.tensor(np.clip(np.sqrt((len(fit)-y[fit].sum(0))/np.maximum(1,y[fit].sum(0))),1,3))
        train_loader=load_batches(tokens,spans,g,y,fit,True);tune_loader=load_batches(tokens,spans,g,y,tune)
        best=-1;best_epoch=0;history=[];start=time.perf_counter()
        for ep in range(1,cfg['max_epochs']+1):
            loss=epoch(model,train_loader,opt,pw);scores=infer(model,tune_loader)
            thresholds=cuts(y[tune],scores);f1=absa_f1(y[tune],binary_predictions(scores,thresholds))
            history.append({'epoch':ep,'loss':loss,'tune_F1':f1});print(name,history[-1],flush=True)
            if f1>best:
                best=f1;best_epoch=ep;best_cuts=thresholds
                torch.save(model.state_dict(),folder/'inner_best.pt')
            if ep-best_epoch>=cfg['patience']:break
        selected.append((name,cfg,best_epoch,best_cuts,best,time.perf_counter()-start))
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    # No outer-validation feedback was used for selecting either head.
    del tokens,g,preprocessing
    tokens,g,preprocessing=prepare(raw,spans,texts,outer);rows=[]
    for name,cfg,best_epoch,thresholds,best,inner_seconds in selected:
        folder=ROOT/name;seed();model=Attention(g.shape[1],cfg['hybrid'],cfg['dropout'])
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.01)
        pw=torch.tensor(np.clip(np.sqrt((len(outer)-y[outer].sum(0))/np.maximum(1,y[outer].sum(0))),1,3))
        train_loader=load_batches(tokens,spans,g,y,outer,True);start=time.perf_counter()
        for ep in range(best_epoch):
            loss=epoch(model,train_loader,opt,pw)
            print(name,'outer80_refit_epoch',ep+1,'/',best_epoch,'loss',round(loss,5),flush=True)
        seconds=time.perf_counter()-start;start=time.perf_counter()
        scores=infer(model,load_batches(tokens,spans,g,y,valid));inference=time.perf_counter()-start
        pred=binary_predictions(scores,thresholds);gt=df.iloc[valid][['id']+LABELS].reset_index(drop=True)
        sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
        r=report(gt,sb,n_boot=1000);r.update(experiment=name,best_epoch=best_epoch,inner_tune_ABSA_F1=best,
            inner_seconds=inner_seconds,refit_seconds=seconds,inference_seconds=inference,
            encoder_frozen=True,split_policy='reference_group_2026',validation_previously_seen=True,blind_file_read=False)
        (folder/'metrics.json').write_text(json.dumps(r,indent=2))
        torch.save(model.state_dict(),folder/'validation_head.pt');joblib.dump(preprocessing,folder/'validation_preprocessing.joblib')
        np.save(folder/'private_validation_scores.npy',scores);sb.to_csv(folder/'validation_predictions.csv',index=False)
        rows.append({'experiment':name,'ABSA_F1':r['ABSA_F1'],'CI95_low':r['ABSA_F1_CI95'][0],'CI95_high':r['ABSA_F1_CI95'][1]})
        print(name,'outer_validation_ABSA_F1',r['ABSA_F1'],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'frozen_attention_comparison.csv',index=False)
