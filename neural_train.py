"""Supervised neural experiments adapted from the user's PhoBERT reference.

Only train.csv is accepted. Blind inference is deliberately absent.
Joint and label-attention heads preserve review-level multi-window supervision.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'private_hf_cache'))
import argparse,hashlib,json,math,platform,random,re,time,unicodedata
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from transformers import AutoConfig,AutoModel,AutoTokenizer,get_cosine_schedule_with_warmup
from pyvi import ViTokenizer
from absa_core import LABELS,reference_splits,binary_predictions,absa_f1
from score_absa import report

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)

def clean(text):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFC',str(text))).strip()

class Reviews(Dataset):
    def __init__(self,texts,labels,tokenizer,segment,max_length=256,stride=64):
        self.labels=labels;self.items=[]
        width=max_length-tokenizer.num_special_tokens_to_add(pair=False)
        if not 0<=stride<width:raise ValueError('Invalid window stride')
        for text in texts:
            text=ViTokenizer.tokenize(clean(text)) if segment else clean(text)
            ids=tokenizer.encode(text,add_special_tokens=False);chunks=[]
            for start in range(0,max(1,len(ids)),width-stride):
                seq=tokenizer.build_inputs_with_special_tokens(ids[start:start+width])
                chunks.append({'input_ids':seq,'attention_mask':[1]*len(seq)})
                if start+width>=len(ids):break
            self.items.append(chunks)
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return self.items[i],self.labels[i]

class Collate:
    def __init__(self,tokenizer):self.tokenizer=tokenizer
    def __call__(self,batch):
        windows=[w for chunks,_ in batch for w in chunks]
        return self.tokenizer.pad(windows,padding=True,return_tensors='pt'),[len(c) for c,_ in batch],torch.tensor(np.asarray([y for _,y in batch]),dtype=torch.float32)

class ReviewModel(nn.Module):
    def __init__(self,name,revision,head='joint',unfreeze=-1):
        super().__init__();self.head_type=head
        cfg=AutoConfig.from_pretrained(name,revision=revision,trust_remote_code=False)
        self.encoder=AutoModel.from_pretrained(name,revision=revision,config=cfg,add_pooling_layer=False,trust_remote_code=False)
        hidden=cfg.hidden_size
        if head=='joint':
            self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(hidden,hidden),nn.Tanh(),nn.Dropout(.2),nn.Linear(hidden,21))
        else:
            self.queries=nn.Parameter(torch.randn(7,hidden)*.02)
            self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(hidden,3))
        if unfreeze>=0:
            layers=self.encoder.encoder.layer
            if unfreeze>len(layers):raise ValueError('Too many unfreeze layers')
            for p in self.encoder.parameters():p.requires_grad=False
            if unfreeze:
                for layer in layers[-unfreeze:]:
                    for p in layer.parameters():p.requires_grad=True
    def forward(self,inputs,counts):
        hidden=self.encoder(**inputs).last_hidden_state
        if self.head_type=='joint':logits=self.head(hidden[:,0])
        else:
            energy=torch.einsum('bth,ah->bat',hidden,self.queries)/math.sqrt(hidden.shape[-1])
            energy=energy.masked_fill(~inputs['attention_mask'].bool()[:,None,:],-1e4)
            pooled=torch.einsum('bat,bth->bah',energy.softmax(-1),hidden)
            logits=self.head(pooled).reshape(-1,21)
        # One review-level BCE target; never copy all labels to each window.
        return torch.stack([part.max(0).values for part in logits.split(counts)])

def batches(ds,tok,size,shuffle=False,seed=2026):
    return DataLoader(ds,batch_size=size,shuffle=shuffle,collate_fn=Collate(tok),num_workers=0,generator=torch.Generator().manual_seed(seed))

@torch.no_grad()
def infer(model,loader,device):
    model.eval();out=[]
    for inputs,counts,_ in loader:
        out.append(model({k:v.to(device) for k,v in inputs.items()},counts).cpu().numpy())
    return np.concatenate(out)

def cuts(y,logits):
    probs=1/(1+np.exp(-np.clip(logits,-30,30)))
    thresholds=[]
    from sklearn.metrics import f1_score
    for j in range(21):
        grid=np.linspace(.15,.75,13)
        t=max(grid,key=lambda t:(f1_score(y[:,j],probs[:,j]>=t,zero_division=0),-abs(t-.5)))
        thresholds.append(np.log(t/(1-t)))
    return np.asarray(thresholds)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--train',type=Path,default=ROOT/'train.csv');ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();cfg=json.loads(args.config.read_text());out=args.config.parent
    if args.train.name!='train.csv':raise ValueError('Train-only runner')
    if not args.smoke and (out/'metrics.json').exists():raise FileExistsError('Create a new method_lanN directory for a new run')
    seed_all(cfg.get('seed',2026));torch.set_num_threads(3)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu' and not args.smoke:raise RuntimeError('Full training needs a GPU in this runner; --smoke is supported on CPU')
    df=pd.read_csv(args.train);assert list(df.columns)==['id','text']+LABELS
    y=df[LABELS].to_numpy(int);assert np.isin(y,[0,1]).all()
    fit,tune,valid=reference_splits(df,cfg.get('seed',2026))
    tokenizer=AutoTokenizer.from_pretrained(cfg['model_name'],revision=cfg.get('revision'),use_fast=cfg.get('fast_tokenizer',False),trust_remote_code=False)
    idxs={'fit':fit[:4] if args.smoke else fit,'tune':tune,'validation':valid}
    datasets={n:Reviews(df.text.iloc[idx],y[idx],tokenizer,cfg['segment'],cfg.get('max_length',256),cfg.get('stride',64)) for n,idx in idxs.items() if not args.smoke or n=='fit'}
    model=ReviewModel(cfg['model_name'],cfg.get('revision'),cfg.get('head','joint'),cfg.get('unfreeze_layers',-1)).to(device)
    lr=cfg.get('lr',2e-5);encoder_params=list(model.encoder.parameters())
    encoder_ids={id(p) for p in encoder_params}
    opt=torch.optim.AdamW([{'params':[p for p in encoder_params if p.requires_grad],'lr':lr},
        {'params':[p for p in model.parameters() if id(p) not in encoder_ids],'lr':lr*5}],weight_decay=.01)
    loaders={n:batches(ds,tokenizer,cfg.get('batch_size',4),n=='fit') for n,ds in datasets.items()}
    if args.smoke:
        inputs,counts,labels=next(iter(loaders['fit']));start=time.perf_counter()
        logits=model({k:v.to(device) for k,v in inputs.items()},counts)
        loss=nn.functional.binary_cross_entropy_with_logits(logits,labels.to(device));loss.backward();opt.step()
        assert logits.shape==(len(labels),21) and torch.isfinite(loss)
        result={'smoke_passed':True,'device':str(device),'seconds':time.perf_counter()-start,'loss':float(loss.detach()),'note':'Not a validation F1 experiment','blind_file_read':False}
        (out/'smoke.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
    epochs=cfg.get('epochs',8);acc=cfg.get('accumulation',4)
    updates=math.ceil(len(loaders['fit'])/acc)*epochs
    sched=get_cosine_schedule_with_warmup(opt,int(updates*.1),updates)
    pw=torch.tensor(np.clip(np.sqrt((len(fit)-y[fit].sum(0))/np.maximum(1,y[fit].sum(0))),1,3),dtype=torch.float32,device=device)
    scaler=torch.amp.GradScaler('cuda');history=[];best=-1;best_epoch=0;start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train()
        if not any(p.requires_grad for p in model.encoder.parameters()):model.encoder.eval()
        opt.zero_grad(set_to_none=True);total=0
        for step,(inputs,counts,labels) in enumerate(loaders['fit']):
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model({k:v.to(device) for k,v in inputs.items()},counts)
                loss=nn.functional.binary_cross_entropy_with_logits(logits,labels.to(device),pos_weight=pw)
                divisor=min(acc,len(loaders['fit'])-(step//acc)*acc)
            scaler.scale(loss/divisor).backward();total+=float(loss.detach())
            if (step+1)%acc==0 or step+1==len(loaders['fit']):
                scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1)
                scaler.step(opt);scaler.update();sched.step();opt.zero_grad(set_to_none=True)
        logits=infer(model,loaders['tune'],device);thresholds=cuts(y[tune],logits)
        f1=absa_f1(y[tune],binary_predictions(logits,thresholds))
        history.append({'epoch':epoch,'loss':total/len(loaders['fit']),'tune_F1':f1});print(history[-1],flush=True)
        if f1>best:
            best=f1;best_epoch=epoch
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'thresholds':thresholds.tolist()},out/'best.pt')
        if epoch-best_epoch>=cfg.get('patience',3):break
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['state_dict'])
    elapsed=time.perf_counter()-start;start=time.perf_counter();logits=infer(model,loaders['validation'],device)
    seconds=time.perf_counter()-start;pred=binary_predictions(logits,np.asarray(saved['thresholds']))
    gt=df.iloc[valid][['id']+LABELS].reset_index(drop=True);sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
    result=report(gt,sb,n_boot=1000);result.update({'best_epoch':best_epoch,'best_tune_F1':best,
        'train_seconds':elapsed,'validation_inference_seconds':seconds,'train_sha256':hashlib.sha256(args.train.read_bytes()).hexdigest(),
        'torch':torch.__version__,'transformers':__import__('transformers').__version__,'python':platform.python_version(),
        'gpu':torch.cuda.get_device_name(),'VRAM_bytes':torch.cuda.get_device_properties(0).total_memory,
        'blind_file_read':False,'production_refit':False,'validation_previously_seen':True})
    (out/'metrics.json').write_text(json.dumps(result,indent=2));pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    cfg['status']='trained_validation_only'
    args.config.write_text(json.dumps(cfg,indent=2))
    np.savez_compressed(out/'private_validation_scores.npz',scores=logits,ids=gt.id.to_numpy(dtype=str))
    sb.to_csv(out/'validation_predictions.csv',index=False);tokenizer.save_pretrained(out/'tokenizer')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
