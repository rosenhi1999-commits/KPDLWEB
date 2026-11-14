"""Frozen pretrained features from TRAIN ONLY, using CPU dynamic int8 quantization.

This is not encoder fine-tuning and does not use an API for labels.
"""
import os
from pathlib import Path
import train_only_guard
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'private_hf_cache'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse,hashlib,json,time
import numpy as np,pandas as pd,torch
from transformers import AutoModel,AutoTokenizer
from pyvi import ViTokenizer
from neural_train import clean

NAME='vinai/phobert-base-v2';REV='86cd7fd4c148980922ac11a2cf5e257f2ba639e1'

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--token-cache',action='store_true')
    ap.add_argument('--segmenter',choices=['pyvi','rdr'],default='pyvi');args=ap.parse_args()
    feature_stem='phobert' if args.segmenter=='pyvi' else 'phobert_rdr'
    segment=lambda text:ViTokenizer.tokenize(clean(text))
    if args.segmenter=='rdr':
        import shutil
        java=shutil.which('java')
        if java:os.environ.setdefault('JAVA_HOME',str(Path(java).resolve().parent.parent))
        import py_vncorenlp
        segmenter=py_vncorenlp.VnCoreNLP(annotators=['wseg'],save_dir=str(ROOT/'private_vncorenlp'))
        segment=lambda text:' '.join(segmenter.word_segment(clean(text)))
    torch.set_num_threads(2)
    started=time.perf_counter();df=pd.read_csv(ROOT/'train.csv')
    train_hash=hashlib.sha256((ROOT/'train.csv').read_bytes()).hexdigest()
    tok=AutoTokenizer.from_pretrained(NAME,revision=REV,use_fast=False,trust_remote_code=False)
    model=AutoModel.from_pretrained(NAME,revision=REV,add_pooling_layer=False,trust_remote_code=False).eval()
    model=torch.ao.quantization.quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8)
    width=256-tok.num_special_tokens_to_add(pair=False);stride=64;chunks=[]
    for i,text in enumerate(df.text):
        ids=tok.encode(segment(text),add_special_tokens=False)
        for start in range(0,max(1,len(ids)),width-stride):
            seq=tok.build_inputs_with_special_tokens(ids[start:start+width])
            chunks.append((i,{'input_ids':seq,'attention_mask':[1]*len(seq)}))
            if start+width>=len(ids):break
    chunks.sort(key=lambda x:len(x[1]['input_ids']))
    token_store=None;token_chunks=[];offset=0
    if args.token_cache:
        total_tokens=sum(len(item['input_ids']) for _,item in chunks)
        token_store=np.memmap(ROOT/('private_'+feature_stem+'_tokens.f16'),dtype=np.float16,mode='w+',shape=(total_tokens,model.config.hidden_size))
    dim=model.config.hidden_size*3;pooled=np.full((len(df),dim),-np.inf,dtype=np.float32)
    print('Reviews',len(df),'windows',len(chunks),'feature_dim',dim,flush=True)
    with torch.inference_mode():
        for start in range(0,len(chunks),16):
            group=chunks[start:start+16];batch=tok.pad([x[1] for x in group],padding=True,return_tensors='pt')
            out=model(**batch,output_hidden_states=True)
            mask=batch['attention_mask'].bool()&(batch['input_ids']!=tok.bos_token_id)&(batch['input_ids']!=tok.eos_token_id)
            mask=mask[:,:,None];denom=mask.sum(1).clamp(min=1)
            mean=(out.last_hidden_state*mask).sum(1)/denom
            avg4=(torch.stack(out.hidden_states[-4:]).mean(0)*mask).sum(1)/denom
            vectors=torch.cat([out.last_hidden_state[:,0],mean,avg4],1).numpy()
            if token_store is not None:
                hidden=out.last_hidden_state.numpy()
                for j,(review,item) in enumerate(group):
                    length=len(item['input_ids']);token_store[offset:offset+length]=hidden[j,:length]
                    token_chunks.append([review,offset,length]);offset+=length
            for (i,_),v in zip(group,vectors):pooled[i]=np.maximum(pooled[i],v)
            if start%320==0 or start+16>=len(chunks):
                print('Encoded windows',min(start+16,len(chunks)),'/',len(chunks),'seconds',round(time.perf_counter()-started,1),flush=True)
    assert np.isfinite(pooled).all()
    if token_store is not None:
        token_store.flush()
        token_meta={'shape':[offset,model.config.hidden_size],'chunks':token_chunks,'train_sha256':train_hash,
            'model_name':NAME,'revision':REV,'encoder_frozen':True,'quantization':'dynamic_int8_linear','segmenter':args.segmenter,'blind_file_read':False}
        (ROOT/('private_'+feature_stem+'_tokens.json')).write_text(json.dumps(token_meta))
    np.save(ROOT/('private_'+feature_stem+'_features.npy'),pooled)
    meta={'model_name':NAME,'revision':REV,'train_sha256':train_hash,'rows':len(df),
        'windows':len(chunks),'feature_dim':dim,'seconds':time.perf_counter()-started,
        'encoder_frozen':True,'supervised_encoder_finetuning':False,'quantization':'dynamic_int8_linear','segmenter':args.segmenter,
        'torch':torch.__version__,'threads':2,'pooling':'last_CLS + last_masked_mean + last4_masked_mean; max over windows',
        'blind_file_read':False,'label_access_during_extraction':False}
    (ROOT/(feature_stem+'_feature_extraction.json')).write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)
