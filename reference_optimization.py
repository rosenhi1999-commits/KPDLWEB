"""Configurable, review-level loss and optimizer for the supplied reference."""
import re
import numpy as np
import torch
from torch import nn

def optimizer_for(model,cfg):
    lr=float(cfg.get('lr',2e-5));mult=float(cfg.get('head_lr_multiplier',5))
    decay=float(cfg.get('layer_lr_decay',1));wd=float(cfg.get('weight_decay',.01))
    if lr<=0 or mult<=0 or not 0<decay<=1:raise ValueError('Invalid optimizer settings')
    layers=len(model.encoder.encoder.layer);groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if name.startswith('encoder.'):
            match=re.search(r'encoder\.layer\.(\d+)\.',name)
            depth=layers-1-int(match[1]) if match else layers
            rate=lr*decay**depth
        else:rate=lr*mult
        no_decay=cfg.get('exclude_bias_norm_decay',False) and (name.endswith('.bias') or 'layernorm.weight' in name.lower())
        key=(rate,0. if no_decay else wd)
        groups.setdefault(key,[]).append(p)
    return torch.optim.AdamW([{'params':params,'lr':rate,'weight_decay':weight_decay}
        for (rate,weight_decay),params in groups.items()])

def positive_weights(y,cfg,device):
    power=float(cfg.get('positive_weight_power',.5));cap=float(cfg.get('positive_weight_cap',3))
    if not 0<=power<=1 or cap<1:raise ValueError('Invalid positive weighting')
    ratio=(len(y)-y.sum(0))/np.maximum(1,y.sum(0))
    return torch.as_tensor(np.clip(ratio**power,1,cap),dtype=torch.float32,device=device)

def supervised_loss(logits,labels,pw,cfg):
    loss=nn.functional.binary_cross_entropy_with_logits(logits,labels,pos_weight=pw,reduction='none')
    mention=float(cfg.get('mention_loss_weight',1))
    if mention<=0:raise ValueError('mention_loss_weight must be positive')
    weights=torch.ones(21,device=logits.device);weights[::3]=mention
    # With default mention weight 1 this is the exact original mean BCE.
    return (loss*weights).sum()/weights.sum()/len(labels)

def symmetric_bernoulli_kl(first,second):
    a,b=first.float(),second.float()
    polarity=[i for i in range(21) if i%3!=0]
    # Half the sum of KL(Bernoulli(p)||Bernoulli(q)) and its reverse.
    return (.5*(torch.sigmoid(a)-torch.sigmoid(b))*(a-b))[:,polarity].mean()

def training_objective(model,inputs,counts,labels,pw,cfg):
    first=model(inputs,counts);loss=supervised_loss(first,labels,pw,cfg)
    strength=float(cfg.get('rdrop_alpha',0))
    if strength<0:raise ValueError('rdrop_alpha must be nonnegative')
    if strength:
        second=model(inputs,counts)
        loss=.5*(loss+supervised_loss(second,labels,pw,cfg))+strength*symmetric_bernoulli_kl(first,second)
    return loss,first
