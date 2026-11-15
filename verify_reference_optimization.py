"""Regression checks for reference-equivalent defaults and tune-ID isolation."""
from pathlib import Path
import tempfile,json
import numpy as np,pandas as pd,torch
from torch import nn
from reference_optimization import supervised_loss,positive_weights,optimizer_for,symmetric_bernoulli_kl,training_objective
from calibrate_reference import load_tune,fit_thresholds
from absa_core import LABELS,reference_splits,binary_predictions
from neural_train import ReviewBatchSampler

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    torch.manual_seed(2026);y=np.array([[0,1,0]*7,[1,0,1]*7,[1,1,0]*7])
    pw=positive_weights(y,{},'cpu')
    original=torch.as_tensor(np.clip(np.sqrt((len(y)-y.sum(0))/np.maximum(1,y.sum(0))),1,3),dtype=torch.float32)
    assert torch.equal(pw,original)
    logits=torch.randn(3,21,requires_grad=True);labels=torch.tensor(y,dtype=torch.float32)
    old=nn.BCEWithLogitsLoss(pos_weight=original)(logits,labels)
    new=supervised_loss(logits,labels,pw,{})
    assert torch.allclose(old,new,atol=1e-7)
    a=torch.autograd.grad(old,logits,retain_graph=True)[0];b=torch.autograd.grad(new,logits)[0]
    assert torch.allclose(a,b,atol=1e-7)
    first=torch.randn(3,21,requires_grad=True);second=torch.randn(3,21,requires_grad=True)
    penalty=symmetric_bernoulli_kl(first,second)
    assert penalty>=0 and torch.allclose(penalty,symmetric_bernoulli_kl(second,first))
    assert symmetric_bernoulli_kl(first,first)==0
    mention_only=first.detach().clone();mention_only[:,::3]+=2
    assert symmetric_bernoulli_kl(first,mention_only)==0
    penalty.backward();assert torch.isfinite(first.grad).all() and torch.isfinite(second.grad).all()
    calls=[0];fixed=torch.randn(3,21,requires_grad=True)
    def forward(inputs,counts):calls[0]+=1;return fixed
    loss,_=training_objective(forward,{},[],labels,pw,{})
    assert calls[0]==1 and torch.allclose(loss,supervised_loss(fixed,labels,pw,{}))
    calls[0]=0;loss,_=training_objective(forward,{},[],labels,pw,{'rdrop_alpha':.5})
    assert calls[0]==2 and torch.isfinite(loss)
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.encoder=nn.Module();self.encoder.encoder=nn.Module()
            self.encoder.encoder.layer=nn.ModuleList([nn.Linear(2,2),nn.Linear(2,2)])
            self.encoder.embeddings=nn.Linear(2,2);self.head=nn.Linear(2,21)
    model=Tiny();opt=optimizer_for(model,{'lr':2e-5,'head_lr_multiplier':2,'layer_lr_decay':.9,'exclude_bias_norm_decay':True})
    group={id(p):(g['lr'],g['weight_decay']) for g in opt.param_groups for p in g['params']}
    assert np.isclose(group[id(model.head.weight)][0],4e-5)
    assert np.isclose(group[id(model.encoder.encoder.layer[-1].weight)][0],2e-5)
    assert group[id(model.head.bias)][1]==0
    assert group[id(model.encoder.embeddings.weight)][0]<group[id(model.encoder.encoder.layer[-1].weight)][0]
    class TinyReviews:
        def __init__(self):self.items=[[{'input_ids':list(range(i+2))}] for i in range(13)]
        def __len__(self):return len(self.items)
    ds=TinyReviews()
    batches=list(ReviewBatchSampler(ds,4,torch.Generator().manual_seed(2026)))
    assert sorted(i for batch in batches for i in batch)==list(range(13))
    train=pd.read_csv(ROOT/'train.csv');_,tune,valid=reference_splits(train)
    with tempfile.TemporaryDirectory(dir=ROOT,prefix='private_calibration_check_') as temp:
        temp=Path(temp)
        probabilities=np.tile(np.linspace(.1,.9,len(tune))[:,None],(1,21))
        np.savez(temp/'tune.npz',ids=train.id.iloc[tune].to_numpy(dtype=str)[::-1],scores=probabilities[::-1])
        yy,p=load_tune(temp/'tune.npz',train)
        assert np.array_equal(yy,train[LABELS].iloc[tune].to_numpy(int)) and np.allclose(p,probabilities)
        np.savez(temp/'heldout.npz',ids=train.id.iloc[valid].to_numpy(dtype=str),scores=probabilities)
        try:load_tune(temp/'heldout.npz',train)
        except ValueError:pass
        else:raise AssertionError('Validation IDs accepted for tuning')
    truth=np.tile(np.array([0,1]*10)[:,None],(1,21));scores=.1+.8*truth
    for method in ['fixed','reference_grid','exact_pr','shrink25','shrink75']:
        thresholds=fit_thresholds(truth,scores,method)
        assert np.array_equal(binary_predictions(scores,thresholds),truth)
    result={'checks_passed':['default_loss_and_gradients_match_reference','default_positive_weight_matches_reference',
        'optimizer_lr_and_no_decay_groups','bucket_sampler_visits_every_review_once',
        'tune_score_id_alignment','validation_scores_rejected_even_with_same_row_count','threshold_clean_separation',
        'rdrop_symmetric_nonnegative_zero_for_identical_predictions','rdrop_excludes_auxiliary_mentions','rdrop_disabled_one_forward_enabled_two'],
        'full_finetuning_completed':False,'new_F1_claimed':False,'blind_file_read':False}
    (ROOT/'reference_optimization_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
