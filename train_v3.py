"""Сравниваем CatBoost с предыдущей версией на одной и той же разметке train.

Финальное обучение запускается отдельно в refit_v3.py, чтобы освободить RAM.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier,Pool
from retrieval_v1 import SEED
from retrieval_v3 import FEATURES
from weights import demographic_weights
from retrieval_v1 import topk

def query_metric(scores,y,bounds,records,mask=None):
    values=[]
    for k,r in enumerate(records):
        a,b=bounds[k:k+2];keep=np.arange(b-a) if mask is None else np.flatnonzero(mask[a:b])
        chosen=keep[topk(scores[a:b][keep],50)]
        values.append(float(y[a:b][chosen].sum()/len(r['positives'])))
    return np.array(values)

def target_average(values,records,part):
    total=0.;den=0.
    for seen,ws in [(False,.63),(True,.37)]:
        for filtered,wf in [(False,.631),(True,.369)]:
            ids=[i for i,r in enumerate(records) if r['split']==part and r['history_seen']==seen and bool(r['search_infm_params_text'])==filtered]
            if ids:total+=ws*wf*float(values[ids].mean());den+=ws*wf
    return total/max(den,1e-9)

if __name__=='__main__':
    work=Path('work_v3');oldwork=Path('work_v2')
    records=json.loads((oldwork/'records.json').read_text());d=np.load(work/'features.npz')
    x=d['x'];y=d['y'];old=d['old'];bounds=d['bounds']
    parts=np.array([r['split'] for r in records]);n=np.diff(bounds)
    fit=np.repeat(parts=='fit',n);dev=np.repeat(parts=='dev',n)
    fit_ids,fit_w=demographic_weights(records,Path('upload'),'fit')
    dev_ids,dev_w=demographic_weights(records,Path('upload'),'dev')
    all_ids,all_w=demographic_weights(records,Path('upload'),'all')
    record_w=np.ones(len(records),dtype='float32');record_w[fit_ids]=fit_w
    weights=np.repeat(record_w,n)
    assert x.shape[1]==len(FEATURES)
    print('Rows',len(x),'Features',x.shape[1],flush=True)
    baseline=CatBoostClassifier();baseline.load_model(str(oldwork/'classifier_validation.cbm'))
    baseline_raw=baseline.predict(x[:,:42],prediction_type='RawFormulaVal')
    base=query_metric(baseline_raw,y,bounds,records,old)
    print('V2 dev',base[parts=='dev'].mean(),'test',base[parts=='test'].mean(),flush=True)
    model=CatBoostClassifier(iterations=900,depth=7,learning_rate=.055,
        loss_function='Logloss',random_seed=SEED,thread_count=6,l2_leaf_reg=8,verbose=100)
    model.fit(Pool(x[fit],y[fit],weight=weights[fit]),eval_set=(x[dev],y[dev]),early_stopping_rounds=100)
    best=-1;best_trees=model.tree_count_;scores=np.zeros(len(y),dtype='float32')
    for step,pred in enumerate(model.staged_predict(x[dev],eval_period=50,prediction_type='RawFormulaVal'),1):
        trees=min(step*50,model.tree_count_);scores[dev]=pred
        val=query_metric(scores,y,bounds,records);objective=float(np.average(val[dev_ids],weights=dev_w))
        print('trees',trees,'weighted dev',objective,flush=True)
        if objective>best+1e-10:best=objective;best_trees=trees
    model.shrink(best_trees);model.save_model(str(work/'classifier_validation.cbm'))
    pred=model.predict(x,prediction_type='RawFormulaVal')
    vals=query_metric(pred,y,bounds,records)
    delta=vals[parts=='test']-base[parts=='test'];rng=np.random.default_rng(SEED)
    boots=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)]
    oracle=np.array([y[a:b].sum()/len(r['positives']) for r,a,b in zip(records,bounds[:-1],bounds[1:])])
    test_ids,test_w=demographic_weights(records,Path('upload'),'test')
    metrics={'trees':best_trees,'dev':float(vals[parts=='dev'].mean()),
             'dev_target_weighted':float(np.average(vals[dev_ids],weights=dev_w)),
             'test':float(vals[parts=='test'].mean()),
             'test_target_weighted':float(np.average(vals[test_ids],weights=test_w)),
             'v2_dev':float(base[parts=='dev'].mean()),'v2_test':float(base[parts=='test'].mean()),
             'v2_dev_target_weighted':float(np.average(base[dev_ids],weights=dev_w)),
             'v2_test_target_weighted':float(np.average(base[test_ids],weights=test_w)),
             'v3_candidate_oracle_test':float(oracle[parts=='test'].mean()),
             'test_delta_ci95':np.quantile(boots,[.025,.975]).tolist()}
    print(json.dumps(metrics,indent=2),flush=True)
    (work/'metrics_v3.json').write_text(json.dumps(metrics,indent=2))
