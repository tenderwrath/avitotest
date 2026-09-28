"""Сопоставление структуры обучающих ситуаций и неразмеченных запросов.

Используются только признаки запросов benchmark, без item_id и без разметки.
"""
from collections import Counter
import numpy as np
import pandas as pd
from retrieval_v1 import clean

def key(row,seen=None):
    if seen is None:seen=row['history_seen']
    return (bool(seen),min(len(row['search_query'].split()),5),
            bool(row['search_infm_params_text']))

def demographic_weights(records,data,part,lo=.25,hi=4.):
    train=pd.read_parquet(data/'train.parquet',columns=['search_query'])
    seen_texts=set(train.search_query.map(clean))
    bench=pd.read_parquet(data/'benchmark_queries.parquet')
    target=Counter(key(z,clean(z['search_query']) in seen_texts) for z in bench.to_dict('records'))
    ids=list(range(len(records))) if part=='all' else [i for i,r in enumerate(records) if r['split']==part]
    source=Counter(key(records[i]) for i in ids)
    weights=np.array([np.clip((target[key(records[i])]+.5)/(len(bench)+.5*len(source)) /
                             ((source[key(records[i])]+.5)/(len(ids)+.5*len(source))),lo,hi)
                      for i in ids],dtype='float32')
    return np.array(ids),weights

def weighted_recall(vals,records,data,part):
    ids,weights=demographic_weights(records,data,part)
    return float(np.average(vals[ids],weights=weights))
