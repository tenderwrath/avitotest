"""Финальное обучение в отдельном процессе, чтобы освободить память проверки."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier,Pool
from retrieval_v1 import SEED
from retrieval_v3 import FEATURES
from weights import demographic_weights

if __name__=='__main__':
    work=Path('work_v3');r=json.loads(Path('work_v2/records.json').read_text())
    d=np.load(work/'features.npz');x=d['x'];y=d['y'];n=np.diff(d['bounds'])
    metrics=json.loads((work/'metrics_v3.json').read_text());trees=metrics['trees']
    _,weights=demographic_weights(r,Path('upload'),'all')
    model=CatBoostClassifier(iterations=trees,depth=7,learning_rate=.055,
        loss_function='Logloss',random_seed=SEED,thread_count=6,l2_leaf_reg=8,verbose=100)
    model.fit(Pool(x,y,weight=np.repeat(weights,n)))
    model.save_model(str(work/'classifier.cbm'))
    (work/'selection.json').write_text(json.dumps({'trees':trees,'features':FEATURES},indent=2))
