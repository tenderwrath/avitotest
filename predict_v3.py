"""Итоговый CSV после обучения: полный train используется только как история."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='4'
import argparse,pickle,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from retrieval_v1 import load_data,topk,clean
from retrieval_v3 import Retriever,FEATURES
from validate_answer import validate_answer

def predict(data,work,dense_work,index_work,output):
    items,history,queries=load_data(data)
    index=index_work/'index.pkl'
    if index.exists():
        with index.open('rb') as f:ret=pickle.load(f)
        assert np.array_equal(ret.ids,items.item_id.to_numpy())
        ret.__class__=Retriever
        ret.build_history(history)
    else:
        ret=Retriever(items,history)
    ret.attach(dense_work,work)
    model=CatBoostClassifier();model.load_model(str(work/'classifier.cbm'))
    assert len(model.feature_names_)==len(FEATURES)
    baseline=None
    if (work/'classifier_v2.cbm').exists():
        baseline=CatBoostClassifier();baseline.load_model(str(work/'classifier_v2.cbm'))
    output.parent.mkdir(parents=True,exist_ok=True)
    started=time.time();answers=[]
    def one(row):
        candidates,features,_,_,old=ret.retrieve(row)
        # Для уже знакомого текста доступна история пользовательских выборов.
        # Базовая модель на прежнем множестве кандидатов сохраняет этот сигнал;
        # для нового текста используем расширенный поиск по описаниям.
        if baseline is not None and clean(row['search_query']) in ret.history_query_map:
            candidates=candidates[old];features=features[old,:42]
            scores=baseline.predict(features,prediction_type='RawFormulaVal',thread_count=1)
        else:
            scores=model.predict(features,prediction_type='RawFormulaVal',thread_count=1)
        return ' '.join(ret.ids[candidates[topk(scores,50)]])
    with ThreadPoolExecutor(max_workers=3) as pool:
        for k,answer in enumerate(pool.map(one,queries.to_dict('records')),1):
            answers.append(answer)
            if k%100==0:print('Predicted',k,'seconds',round(time.time()-started),flush=True)
    pd.DataFrame({'query_id':queries.query_id,'answer':answers}).to_csv(output,index=False)
    validate_answer(output,queries,items)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('upload'))
    p.add_argument('--work',type=Path,default=Path('work_v3'))
    p.add_argument('--dense-work',type=Path,default=None)
    p.add_argument('--index-work',type=Path,default=None)
    p.add_argument('--output',type=Path,default=Path('answer_v3.csv'))
    a=p.parse_args();predict(a.data,a.work,a.dense_work or a.work,a.index_work or a.work,a.output)
