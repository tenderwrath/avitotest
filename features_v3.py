"""Создаём дополнительные семантические кандидаты; в dev/test позитивы не доклеиваются."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='4'
import json,pickle,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from retrieval_v3 import Retriever

if __name__=='__main__':
    old=Path('work_v2');work=Path('work_v3');work.mkdir(exist_ok=True)
    records=json.loads((old/'records.json').read_text())
    with (old/'index.pkl').open('rb') as f:ret=pickle.load(f)
    # Индекс BM25 и обучающая история совпадают с версией 2.
    ret.__class__=Retriever
    ret.attach(old,work)
    xs=[];ys=[];cs=[];oldmask=[];bounds=[0];oldrec=[];newrec=[];start=time.time()
    with ThreadPoolExecutor(max_workers=3) as pool:
        for k,(row,(c,x,_,_,m)) in enumerate(zip(records,pool.map(ret.retrieve,records))):
            pos=set(row['positives']);y=np.array([ret.ids[j] in pos for j in c],dtype='uint8')
            xs.append(x);ys.append(y);cs.append(c);oldmask.append(m);bounds.append(bounds[-1]+len(c))
            oldrec.append(y[m].sum()/len(pos));newrec.append(y.sum()/len(pos))
            if (k+1)%100==0:
                print('Features',k+1,'v2 pool',round(float(np.mean(oldrec)),4),
                      'v3 pool',round(float(np.mean(newrec)),4),
                      'seconds',round(time.time()-start),flush=True)
    np.savez_compressed(work/'features.npz',x=np.concatenate(xs),y=np.concatenate(ys),
                        c=np.concatenate(cs),old=np.concatenate(oldmask),bounds=np.array(bounds))
    print('Saved features',flush=True)
