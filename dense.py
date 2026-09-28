"""Локальные эмбеддинги E5. Загрузка весов нужна только при подготовке окружения."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import argparse,json,time,re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_ID='intfloat/multilingual-e5-small'
def encode(texts,model_path,batch_size=128,max_length=64):
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModel.from_pretrained(model_path,local_files_only=True).eval()
    # CPU dynamic quantization: веса линейных слоёв int8, выходные векторы float32.
    model=torch.ao.quantization.quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8)
    out=[];started=time.time()
    with torch.inference_mode():
        for start in range(0,len(texts),batch_size):
            batch=tokenizer(texts[start:start+batch_size],max_length=max_length,padding=True,truncation=True,return_tensors='pt')
            z=model(**batch).last_hidden_state
            mask=batch['attention_mask'].unsqueeze(-1)
            z=(z*mask).sum(1)/mask.sum(1).clamp(min=1)
            z=torch.nn.functional.normalize(z,p=2,dim=1)
            out.append(z.numpy())
            if start==0 or (start//batch_size+1)%50==0:
                print('Encoded',min(start+batch_size,len(texts)),'/',len(texts),'seconds',round(time.time()-started,1),flush=True)
    return np.concatenate(out).astype('float32')

def title_text(series):
    return series.fillna('').astype(str).str.lower().str.replace('ё','е',regex=False).str.replace(r'\s+',' ',regex=True).str.strip()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['download','items','descriptions','queries'],required=True)
    p.add_argument('--data',type=Path,default=Path('upload'));p.add_argument('--work',type=Path,default=Path('work_v2'))
    p.add_argument('--model-path',type=Path,default=Path('work_v2/e5_model'));a=p.parse_args();a.work.mkdir(exist_ok=True,parents=True)
    if a.mode=='download':
        revision='614241f622f53c4eeff9890bdc4f31cfecc418b3';a.model_path.mkdir(exist_ok=True,parents=True)
        tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=revision);mod=AutoModel.from_pretrained(MODEL_ID,revision=revision)
        tok.save_pretrained(a.model_path);mod.save_pretrained(a.model_path)
        (a.model_path/'source.json').write_text(json.dumps({'model_id':MODEL_ID,'revision':revision},indent=2))
        print('Model downloaded',revision,flush=True)
    elif a.mode=='items':
        d=pd.read_parquet(a.data/'benchmark_items.parquet',columns=['item_id','item_title_raw'])
        titles=title_text(d.item_title_raw);codes,unique=pd.factorize(titles,sort=True)
        vec=encode(['passage: '+s for s in unique],a.model_path)
        np.save(a.work/'item_dense.npy',vec[codes]);np.save(a.work/'dense_item_ids.npy',d.item_id.to_numpy())
        print('Saved item vectors',flush=True)
    elif a.mode=='descriptions':
        d=pd.read_parquet(a.data/'benchmark_items.parquet',columns=['item_title_raw','item_description_raw'])
        # Заголовок удерживает основную тему, первые слова описания уточняют услугу.
        titles=title_text(d.item_title_raw)
        descriptions=title_text(d.item_description_raw).str.slice(0,350)
        document=titles+'; '+descriptions
        codes,unique=pd.factorize(document,sort=True)
        vec=encode(['passage: '+s for s in unique],a.model_path,batch_size=128,max_length=64)
        np.save(a.work/'item_description_dense.npy',vec[codes])
        print('Saved description vectors',flush=True)
    else:
        records=json.loads((a.work/'records.json').read_text()) if (a.work/'records.json').exists() else []
        q=pd.read_parquet(a.data/'benchmark_queries.parquet')
        texts=sorted(set([r['search_query'] for r in records]+q.search_query.tolist()))
        vec=encode(['query: '+s for s in texts],a.model_path,batch_size=128)
        np.save(a.work/'query_dense.npy',vec);(a.work/'dense_query_texts.json').write_text(json.dumps(texts,ensure_ascii=False))
