"""Воспроизводимый запуск предсказания с включёнными весами E5."""
import argparse,subprocess,sys
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('upload'))
    p.add_argument('--work',type=Path,default=Path('work_v3'))
    p.add_argument('--output',type=Path,default=Path('answer_v3.csv'))
    a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parent
    for mode in ['items','descriptions','queries']:
        cmd=[sys.executable,'-u',str(root/'dense.py'),'--mode',mode,
             '--data',str(a.data),'--work',str(a.work),
             '--model-path',str(a.work/'e5_model')]
        print('RUN',cmd,flush=True);subprocess.run(cmd,check=True)
    cmd=[sys.executable,'-u',str(root/'predict_v3.py'),'--data',str(a.data),
         '--work',str(a.work),'--output',str(a.output)]
    print('RUN',cmd,flush=True);subprocess.run(cmd,check=True)
