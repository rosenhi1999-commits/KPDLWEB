"""Prepare a new private replay directory without copying published run folders."""
from pathlib import Path
import argparse,shutil,json,hashlib
import train_only_guard

SOURCE=Path(__file__).resolve().parent
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=Path,default=SOURCE/'train.csv')
    ap.add_argument('--destination',type=Path,default=SOURCE/'private_train_only_replay');a=ap.parse_args()
    if a.train.name!='train.csv':raise ValueError('Only train.csv may be copied')
    if not a.train.is_file():raise FileNotFoundError('Put train.csv on the local machine first')
    a.destination.mkdir(parents=True,exist_ok=False)
    for p in list(SOURCE.glob('*.py'))+list(SOURCE.glob('requirements*.txt')):
        shutil.copy2(p,a.destination/p.name)
    shutil.copy2(a.train,a.destination/'train.csv')
    if (SOURCE/'submission_demo.csv').exists():shutil.copy2(SOURCE/'submission_demo.csv',a.destination/'submission_demo.csv')
    for name in ['TFIDF_lan1','TFIDF_lan2']:
        if (SOURCE/name/'train.py').exists():
            (a.destination/name).mkdir();shutil.copy2(SOURCE/name/'train.py',a.destination/name/'train.py')
    info={'train_sha256':hashlib.sha256(a.train.read_bytes()).hexdigest(),'blind_files_copied':False,
        'note':'Only code, requirements, train.csv and optional submission template were copied.'}
    (a.destination/'private_workspace.json').write_text(json.dumps(info,indent=2))
    print(str(a.destination.resolve()))
