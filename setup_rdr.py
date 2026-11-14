"""Download and verify official VnCoreNLP word-segmentation resources only."""
from pathlib import Path
import hashlib,urllib.request

ROOT=Path(__file__).resolve().parent/'private_vncorenlp'
FILES={
'VnCoreNLP-1.2.jar':'9e2811cdbc2ddfc71d04be5dc36e185c88dcd1ad4d5d69e4ff2e1369dccf7793',
'models/wordsegmenter/wordsegmenter.rdr':'9e62f96bd93e37a24f364238e8d8ae986fa5dad6dbc9f4eae622ab3651b7fa06',
'models/wordsegmenter/vi-vocab':'0a47c5b55bbce163029d37730a67b9479740388695c29c106c112b815613eaa5'}

if __name__=='__main__':
    for relative,expected in FILES.items():
        p=ROOT/relative;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            urllib.request.urlretrieve('https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/'+relative,p)
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('Resource changed; expected pinned SHA256 for '+relative)
        print('Verified',relative,actual)
