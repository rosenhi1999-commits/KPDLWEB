"""Create and execute an aggregate TRAIN-ONLY notebook with private error export."""
from pathlib import Path
import argparse,os,sys
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    nb=nbf.v4.new_notebook();nb.metadata.kernelspec={'name':'python3','display_name':'Python 3','language':'python'}
    nb.cells=[nbf.v4.new_markdown_cell('''# ABSA — chỉ dùng train.csv

Không đọc hoặc dự đoán blind; không tạo file nộp. Đây là notebook nghiên cứu validation, chưa là bộ nộp hoàn chỉnh. Mỗi phương pháp có thư mục riêng. Chỉ so sánh trực tiếp các lượt cùng split. Validation đã được xem trong nhiều lượt phát triển; không coi đây là test độc lập.

PhoBERT trong các lượt mới là encoder đóng băng int8, học classifier/attention head. Chưa fine-tune toàn bộ encoder. Cell stacking chạy lại meta từ đầu ra nguồn đã lưu, không huấn luyện lại encoder.
'''),nbf.v4.new_code_cell('''import train_only_guard
import json,time
from pathlib import Path
import numpy as np,pandas as pd,joblib
from scipy import sparse
from absa_core import LABELS,reference_splits,prepare_texts,binary_predictions
from aspect_scope import scoped_texts
from score_absa import report,show
ROOT=Path.cwd()
train=pd.read_csv(ROOT/'train.csv')
assert list(train.columns)==['id','text']+LABELS
fit,tune,valid=reference_splits(train)
print('Train:',len(train),'fit:',len(fit),'tune:',len(tune),'validation:',len(valid))
print('Blind CSV audit guard active.')
'''),nbf.v4.new_code_cell('''rows=[]
for path in sorted(ROOT.glob('*/metrics.json')):
    m=json.loads(path.read_text())
    policy=m.get('split_policy','legacy_random42')
    if path.parent.name=='Ensemble_lan2':policy='reference_group_2026'
    rows.append({'experiment':path.parent.name,'split':policy,'ABSA_F1':m['ABSA_F1'],
        'CI95':m.get('ABSA_F1_CI95',None),'accuracy':m.get('accuracy_TB',m.get('accuracy_mean_per_label'))})
table=pd.DataFrame(rows)
print('Completed measured runs:',len(table))
display(table[table.split=='reference_group_2026'].sort_values('ABSA_F1',ascending=False))
print('Target 0.96 reached:',bool((table.ABSA_F1>=.96).any()))
'''),nbf.v4.new_markdown_cell('''## Tái chạy toàn bộ suy luận của mô hình theo mệnh đề

Mô hình validation chỉ học outer80. Tách ngưỡng trên tune trước khi refit. Các lexical scope chọn đầu vào; nhãn cuối do mô hình học từ train, không phải quy tắc gán nhãn.
'''),nbf.v4.new_code_cell('''folder=ROOT/'ScopedTFIDF_lan4'
bundle=joblib.load(folder/'validation_model.joblib')
texts=np.asarray(prepare_texts(train.text.iloc[valid],'lexical'),dtype=object)
scopes=scoped_texts(train.text.iloc[valid])
started=time.perf_counter();G=bundle['global_vectorizer'].transform(texts);parts=[]
for aspect,vec,model in zip(['nhan_vien','sach_se','vi_tri','phong','gia_tri','an_uong','tien_nghi'],bundle['local_vectorizers'],bundle['models']):
    L=vec.transform(scopes[aspect])
    parts.append(model.decision_function(sparse.hstack([G,L*bundle['config']['scope_weight']],format='csr')))
scores=np.hstack(parts);pred=binary_predictions(scores,bundle['thresholds'])
gt=train.iloc[valid][['id']+LABELS].reset_index(drop=True)
sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
metrics=report(gt,sb,n_boot=1000);show(metrics)
assert abs(metrics['ABSA_F1']-json.loads((folder/'metrics.json').read_text())['ABSA_F1'])<1e-12
print('Validation model inference seconds:',round(time.perf_counter()-started,4))
'''),nbf.v4.new_markdown_cell('''## Tái tính stacking từ đầu ra encoder/classifier đã lưu

Không coi thời gian cell này là thời gian huấn luyện hay chạy lại encoder. Base neural heads đã chọn epoch trên tune; meta CV không phải nested encoder CV hoàn toàn. Validation vẫn ngoài mọi bước fit/chọn ngưỡng của mô hình này.
'''),nbf.v4.new_code_cell('''folder=ROOT/'Stacking_lan1'
bundle=joblib.load(folder/'meta_models.joblib')
sources=[np.load(ROOT/name/'private_validation_scores.npy') for name in bundle['sources']]
scores=[]
for j,(model,cfg) in enumerate(zip(bundle['models'],bundle['labels'])):
    idx=list(range(21)) if cfg['mode']=='all' else list(range((j//3)*3,(j//3+1)*3))
    scores.append(model.decision_function(np.concatenate([s[:,idx] for s in sources],axis=1)))
scores=np.column_stack(scores);pred=binary_predictions(scores,np.asarray(bundle['thresholds']))
sb=pd.DataFrame(pred,columns=LABELS);sb.insert(0,'id',gt.id)
metrics=report(gt,sb,n_boot=1000);show(metrics)
assert abs(metrics['ABSA_F1']-json.loads((folder/'metrics.json').read_text())['ABSA_F1'])<1e-12
print('Target 0.96 reached:',metrics['ABSA_F1']>=.96)
'''),nbf.v4.new_code_cell('''# Chỉ lưu lỗi từ validation của TRAIN; không đưa văn bản lên output công khai.
errors=[]
truth=gt[LABELS].to_numpy(int)
for k in np.flatnonzero((truth!=pred).any(axis=1))[:5]:
    fp=[LABELS[j] for j in np.flatnonzero((pred[k]==1)&(truth[k]==0))]
    fn=[LABELS[j] for j in np.flatnonzero((pred[k]==0)&(truth[k]==1))]
    errors.append({'id':str(gt.id.iloc[k]),'text':str(train.text.iloc[valid[k]]),
        'false_positive':fp,'false_negative':fn,
        'review_note':'Đối chiếu phạm vi khía cạnh và phủ định/gợi ý cải thiện; phân biệt nhắc với cảm xúc. Đây là phân tích lỗi train, không sửa nhãn hoặc dự đoán blind.'})
(folder/'private_error_analysis.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
print('Private train-validation error examples saved:',len(errors))
'''),nbf.v4.new_code_cell('''provenance=ROOT/'augmentation_provenance.json'
if provenance.exists():
    p=json.loads(provenance.read_text())
    print('Synthetic training rows:',p['rows'])
    print('Tool:',p['tool'])
    print('User prompt:',p['user_prompt'])
    print('Construction specification:',p['construction_spec'])
    print('SHA256:',p['sha256'])
    print('If selecting an augmented model, include du_lieu_them.csv and provenance in final zip.')
    if (ROOT/'augmentation_comparison.csv').exists():display(pd.read_csv(ROOT/'augmentation_comparison.csv'))
print('TRAIN-ONLY: no submission.csv generated, no blind CSV opened.')
''')]
    if a.execute:
        os.environ.setdefault('JUPYTER_PATH',str(Path(sys.prefix)/'share/jupyter'))
        os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(ROOT/'private_jupyter_runtime'))
        os.environ.setdefault('IPYTHONDIR',str(ROOT/'private_ipython'))
        NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(nb,ROOT/'submission.ipynb');print('Written submission.ipynb; executed:',a.execute)
