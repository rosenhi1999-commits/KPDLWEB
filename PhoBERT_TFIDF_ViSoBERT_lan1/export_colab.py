"""Build an executed, reproducible submission folder; select by public score.

No portal login, OTP handling, or remote submission is performed.
Original train/test CSVs are read locally and never included in the ZIP.
"""
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen
import zipfile

import nbformat
from nbclient import NotebookClient

BASE_COMMIT = '8b68d0fc864b23442064940a7e13d84d1a4664a3'
REPO = 'rosenhi1999-commits/KPDLWEB'


class LiveNotebookClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg.get('msg_type') == 'stream':
            print(msg['content']['text'], end='', flush=True)
        return super().process_message(msg, cell, cell_index)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def hardware():
    import torch
    cpu = platform.processor() or 'not recorded'
    ram = None
    if Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.partition(':')[2].strip(); break
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemTotal:'):
                ram = int(line.split()[1]) * 1024; break
    return {'CPU': cpu, 'logical_CPU_count': os.cpu_count(), 'RAM_bytes_host': ram,
        'OS': platform.platform(), 'Python': platform.python_version(),
        'PyTorch': torch.__version__, 'CUDA': torch.version.cuda,
        'GPU': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'VRAM_bytes': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0}


def extract_selected(archive, target):
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        checkpoints = [n for n in z.namelist() if Path(n).name == 'model.fp16.pt']
        assert len(checkpoints) == 1, 'Chọn đúng ZIP absa-selected-encoder'
        prefix = str(Path(checkpoints[0]).parent)
        prefix = '' if prefix == '.' else prefix + '/'
        for name in ['model.fp16.pt', 'config.json', 'README.txt', 'run_info.json',
                     'ci_provenance.json', 'metrics_after_storage_rounding.json']:
            member = prefix + name
            if member in z.namelist():
                with z.open(member) as source, (target / name).open('wb') as dest:
                    shutil.copyfileobj(source, dest)
    assert (target / 'config.json').exists() and (target / 'model.fp16.pt').exists()


def copy_model(source, target):
    target.mkdir(parents=True)
    for name in ['config.json', 'recipe.json', 'selection.json', 'history.csv', 'metrics.json',
                 'run_info.json', 'training_hardware.json', 'ci_provenance.json',
                 'model.pt', 'model.fp16.pt', 'model.joblib', 'tune_comparison.csv',
                 'tune_scores.npz', 'validation_scores.npz']:
        p = source / name
        if p.exists():
            shutil.copy2(p, target / name)
    if (source / 'tokenizer').exists():
        shutil.copytree(source / 'tokenizer', target / 'tokenizer')


def write_requirements(folder):
    import torch
    packages = ['torch', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib',
        'transformers', 'tokenizers', 'huggingface-hub', 'safetensors', 'sentencepiece',
        'protobuf', 'pyvi', 'sklearn-crfsuite', 'python-crfsuite',
        'iterative-stratification', 'nbformat', 'nbclient', 'ipykernel']
    lines = [f'# Python {platform.python_version()}']
    if '+' in torch.__version__:
        channel = 'cu' + torch.version.cuda.replace('.', '') if torch.version.cuda else 'cpu'
        lines.append('--extra-index-url https://download.pytorch.org/whl/' + channel)
    for name in packages:
        lines.append(name + '==' + importlib.metadata.version(name))
    (folder / 'requirements.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    frozen = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (folder / 'environment_full.txt').write_text(frozen, encoding='utf-8')


def make_notebook():
    nbf = nbformat.v4
    cells = [nbf.new_markdown_cell('''# KPW2.3 — tái hiện checkpoint và tạo submission

Notebook này thực sự chạy validation và inference từ trọng số đính kèm.
Test chỉ dùng để dự đoán; không dùng để huấn luyện, chọn ngưỡng hay chọn trọng số.
Điểm validation dưới đây khác điểm công khai của trang chấm.
Không tạo thêm dữ liệu hoặc nhãn bằng LLM. Hạt giống: 2026.
'''), nbf.new_code_cell('''import os, sys, json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from IPython.display import display
ROOT = Path.cwd()
DATA = Path(os.environ.get("ABSA_DATA_DIR", str(ROOT / "data")))
sys.path.insert(0, str(ROOT / "code"))
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
random.seed(2026)
np.random.seed(2026)
torch.manual_seed(2026)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2026)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_num_threads(2)
from predict_bundle import Predictor, LABELS, write_submission, file_hash
from absa_core import reference_splits
import score_absa
bundle = json.loads((ROOT / "BUNDLE_CONFIG.json").read_text())
print("Python:", sys.version)
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
print("Nguồn model:", bundle["model_dir"])
engine = Predictor(ROOT / bundle["model_dir"], batch_size=8)
'''), nbf.new_code_cell('''# Đo lại trên validation tách từ train; không cập nhật model/ngưỡng.
train = pd.read_csv(DATA / "train.csv", dtype={"id": str})
assert list(train.columns) == ["id", "text"] + LABELS
assert train.id.is_unique and not train.isna().any().any()
assert train[LABELS].isin([0, 1]).all().all()
_, _, valid = reference_splits(train, 2026)
started = time.perf_counter()
probabilities = engine.probabilities(train.text.iloc[valid])
prediction = engine.frame(train.id.iloc[valid], probabilities)
truth = train.iloc[valid][["id"] + LABELS].reset_index(drop=True)
metrics = score_absa.report(truth, prediction, n_boot=1000)
metrics.update(validation_previously_seen=True, production_refit=False,
               checkpoint_replay_seconds=time.perf_counter()-started)
(ROOT / "validation_metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
score_absa.show(metrics)
'''), nbf.new_code_cell('''# Dự đoán test bằng checkpoint/ngưỡng cố định; không huấn luyện.
started = time.perf_counter()
submission = write_submission(
    engine, DATA / "test_blind.csv", ROOT / "code" / "submission_demo.csv",
    ROOT / "submission.csv",
)
seconds = time.perf_counter() - started
inference = {
    "rows": len(submission), "columns": list(submission.columns),
    "submission_sha256": file_hash(ROOT / "submission.csv"),
    "test_sha256": file_hash(DATA / "test_blind.csv"),
    "inference_seconds": seconds,
    "test_used_for_training_or_tuning": False,
    "model_files_sha256": {
        str(p.relative_to(ROOT)): file_hash(p)
        for p in (ROOT / "models").rglob("*")
        if p.is_file() and p.name in ["model.pt", "model.fp16.pt", "model.joblib", "config.json"]
    },
}
(ROOT / "inference_report.json").write_text(
    json.dumps(inference, ensure_ascii=False, indent=2), encoding="utf-8")
print("Thời gian inference:", round(seconds, 2), "giây")
display(submission.head())
'''), nbf.new_markdown_cell('''## Huấn luyện lại

Mã huấn luyện và cấu hình được giữ trong `code/` và `models/`.
Checkpoint GitHub được chọn trước khi có điểm công khai; không huấn luyện lại
trong lượt tái hiện này. Thời gian/máy của lượt gốc xem `training_provenance.json`.
Máy và thời gian inference hiện tại được ghi riêng.
Để chạy lượt ba mô hình mới: `python code/colab_runner.py --train data/train.csv --out runs`.
Mỗi cấu hình mới tạo thư mục riêng; không dùng điểm validation làm điểm công khai.
''')]
    return nbf.new_notebook(cells=cells, metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})


def create_zip(folder, team):
    output = folder / (team + '.zip')
    manifest = {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*')
                if p.is_file() and p != output and p.name != 'MANIFEST_SHA256.json'
                and '__pycache__' not in p.parts}
    dump(folder / 'MANIFEST_SHA256.json', manifest)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p != output and '__pycache__' not in p.parts:
                z.write(p, p.relative_to(folder))
    return output


def register_score(root, folder, score, team):
    folder = folder.resolve()
    csv_hash = sha(folder / 'submission.csv')
    record = {'folder': str(folder), 'submission_sha256': csv_hash,
              'public_score': score, 'score_entered_by_user': True}
    inference_path = folder / 'inference_report.json'
    if inference_path.exists():
        inference = json.loads(inference_path.read_text())
        assert inference['submission_sha256'] == csv_hash, 'CSV changed since inference'
        for name, expected in inference.get('model_files_sha256', {}).items():
            assert sha(folder / name) == expected, 'Model/config changed since inference'
        record['test_sha256'] = inference['test_sha256']
    dump(folder / 'PUBLIC_SCORE.json', record)
    config_path = folder / 'BUNDLE_CONFIG.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config.update(public_score=score,
            public_score_status='entered_by_user' if score is not None else 'not_entered',
            submission_sha256=csv_hash)
        dump(config_path, config)
    registry = root / 'public_score_registry.json'
    records = json.loads(registry.read_text()) if registry.exists() else []
    records = [r for r in records if r['folder'] != str(folder)] + [record]
    dump(registry, records)
    # Never substitute validation metrics for missing portal scores.
    eligible = [r for r in records if r['public_score'] is not None
                and r.get('test_sha256') == record.get('test_sha256')]
    if not eligible:
        return None
    selected = max(eligible, key=lambda r: r['public_score'])
    source = Path(selected['folder'])
    if sha(source / 'submission.csv') != selected['submission_sha256']:
        raise ValueError('CSV changed after its public score was recorded')
    source_zip = create_zip(source, team)
    chosen = root / 'CHON_NOP_GIANG_VIEN'
    chosen.mkdir(exist_ok=True)
    shutil.copy2(source_zip, chosen / (team + '.zip'))
    shutil.copy2(source / 'submission.csv', chosen / 'submission.csv')
    dump(chosen / 'BEST_PUBLIC_SCORE.json', selected)
    return chosen


def main():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group()
    source.add_argument('--archive', type=Path)
    source.add_argument('--three-model-result', type=Path,
                        help='latest_result.json produced by colab_runner.py')
    p.add_argument('--train', type=Path, default=Path('/content/train.csv'))
    p.add_argument('--test', type=Path, default=Path('/content/test_blind.csv'))
    p.add_argument('--team', default='KPW2.3')
    p.add_argument('--out', type=Path, default=Path('/content/absa_deliverables'))
    p.add_argument('--record-score', type=float)
    p.add_argument('--run-folder', type=Path)
    args = p.parse_args()
    team = args.team.removesuffix('.zip')
    assert re.fullmatch(r'[A-Za-z0-9_.-]+', team)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.record_score is not None:
        assert 0 <= args.record_score <= 1 and args.run_folder
        chosen = register_score(args.out, args.run_folder, args.record_score, team)
        print('THƯ MỤC ĐIỂM CÔNG KHAI CAO NHẤT:', chosen)
        return
    assert args.archive or args.three_model_result, 'Cần ZIP checkpoint hoặc kết quả ba mô hình'
    assert args.train.name == 'train.csv' and args.train.exists()
    assert args.test.name == 'test_blind.csv' and args.test.exists()
    existing = [d for d in args.out.glob(team + '_lan*')
                if d.name.removeprefix(team + '_lan').isdigit()]
    number = max([0] + [int(d.name.removeprefix(team + '_lan')) for d in existing]) + 1
    folder = args.out / f'{team}_lan{number}'
    (folder / 'code').mkdir(parents=True)
    here = Path(__file__).resolve().parent
    for name in ['predict_bundle.py', 'colab_runner.py', 'export_colab.py']:
        shutil.copy2(here / name, folder / 'code' / name)
    for name in ['score_absa.py', 'absa_core.py', 'neural_train.py',
                 'reference_optimization.py', 'train_only_guard.py', 'submission_demo.csv']:
        url = f'https://raw.githubusercontent.com/{REPO}/{BASE_COMMIT}/{name}'
        with urlopen(url, timeout=90) as response:
            (folder / 'code' / name).write_bytes(response.read())
    if args.archive:
        model_dir = folder / 'models' / 'selected_encoder'
        extract_selected(args.archive, model_dir)
        cfg = json.loads((model_dir / 'config.json').read_text())
        provenance = {'archive_sha256': sha(args.archive),
            'original_run_info': json.loads((model_dir / 'run_info.json').read_text()),
            'training_CPU_model': None, 'training_RAM_bytes': None,
            'missing_hardware_note': 'Old checkpoint archive did not record CPU model or RAM. Do not substitute inference hardware.'}
        ci_path = model_dir / 'ci_provenance.json'
        if ci_path.exists():
            ci = json.loads(ci_path.read_text())
            provenance['github_provenance'] = ci
            experiment = ci.get('experiment', '')
            if re.fullmatch(r'(PhoBERT|ViSoBERT)_lan\d+', experiment):
                for name in ['metrics.json', 'history.csv']:
                    try:
                        with urlopen(f'https://raw.githubusercontent.com/{REPO}/main/{experiment}/{name}', timeout=30) as r:
                            (model_dir / ('source_' + name)).write_bytes(r.read())
                    except Exception as exc:
                        provenance['metadata_download_note'] = str(exc)
        source_metrics = model_dir / 'source_metrics.json'
        if source_metrics.exists():
            m = json.loads(source_metrics.read_text())
            provenance['original_training'] = {k: m.get(k) for k in [
                'device', 'gpu', 'VRAM_bytes', 'python', 'torch', 'transformers',
                'train_seconds', 'epochs_completed', 'best_epoch']}
            provenance['original_training']['CPU_threads'] = cfg.get('cpu_threads')
    else:
        result = json.loads(args.three_model_result.read_text())
        assert not result.get('cached_neural_sources'), 'Cache benchmark has no complete encoder checkpoints'
        final = Path(result['ensemble'])
        for src in [final] + [Path(s) for s in result['sources']]:
            copy_model(src, folder / 'models' / src.name)
        model_dir = folder / 'models' / final.name
        provenance = {'source_result': result, 'models': {}}
        for src in map(Path, result['sources']):
            hp = src / 'training_hardware.json'
            provenance['models'][src.name] = json.loads(hp.read_text()) if hp.exists() else {'note': 'See model selection.json/run_info.json'}
    dump(folder / 'BUNDLE_CONFIG.json', {
        'team': team, 'model_dir': str(model_dir.relative_to(folder)),
        'public_score': None, 'public_score_status': 'not_entered',
        'data_files_not_included': ['train.csv', 'test_blind.csv'],
        'additional_synthetic_training_data_used': False,
    })
    dump(folder / 'training_provenance.json', provenance)
    dump(folder / 'inference_hardware.json', hardware())
    write_requirements(folder)
    notebook = make_notebook()
    # A new kernel avoids the training-only audit hook used by earlier notebooks.
    # Save partial real outputs even if execution fails; failed runs are not zipped.
    started = time.perf_counter()
    assert args.train.resolve().parent == args.test.resolve().parent
    env = dict(os.environ, ABSA_DATA_DIR=str(args.train.resolve().parent))
    client = LiveNotebookClient(notebook, timeout=None, kernel_name='python3',
                            resources={'metadata': {'path': str(folder.resolve())}})
    print('Đang thực thi submission.ipynb: validation → dự đoán → kiểm tra định dạng.', flush=True)
    try:
        client.execute(env=env)
    finally:
        nbformat.write(notebook, folder / 'submission.ipynb')
    dump(folder / 'execution_report.json', {'notebook_executed': True,
        'execution_seconds': time.perf_counter() - started,
        'submission_sha256': sha(folder / 'submission.csv')})
    (folder / 'HUONG_DAN.md').write_text(f'''# {team}

## Nộp trên trang chấm
Nộp `submission.csv`, UTF-8, 732 dòng dữ liệu, 22 cột. Không nộp ZIP lên ô CSV.
Đăng nhập bằng gmail đại diện đã đăng ký, tự nhập mã OTP nhận trong hộp thư.

## Môi trường
Python {platform.python_version()}. Các phiên bản chính xác trong requirements.txt.
`python -m pip install -r requirements.txt`
Máy inference: inference_hardware.json. Máy và thời gian huấn luyện gốc:
training_provenance.json và models/*/training_hardware.json nếu có.
Một số trường CPU/RAM của checkpoint GitHub cũ không được ghi nhận; để null,
không coi máy inference là máy huấn luyện. Lượt ba mô hình mới tự ghi phần này.

## Dữ liệu và tái hiện
Tạo thư mục `data/` ngang hàng notebook, đặt train.csv và test_blind.csv do giảng viên cấp.
Mở submission.ipynb, chạy các ô theo thứ tự. Notebook đã có output thực tế.
Train dùng để đo lại validation; test chỉ dùng inference, không fit hoặc chọn ngưỡng.
Thời gian thực tế: execution_report.json, inference_report.json, validation_metrics.json.
Thời gian phụ thuộc GPU/CPU; lần đầu cần mạng tải tokenizer và config Hugging Face.
Các trọng số học sâu và mô hình TF-IDF đã nằm trong models/.

## Kiểm tra và đối chiếu
`python code/score_absa.py --kiem-tra submission.csv data/test_blind.csv`
Đối chiếu SHA256 với inference_report.json và execution_report.json.
Nếu đổi phần cứng, sai khác số học nhỏ có thể làm một số dự đoán gần ngưỡng thay đổi.
Lưu bản submission.csv gốc đã được trang chấm cho điểm.

## Huấn luyện lại
Mã gốc neural_train.py/reference_optimization.py và cấu hình lưu trong models/.
Tái hiện cấu hình đơn: sao chép config.json sang thư mục mới, chạy
`python code/neural_train.py --config THU_MUC_MOI/config.json --train data/train.csv`.
Lượt ba mô hình mới: `python code/colab_runner.py --train data/train.csv --out runs`.
PhoBERT chuẩn hóa/tách từ, ViSoBERT giữ cách viết gốc, TF-IDF dùng cả hai dạng.
Hạt giống 2026; giữ nguyên cách chia nhóm fit/tune/validation.

## Chọn file cuối cùng
PUBLIC_SCORE.json chỉ ghi điểm do người dùng nhập sau khi nộp trên trang chấm.
Không lấy validation để giả làm điểm công khai. Sau mỗi lần nhận điểm, chạy lệnh
ghi điểm ở Colab; CHON_NOP_GIANG_VIEN giữ đúng CSV/ZIP có điểm công khai cao nhất.
Khi chưa có điểm, ZIP trong lượt này là bản dự thảo, chưa xác nhận là cao nhất.
Sau khi trang chấm đóng, nộp {team}.zip đã được chọn.
Không dùng thêm dữ liệu LLM/công cụ tạo nhãn nên không có du_lieu_them.csv.
''', encoding='utf-8')
    register_score(args.out, folder, None, team)
    output = create_zip(folder, team)
    dump(args.out / 'latest_export.json', {'folder': str(folder.resolve()),
        'zip': str(output.resolve()), 'submission': str((folder / 'submission.csv').resolve()),
        'public_score': None})
    print('ĐÃ XUẤT THƯ MỤC:', folder.resolve(), flush=True)
    print('FILE NỘP WEB:', (folder / 'submission.csv').resolve(), flush=True)
    print('ZIP DỰ THẢO CHƯA CÓ ĐIỂM CÔNG KHAI:', output.resolve(), flush=True)


if __name__ == '__main__':
    main()
