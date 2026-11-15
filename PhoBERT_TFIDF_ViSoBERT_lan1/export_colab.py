"""Export an executed internal-evaluation bundle using train.csv only."""
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
from pathlib import Path
import argparse
import ast
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
from colab_runner import SOURCE_COMMIT, REPOSITORY, SOURCE_FILES, clean_source


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
    from colab_runner import hardware_record
    return hardware_record('validation_replay')


def extract_selected(archive, target):
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        found = [n for n in z.namelist() if Path(n).name == 'model.fp16.pt']
        assert len(found) == 1, 'Chọn đúng ZIP checkpoint đã tải'
        prefix = str(Path(found[0]).parent)
        prefix = '' if prefix == '.' else prefix + '/'
        names = ['model.fp16.pt', 'config.json', 'run_info.json',
                 'ci_provenance.json', 'metrics_after_storage_rounding.json']
        for name in names:
            if prefix + name in z.namelist():
                with z.open(prefix + name) as source, (target / name).open('wb') as dest:
                    shutil.copyfileobj(source, dest)
    assert (target / 'config.json').exists()


def copy_model(source, target):
    target.mkdir(parents=True)
    for name in ['config.json', 'recipe.json', 'selection.json', 'history.csv',
                 'metrics.json', 'run_info.json', 'training_hardware.json',
                 'ci_provenance.json', 'model.pt', 'model.fp16.pt', 'model.joblib',
                 'tune_comparison.csv', 'tune_scores.npz', 'validation_scores.npz']:
        if (source / name).exists():
            shutil.copy2(source / name, target / name)


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


def make_notebook(team='KPW2.3'):
    v4 = nbformat.v4
    return v4.new_notebook(cells=[v4.new_markdown_cell(f'''# {team} — đánh giá nội bộ chỉ dùng train.csv

Notebook chạy thật từ checkpoint, giữ output và chỉ số validation.
Dự đoán validation là kết quả nội bộ. Hạt giống 2026. Không tạo thêm dữ liệu.
Validation đã được xem qua các lượt phát triển nên không phải phép đo độc lập mới.
'''), v4.new_code_cell('''import os, sys, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT = Path.cwd()
DATA = Path(os.environ.get("ABSA_TRAIN_PATH", str(ROOT / "data" / "train.csv")))
os.environ["ABSA_ARTIFACT_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT / "code"))
import train_only_policy
random.seed(2026)
np.random.seed(2026)
torch.manual_seed(2026)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2026)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_num_threads(2)
from predict_bundle import Predictor, LABELS, file_hash
from absa_core import reference_splits
import score_absa
assert DATA.name == "train.csv"
print("Python:", sys.version)
print("Thiết bị:", "cuda" if torch.cuda.is_available() else "cpu")
bundle = json.loads((ROOT / "BUNDLE_CONFIG.json").read_text())
engine = Predictor(ROOT / bundle["model_dir"], batch_size=8)
'''), v4.new_code_cell('''train = pd.read_csv(DATA, dtype={"id": str})
assert list(train.columns) == ["id", "text"] + LABELS
assert train.id.is_unique and not train.isna().any().any()
assert train[LABELS].isin([0, 1]).all().all()
fit, tune, valid = reference_splits(train, 2026)
print("Fit / tune / validation:", len(fit), len(tune), len(valid))
started = time.perf_counter()
probabilities = engine.probabilities(train.text.iloc[valid])
prediction = engine.frame(train.id.iloc[valid], probabilities)
truth = train.iloc[valid][["id"] + LABELS].reset_index(drop=True)
assert prediction.id.is_unique and prediction.id.tolist() == truth.id.tolist()
assert prediction[LABELS].isin([0, 1]).all().all()
prediction.to_csv(ROOT / "validation_predictions.csv", index=False, encoding="utf-8")
metrics = score_absa.report(truth, prediction, n_boot=1000)
metrics.update(validation_previously_seen=True, production_refit=False,
               train_sha256=file_hash(DATA),
               replay_seconds=time.perf_counter()-started)
(ROOT / "validation_metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
score_absa.show(metrics)
from IPython.display import display
display(prediction.head())
print("Đây là dự đoán validation nội bộ.")
''')], metadata={'kernelspec': {'display_name': 'Python 3',
                                'language': 'python', 'name': 'python3'}})


def create_zip(folder, team):
    output = folder / (team + '.zip')
    manifest = {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*')
                if p.is_file() and p != output and p.name != 'MANIFEST_SHA256.json'
                and '__pycache__' not in p.parts}
    dump(folder / 'MANIFEST_SHA256.json', manifest)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in folder.rglob('*'):
            if p.is_file() and p != output and '__pycache__' not in p.parts:
                archive.write(p, p.relative_to(folder))
    return output


def select_best_validation(root, folder, team):
    metrics = json.loads((folder / 'validation_metrics.json').read_text())
    record = {'folder': str(folder.resolve()), 'validation_ABSA_F1': metrics['ABSA_F1'],
        'train_sha256': metrics['train_sha256'],
        'validation_previously_seen': True,
        'selection_basis': 'highest observed validation; development comparison',
        'prediction_sha256': sha(folder / 'validation_predictions.csv')}
    registry = root / 'validation_registry.json'
    records = json.loads(registry.read_text()) if registry.exists() else []
    records = [r for r in records if r['folder'] != record['folder']] + [record]
    dump(registry, records)
    eligible = [r for r in records if r['train_sha256'] == record['train_sha256']]
    chosen = max(eligible, key=lambda r: r['validation_ABSA_F1'])
    source = Path(chosen['folder'])
    assert sha(source / 'validation_predictions.csv') == chosen['prediction_sha256']
    target = root / 'CAO_NHAT_VALIDATION'
    target.mkdir(exist_ok=True)
    shutil.copy2(source / (team + '.zip'), target / (team + '.zip'))
    dump(target / 'LUA_CHON.json', chosen)
    return target


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--archive', type=Path)
    source.add_argument('--three-model-result', type=Path)
    source.add_argument('--ensemble-dir', type=Path)
    parser.add_argument('--legacy-code-dir', type=Path)
    parser.add_argument('--train', type=Path, default=Path('/content/train.csv'))
    parser.add_argument('--team', default='KPW2.3')
    parser.add_argument('--out', type=Path, default=Path('/content/absa_deliverables'))
    args = parser.parse_args()
    assert args.train.name == 'train.csv' and args.train.exists()
    team = args.team.removesuffix('.zip')
    assert re.fullmatch(r'[A-Za-z0-9_.-]+', team)
    args.out.mkdir(parents=True, exist_ok=True)
    existing = [p for p in args.out.glob(team + '_lan*')
                if p.name.removeprefix(team + '_lan').isdigit()]
    n = max([0] + [int(p.name.removeprefix(team + '_lan')) for p in existing]) + 1
    folder = args.out / f'{team}_lan{n}'
    code = folder / 'code'
    code.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    for name in ['predict_bundle.py', 'colab_runner.py', 'export_colab.py', 'train_only_policy.py']:
        shutil.copy2(here / name, code / name)
    if (here / 'refine_ensemble.py').exists():
        shutil.copy2(here / 'refine_ensemble.py', code / 'refine_ensemble.py')
    if args.legacy_code_dir:
        legacy = args.legacy_code_dir / 'text_normalization.py'
        if legacy.exists():
            (code / legacy.name).write_text(ast.unparse(ast.parse(legacy.read_text())) + '\n', encoding='utf-8')
    for name in SOURCE_FILES:
        with urlopen(f'https://raw.githubusercontent.com/{REPOSITORY}/{SOURCE_COMMIT}/{name}', timeout=90) as response:
            text = clean_source(name, response.read().decode('utf-8-sig'))
        (code / name).write_text(text, encoding='utf-8')
    if args.archive:
        model = folder / 'models' / 'selected_encoder'
        extract_selected(args.archive, model)
        provenance = {'archive_sha256': sha(args.archive),
            'original_run_info': json.loads((model / 'run_info.json').read_text()),
            'training_CPU_model': None, 'training_RAM_bytes': None,
            'note': 'Old checkpoint did not record CPU model/RAM. Replay hardware is separate.'}
    else:
        if args.ensemble_dir:
            final = args.ensemble_dir.resolve()
            cfg = json.loads((final / 'config.json').read_text())
            sources = []
            for value in cfg['sources']:
                candidate = final / value
                if not candidate.is_dir():
                    candidate = final.parent / value
                assert candidate.is_dir(), 'Không tìm thấy thư mục nguồn: ' + value
                sources.append(candidate.resolve())
            result = {'ensemble': str(final), 'sources': [str(s) for s in sources]}
        else:
            result = json.loads(args.three_model_result.read_text())
            final = Path(result['ensemble'])
        assert not result.get('cached_neural_sources'), 'Cần đủ trọng số mô hình thật'
        for src in [final] + [Path(p) for p in result['sources']]:
            copy_model(src, folder / 'models' / src.name)
        model = folder / 'models' / final.name
        cfg = json.loads((model / 'config.json').read_text())
        cfg['architecture'] = 'three_model_probability_blend'
        cfg['sources'] = ['../' + Path(s).name for s in result['sources']]
        dump(model / 'config.json', cfg)
        provenance = {'models': {}}
        for src in map(Path, result['sources']):
            hardware_path = src / 'training_hardware.json'
            provenance['models'][src.name] = (json.loads(hardware_path.read_text())
                if hardware_path.exists() else {'note': 'See model run_info/selection records'})
    dump(folder / 'BUNDLE_CONFIG.json', {'team': team,
        'model_dir': str(model.relative_to(folder)), 'evaluation_scope': 'train-derived validation only',
        'final_submission_ready': False, 'synthetic_training_data_used': False})
    dump(folder / 'training_provenance.json', provenance)
    dump(folder / 'replay_hardware.json', hardware())
    write_requirements(folder)
    notebook = make_notebook(team)
    started = time.perf_counter()
    client = LiveNotebookClient(notebook, timeout=None, kernel_name='python3',
        resources={'metadata': {'path': str(folder.resolve())}})
    print('Đang chạy notebook đánh giá từ train.csv...', flush=True)
    try:
        client.execute(env=dict(os.environ, ABSA_TRAIN_PATH=str(args.train.resolve())))
    finally:
        nbformat.write(notebook, folder / 'submission.ipynb')
    dump(folder / 'execution_report.json', {'notebook_executed': True,
        'execution_seconds': time.perf_counter()-started, 'input_csv': 'train.csv'})
    (folder / 'HUONG_DAN.md').write_text(f'''# {team} — gói đánh giá nội bộ

Chỉ dùng train.csv. Đây là gói validation nội bộ, chưa phải hồ sơ nộp hoàn chỉnh.
Python {platform.python_version()}; phiên bản thư viện: requirements.txt.
Cài môi trường: python -m pip install -r requirements.txt.
Đặt train.csv trong data/ ngang notebook; mở submission.ipynb và chạy từ trên xuống.
Notebook đã chạy thật, có output và chỉ số; không chèn output giả.
Tham số mô hình nằm trong models/, mã Python trong code/.
Huấn luyện lại ba mô hình: python code/colab_runner.py --train data/train.csv --out runs.
Cùng seed 2026 và cùng chia nhóm fit/tune/validation.
Thời gian/máy huấn luyện: training_provenance.json và models/*/training_hardware.json.
Máy đánh giá hiện tại: replay_hardware.json; thời gian: execution_report.json.
Một số trường máy huấn luyện của checkpoint cũ không được ghi nhận, để null.
Không coi máy đánh giá là máy huấn luyện.
validation_predictions.csv chứa dự đoán nhãn nguyên 0/1 theo ID validation của train.
Không đổi nhãn/ID hoặc đổi tên file này thành bài nộp chính thức.
validation_metrics.json là chỉ số nội bộ; validation đã được xem qua nhiều lượt.
CAO_NHAT_VALIDATION chọn điểm validation quan sát được cao nhất cho cùng train.
MANIFEST_SHA256.json giúp đối chiếu file. Dữ liệu train gốc không nằm trong ZIP.
Không bổ sung dữ liệu tạo sinh nên không có dữ liệu huấn luyện bổ sung.
''', encoding='utf-8')
    output = create_zip(folder, team)
    chosen = select_best_validation(args.out, folder, team)
    dump(args.out / 'latest_export.json', {'folder': str(folder.resolve()),
        'zip': str(output.resolve()), 'best_validation_folder': str(chosen.resolve()),
        'evaluation_scope': 'internal validation only'})
    print('ĐÃ XUẤT:', folder.resolve(), flush=True)
    print('CAO NHẤT VALIDATION:', chosen.resolve(), flush=True)


if __name__ == '__main__':
    main()
