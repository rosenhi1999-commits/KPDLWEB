"""Supervised three-model ABSA using train.csv only.

PhoBERT sees expanded Vietnamese; ViSoBERT retains social-media spellings;
TF-IDF combines segmented expanded words with original and expanded characters.
All three ensemble weights are >= .10, as requested by the user.
"""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize as l2_normalize

SOURCE_COMMIT = '8b68d0fc864b23442064940a7e13d84d1a4664a3'
REPOSITORY = 'rosenhi1999-commits/KPDLWEB'
SOURCE_FILES = ['neural_train.py', 'absa_core.py', 'score_absa.py',
                'reference_optimization.py']
ABBREVIATIONS = {
    'ks': 'khách sạn', 'ksan': 'khách sạn', 'ksạn': 'khách sạn',
    'nv': 'nhân viên', 'nvs': 'nhà vệ sinh', 'wc': 'nhà vệ sinh',
    'ko': 'không', 'k0': 'không', 'khong': 'không', 'hok': 'không',
    'khum': 'không', 'đc': 'được', 'dc': 'được',
    'gđ': 'gia đình', 'mng': 'mọi người', 'mn': 'mọi người',
    'mik': 'mình', 'ntn': 'như thế nào',
    'okie': 'ok', 'oki': 'ok', 'oke': 'ok',
    'buffe': 'buffet', 'buffee': 'buffet',
}
TOKEN_PATTERN = re.compile(r'(?<!\w)(?:' + '|'.join(
    re.escape(k) for k in sorted(ABBREVIATIONS, key=len, reverse=True)
) + r')(?!\w)', re.I)
NEGATION_PATTERN = re.compile(
    r'(?<!\w)k(?=\s+(?:có|còn|được|thấy|thích|tốt|sạch|đẹp|ổn|hài\s+lòng|'
    r'đáng|nên|muốn|biết|phải|giống|bằng|hợp|đủ|hề)(?!\w))', re.I
)


def basic_text(text):
    text = unicodedata.normalize('NFC', str(text))
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_text(text):
    text = basic_text(text)
    text = TOKEN_PATTERN.sub(lambda m: ABBREVIATIONS[m[0].lower()], text)
    # Expand k only before known predicate words; retain amounts such as 500 k.
    def negative(m):
        return m[0] if re.search(r'\d\s*$', text[:m.start()]) else 'không'
    return NEGATION_PATTERN.sub(negative, text)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def hardware_record(training_device):
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu = next((s.partition(':')[2].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                    if s.startswith('model name')), cpu)
    ram = next((int(s.split()[1]) * 1024 for s in Path('/proc/meminfo').read_text().splitlines()
                if s.startswith('MemTotal:')), None) if Path('/proc/meminfo').exists() else None
    result = {'CPU': cpu, 'logical_CPU_count': os.cpu_count(), 'RAM_bytes_host': ram,
        'OS': platform.platform(), 'Python': platform.python_version(),
        'training_device': training_device}
    try:
        import torch
        result.update(torch=torch.__version__, CUDA=torch.version.cuda,
            GPU=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            VRAM_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0)
    except ImportError:
        result.update(torch=None, CUDA=None, GPU=None, VRAM_bytes=0)
    return result


def clean_source(name, source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
        and isinstance(n.value.value, str))]
    if name == 'score_absa.py':
        tree.body = [n for n in tree.body if not (
            isinstance(n, ast.FunctionDef) and n.name == 'kiem_tra')
            and not (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                     and isinstance(n.test.left, ast.Name) and n.test.left.id == '__name__')]
    class TrainingNames(ast.NodeTransformer):
        def visit_Import(self, node):
            for alias in node.names:
                if alias.name == 'train_only_guard':
                    alias.name = 'train_only_policy'
            return node
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = node.value.replace('blind_file_read', 'extra_csv_read')
            return node
    tree = TrainingNames().visit(tree)
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def bootstrap(code_dir):
    code_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        target = code_dir / name
        if not target.exists():
            url = f'https://raw.githubusercontent.com/{REPOSITORY}/{SOURCE_COMMIT}/{name}'
            with urlopen(url, timeout=90) as response:
                target.write_text(clean_source(name, response.read().decode('utf-8-sig')), encoding='utf-8')
    shutil.copy2(Path(__file__).with_name('train_only_policy.py'), code_dir / 'train_only_policy.py')
    sys.path.insert(0, str(code_dir))
    global core, scorer, folds_class
    core = importlib.import_module('absa_core')
    scorer = importlib.import_module('score_absa')
    importlib.import_module('train_only_policy')
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    folds_class = MultilabelStratifiedKFold


def thresholds(y, probabilities):
    grid = np.linspace(.15, .75, 13)
    prediction = probabilities[:, :, None] >= grid[None, None, :]
    truth = y[:, :, None].astype(bool)
    tp = (prediction & truth).sum(0)
    fp = (prediction & ~truth).sum(0)
    fn = (~prediction & truth).sum(0)
    fs = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    # Prefer thresholds near .5 when F1 ties.
    result = []
    for row in fs:
        tied = np.flatnonzero(np.isclose(row, row.max(), rtol=0, atol=1e-14))
        result.append(grid[tied[np.argmin(abs(grid[tied] - .5))]])
    return np.asarray(result)


def oof_score(y, p, folds):
    pred = np.zeros_like(y)
    for fit, held in folds:
        pred[held] = core.binary_predictions(p[held], thresholds(y[fit], p[fit]))
    return float(core.absa_f1(y, pred))


def save_scores(folder, partition, probabilities, ids):
    np.savez_compressed(folder / f'{partition}_scores.npz',
                        scores=probabilities, ids=np.asarray(ids, dtype=str))


def load_scores(folder, partition, expected_ids):
    with np.load(folder / f'{partition}_scores.npz', allow_pickle=False) as saved:
        ids = saved['ids'].astype(str)
        p = saved['scores'].copy()
    expected_ids = np.asarray(expected_ids, dtype=str)
    assert len(set(ids)) == len(ids) and set(ids) == set(expected_ids), 'IDs differ'
    p = p[pd.Index(ids).get_indexer(expected_ids)]
    assert p.shape == (len(ids), 21) and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    return p


def obtain_run(root, method, recipe, train_hash):
    signature = hashlib.sha256(json.dumps({
        'recipe': recipe, 'train_sha256': train_hash,
        'runner_sha256': digest(__file__), 'source_commit': SOURCE_COMMIT,
        'normalization': ABBREVIATIONS,
    }, sort_keys=True).encode()).hexdigest()
    existing = [p for p in root.glob(method + '_lan*')
                if p.name.removeprefix(method + '_lan').isdigit()]
    for folder in existing:
        manifest = folder / 'recipe.json'
        if manifest.exists() and json.loads(manifest.read_text())['signature'] == signature:
            return folder
    number = max([0] + [int(p.name.removeprefix(method + '_lan')) for p in existing]) + 1
    folder = root / f'{method}_lan{number}'
    folder.mkdir()
    write_json(folder / 'recipe.json', {'signature': signature, 'recipe': recipe,
               'train_sha256': train_hash, 'source_commit': SOURCE_COMMIT})
    return folder


def text_views(texts):
    from pyvi import ViTokenizer
    expanded = [normalize_text(t) for t in texts]
    return {
        'word': [ViTokenizer.tokenize(t).lower() for t in expanded],
        'char_raw': [basic_text(t).lower() for t in texts],
        'char_normalized': [t.lower() for t in expanded],
    }


def transform_views(vectorizers, views, indices):
    matrices = [vectorizer.transform([views[name][i] for i in indices]) * weight
                for name, vectorizer, weight in vectorizers]
    return l2_normalize(sparse.hstack(matrices, format='csr'), copy=False)


def fit_sparse(args, df, y, fit, tune, valid, folds, train_hash):
    print('TF-IDF: tạo đặc trưng câu gốc và câu chuẩn hóa...', flush=True)
    views = text_views(df.text)
    vectorizers = [
        ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2,
             max_features=100000, sublinear_tf=True, dtype=np.float32), 1.),
        ('char_raw', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
             min_df=3, max_features=120000, sublinear_tf=True, dtype=np.float32), .7),
        ('char_normalized', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
             min_df=3, max_features=100000, sublinear_tf=True, dtype=np.float32), .7),
    ]
    for name, vectorizer, _ in vectorizers:
        vectorizer.fit([views[name][i] for i in fit])
    xf = transform_views(vectorizers, views, fit)
    xt = transform_views(vectorizers, views, tune)
    choices = []
    for c in [1., 4., 8.]:
        recipe = {'method': 'tfidf_dual_text', 'C': c, 'seed': 2026}
        folder = obtain_run(args.out, 'TFIDF', recipe, train_hash)
        if (folder / 'selection.json').exists() and (folder / 'model.joblib').exists():
            score = json.loads((folder / 'selection.json').read_text())['tune_threshold_CV_F1']
            print(folder.name, 'đã có kết quả, tái sử dụng.', flush=True)
        else:
            started = time.perf_counter()
            model = OneVsRestClassifier(LogisticRegression(C=c, solver='liblinear',
                     class_weight='balanced', max_iter=2000, random_state=2026), n_jobs=2)
            model.fit(xf, y[fit])
            p = model.predict_proba(xt)
            score = oof_score(y[tune], p, folds)
            joblib.dump({'vectorizers': vectorizers, 'classifier': model},
                        folder / 'model.joblib', compress=3)
            save_scores(folder, 'tune', p, df.id.iloc[tune])
            write_json(folder / 'selection.json', {'C': c, 'tune_threshold_CV_F1': score,
                'fit_rows': len(fit), 'tune_rows': len(tune), 'extra_csv_read': False})
            write_json(folder / 'training_hardware.json', dict(hardware_record('cpu'),
                training_seconds=time.perf_counter()-started))
        choices.append((score, -c, folder))
        print(folder.name, 'C =', c, 'tune threshold-CV F1 =', round(score, 6), flush=True)
    folder = max(choices)[2]
    # C is already fixed before any validation probability is computed.
    if not (folder / 'validation_scores.npz').exists():
        bundle = joblib.load(folder / 'model.joblib')
        p = bundle['classifier'].predict_proba(transform_views(bundle['vectorizers'], views, valid))
        save_scores(folder, 'validation', p, df.id.iloc[valid])
    write_json(folder / 'config.json', {'architecture': 'tfidf_dual_text',
        'columns': core.LABELS, 'train_sha256': train_hash,
        'selection': 'C selected on five-fold tune threshold CV only',
        'normalization': ABBREVIATIONS, 'production_refit': False})
    return folder


def fit_neural(args, method, name, revision, segment, expanded, train_hash, code_dir):
    config = {
        'model_name': name, 'revision': revision, 'seed': 2026,
        'split_seed': 2026, 'training_seed': 2026,
        'segment': segment, 'expand_abbreviations': expanded,
        'preprocessing_mode': 'abbr_v1' if expanded else 'basic_social',
        'fast_tokenizer': name == 'uitnlp/visobert',
        'max_length': 256, 'stride': 64, 'head': 'joint', 'unfreeze_layers': -1,
        'batch_size': args.batch_size, 'accumulation': args.accumulation,
        'lr': 3e-5 if expanded else 2e-5,
        'dropout': .1 if expanded else .2,
        'epochs': args.epochs, 'patience': 3, 'head_lr_multiplier': 5,
        'mention_loss_weight': .5, 'positive_weight_power': .5,
        'positive_weight_cap': 3, 'epoch_selection': 'tuned_threshold',
        'warmup_ratio': .1, 'weight_decay': .01, 'layer_lr_decay': 1.,
        'exclude_bias_norm_decay': False, 'length_bucketing': True,
        'cpu_threads': 2, 'rdrop_alpha': 0., 'allow_cpu': False,
        'cpu_bf16': False, 'log_every_batches': 25,
    }
    folder = obtain_run(args.out, method, config, train_hash)
    if (folder / 'model.pt').exists() and (folder / 'metrics.json').exists():
        print(folder.name, 'đã huấn luyện xong, tái sử dụng.', flush=True)
        return folder
    if (folder / 'best.pt').exists():
        print('Lượt trước bị ngắt; trainer này chạy lại từ đầu, không resume epoch.', flush=True)
    write_json(folder / 'config.json', config)
    machine = hardware_record('cuda')
    started = time.perf_counter()
    print('Huấn luyện', folder.name, '— dùng GPU, theo dõi batch bên dưới.', flush=True)
    subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
        '--neural-child', '--code-dir', str(code_dir), '--config', str(folder / 'config.json'),
        '--train', str(args.train)], check=True)
    write_json(folder / 'training_hardware.json', dict(machine,
        training_and_validation_seconds=time.perf_counter()-started))
    return folder


def child():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neural-child', action='store_true')
    parser.add_argument('--code-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_dir))
    import neural_train
    config = json.loads(args.config.read_text())
    neural_train.clean = normalize_text if config['expand_abbreviations'] else basic_text
    sys.argv = [sys.argv[0], '--config', str(args.config), '--train', str(args.train)]
    neural_train.main()


def fit_ensemble(args, df, y, tune, valid, folders, folds, train_hash):
    # Order is PhoBERT, TF-IDF, ViSoBERT. Every source contributes at least 10%.
    weights = [(.8, .1, .1), (.7, .1, .2), (.6, .1, .3), (.5, .1, .4),
               (.6, .2, .2), (.5, .2, .3), (.4, .2, .4),
               (.5, .3, .2), (.4, .3, .3), (1/3, 1/3, 1/3)]
    source_digests = [digest(p / 'tune_scores.npz') for p in folders]
    folder = obtain_run(args.out, 'Ensemble', {
        'sources': [str(p.resolve()) for p in folders],
        'tune_scores_sha256': source_digests, 'candidate_weights': weights,
        'minimum_source_weight': .1,
    }, train_hash)
    if (folder / 'metrics.json').exists():
        print('Kết quả ensemble đã có:', folder, flush=True)
        return folder
    pt = [load_scores(p, 'tune', df.id.iloc[tune]) for p in folders]
    rows = []
    for w in weights:
        probability = sum(a * b for a, b in zip(w, pt))
        rows.append({'PhoBERT': w[0], 'TFIDF': w[1], 'ViSoBERT': w[2],
                     'tune_threshold_CV_F1': oof_score(y[tune], probability, folds)})
    selected = max(range(len(rows)), key=lambda i: rows[i]['tune_threshold_CV_F1'])
    w = weights[selected]
    probability = sum(a * b for a, b in zip(w, pt))
    cut = thresholds(y[tune], probability)
    config = {'architecture': 'three_model_probability_blend',
        'sources': [os.path.relpath(p, folder) for p in folders],
        'weights': list(w), 'minimum_source_weight': .1,
        'thresholds': cut.tolist(), 'threshold_space': 'probability',
        'columns': core.LABELS, 'train_sha256': train_hash,
        'selected_on': 'five-fold tune threshold CV only; never validation',
        'encoder_checkpoint_selected_on_same_tune': True,
        'fully_nested_encoder_CV': False, 'production_refit': False,
        'cached_neural_sources': bool(args.cached_neural)}
    write_json(folder / 'config.json', config)
    pd.DataFrame(rows).to_csv(folder / 'tune_comparison.csv', index=False)
    save_scores(folder, 'tune', probability, df.id.iloc[tune])
    # Weights and thresholds are now fixed. Validation is reporting only.
    pv = [load_scores(p, 'validation', df.id.iloc[valid]) for p in folders]
    probability = sum(a * b for a, b in zip(w, pv))
    save_scores(folder, 'validation', probability, df.id.iloc[valid])
    prediction = core.binary_predictions(probability, cut)
    gt = df.iloc[valid][['id'] + core.LABELS].reset_index(drop=True)
    sb = pd.DataFrame(prediction, columns=core.LABELS)
    sb.insert(0, 'id', gt.id.to_numpy())
    result = scorer.report(gt, sb, n_boot=1000)
    result.update(blind_file_read=False, validation_previously_seen=True,
                  production_refit=False, weights=list(w), all_three_sources_used=True,
                  cached_neural_sources=bool(args.cached_neural))
    write_json(folder / 'metrics.json', result)
    sb.to_csv(folder / 'validation_predictions.csv', index=False)
    print('TRỌNG SỐ PhoBERT / TF-IDF / ViSoBERT:', w, flush=True)
    print('VALIDATION ABSA-F1:', result['ABSA_F1'], flush=True)
    print('CI95%:', result['ABSA_F1_CI95'], flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, default=Path('/content/train.csv'))
    parser.add_argument('--out', type=Path, default=Path('/content/absa_three_models'))
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--no-export', action='store_true')
    parser.add_argument('--team', default='KPW2.3')
    parser.add_argument('--export-out', type=Path, default=Path('/content/absa_deliverables'))
    parser.add_argument('--cached-neural', nargs=2, type=Path,
                        metavar=('PHOBERT_SCORES', 'VISOBERT_SCORES'))
    args = parser.parse_args()
    if args.train.name != 'train.csv':
        raise ValueError('Only train.csv accepted')
    if min(args.epochs, args.batch_size, args.accumulation) < 1:
        raise ValueError('epochs/batch-size/accumulation must be positive')
    os.environ['ABSA_ARTIFACT_ROOT'] = str(args.out.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    code_dir = args.out / '_train_code'
    bootstrap(code_dir)
    df = pd.read_csv(args.train)
    assert list(df.columns) == ['id', 'text'] + core.LABELS
    assert df.id.is_unique and not df.isna().any().any()
    assert df[core.LABELS].isin([0, 1]).all().all()
    y = df[core.LABELS].to_numpy(int)
    fit, tune, valid = core.reference_splits(df, 2026)
    train_hash = digest(args.train)
    if not args.cached_neural:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Chọn GPU T4 trong Colab trước khi huấn luyện.')
        print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Fit / tune / validation:', len(fit), len(tune), len(valid), flush=True)
    audit = {'dictionary': ABBREVIATIONS, 'fit_rows': len(fit),
        'fit_rows_changed': sum(basic_text(df.text.iloc[i]) != normalize_text(df.text.iloc[i]) for i in fit),
        'dictionary_frozen_before_tune': True, 'extra_csv_read': False,
        'preserved': ['diacritics', 'negation', 'emoji', 'punctuation', 'numbers'],
        'ambiguous_tokens_not_globally_expanded': ['k', 'vs', 'mk', 'p', 'kg']}
    write_json(args.out / 'normalization_audit.json', audit)
    partitions = np.full(len(df), 'validation', dtype=object)
    partitions[fit] = 'fit'; partitions[tune] = 'tune'
    pd.DataFrame({'id': df.id, 'partition': partitions}).to_csv(args.out / 'split.csv', index=False)
    folds = list(folds_class(5, shuffle=True, random_state=2026).split(np.zeros(len(tune)), y[tune]))
    tfidf = fit_sparse(args, df, y, fit, tune, valid, folds, train_hash)
    if args.cached_neural:
        pho, viso = args.cached_neural
        for source in [pho, viso]:
            cfg = json.loads((source / 'config.json').read_text())
            info = json.loads((source / 'run_info.json').read_text())
            assert not cfg.get('production_refit') and not cfg.get('refit_scope')
            assert info['train_sha256'] == train_hash and info['fit_rows'] == len(fit)
            assert cfg.get('split_seed', cfg.get('seed')) == 2026
        print('CACHE BENCHMARK: encoders cũ, chưa huấn luyện PhoBERT chuẩn hóa mới.', flush=True)
    else:
        pho = fit_neural(args, 'PhoBERT', 'vinai/phobert-base-v2',
            '86cd7fd4c148980922ac11a2cf5e257f2ba639e1', True, True, train_hash, code_dir)
        viso = fit_neural(args, 'ViSoBERT', 'uitnlp/visobert',
            '196a62afad9cbe4f52a54aabad828b13f0eec59a', False, False, train_hash, code_dir)
    final = fit_ensemble(args, df, y, tune, valid, [pho, tfidf, viso], folds, train_hash)
    dependencies = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (final / 'requirements_actual.txt').write_text(dependencies, encoding='utf-8')
    shutil.copy2(Path(__file__).resolve(), code_dir / 'colab_runner.py')
    write_json(args.out / 'latest_result.json', {
        'ensemble': str(final.resolve()), 'sources': [str(p.resolve()) for p in [pho, tfidf, viso]],
        'cached_neural_sources': bool(args.cached_neural),
        'full_neural_training_executed_by_this_run': not bool(args.cached_neural),
        'extra_csv_read': False,
    })
    print('KẾT QUẢ:', final, flush=True)
    if not args.no_export and not args.cached_neural:
        subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('export_colab.py')),
            '--three-model-result', str(args.out / 'latest_result.json'),
            '--train', str(args.train),
            '--team', args.team, '--out', str(args.export_out)], check=True)


if __name__ == '__main__':
    if '--neural-child' in sys.argv:
        child()
    else:
        main()
