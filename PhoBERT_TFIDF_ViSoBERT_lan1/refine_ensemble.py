"""Tune-only cross-validated refinement of three already-trained models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import colab_runner as runner


def fit_cuts(y, probabilities, strength):
    result = runner.thresholds(y, probabilities)
    if strength:
        for offset in range(3):
            anchor = runner.thresholds(y[:, offset::3].reshape(-1, 1),
                                       probabilities[:, offset::3].reshape(-1, 1))[0]
            for j in range(offset, 21, 3):
                positives = y[:, j].sum()
                weight = positives / (positives + strength)
                result[j] = weight * result[j] + (1 - weight) * anchor
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, default=Path('/content/train.csv'))
    parser.add_argument('--sources', type=Path, nargs=3, required=True,
                        metavar=('PHOBERT', 'TFIDF', 'VISOBERT'))
    parser.add_argument('--out', type=Path, default=Path('/content/absa_runs'))
    parser.add_argument('--legacy-code-dir', type=Path, default=Path('/content/absa_code'))
    parser.add_argument('--export-out', type=Path, default=Path('/content/absa_deliverables'))
    parser.add_argument('--no-export', action='store_true')
    args = parser.parse_args()
    assert args.train.name == 'train.csv'
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ['ABSA_ARTIFACT_ROOT'] = str(args.out.resolve())
    runner.bootstrap(args.out / '_refine_code')
    df = pd.read_csv(args.train)
    assert list(df.columns) == ['id', 'text'] + runner.core.LABELS
    assert df.id.is_unique and not df.isna().any().any()
    assert df[runner.core.LABELS].isin([0, 1]).all().all()
    y = df[runner.core.LABELS].to_numpy(int)
    _, tune, valid = runner.core.reference_splits(df, 2026)
    p_tune = [runner.load_scores(p, 'tune', df.id.iloc[tune]) for p in args.sources]
    folds = list(runner.folds_class(5, shuffle=True, random_state=2026).split(
        np.zeros(len(tune)), y[tune]))
    weights = [(.45,.15,.40), (.5,.15,.35), (.55,.15,.30), (.6,.15,.25),
               (.65,.15,.20), (.7,.15,.15), (.8,.1,.1), (.7,.1,.2),
               (.6,.2,.2), (.5,.2,.3), (.4,.3,.3)]
    rows = []
    for w in weights:
        p = sum(a * b for a, b in zip(w, p_tune))
        for strength in [25, 0]:
            prediction = np.zeros_like(y[tune])
            for fit, held in folds:
                cuts = fit_cuts(y[tune][fit], p[fit], strength)
                prediction[held] = runner.core.binary_predictions(p[held], cuts)
            score = float(runner.core.absa_f1(y[tune], prediction))
            rows.append({'PhoBERT': w[0], 'TFIDF': w[1], 'ViSoBERT': w[2],
                         'threshold_shrink_strength': strength,
                         'tune_threshold_CV_F1': score})
    chosen = max(rows, key=lambda r: r['tune_threshold_CV_F1'])
    w = [chosen[k] for k in ['PhoBERT', 'TFIDF', 'ViSoBERT']]
    p = sum(a * b for a, b in zip(w, p_tune))
    cuts = fit_cuts(y[tune], p, chosen['threshold_shrink_strength'])
    recipe = {'method': 'tune_threshold_CV_shrink_v1',
        'source_tune_sha256': [runner.digest(s / 'tune_scores.npz') for s in args.sources],
        'refinement_code_sha256': runner.digest(__file__), 'weights': weights,
        'strengths': [25, 0], 'seed': 2026}
    folder = runner.obtain_run(args.out, 'Ensemble', recipe, runner.digest(args.train))
    runner.write_json(folder / 'config.json', {
        'architecture': 'three_model_probability_blend',
        'sources': [os.path.relpath(s.resolve(), folder.resolve()) for s in args.sources],
        'weights': w, 'thresholds': cuts.tolist(), 'threshold_space': 'probability',
        'columns': runner.core.LABELS, 'seed': 2026,
        'threshold_shrink_strength': chosen['threshold_shrink_strength'],
        'selection': 'five-fold tune threshold CV only',
        'fully_nested_encoder_CV': False, 'production_refit': False})
    pd.DataFrame(rows).sort_values('tune_threshold_CV_F1', ascending=False).to_csv(
        folder / 'tune_comparison.csv', index=False)
    runner.save_scores(folder, 'tune', p, df.id.iloc[tune])
    # Validation is loaded only after the complete choice is fixed.
    p_valid = [runner.load_scores(s, 'validation', df.id.iloc[valid]) for s in args.sources]
    probabilities = sum(a * b for a, b in zip(w, p_valid))
    prediction = runner.core.binary_predictions(probabilities, cuts)
    truth = df.iloc[valid][['id'] + runner.core.LABELS].reset_index(drop=True)
    proposed = pd.DataFrame(prediction, columns=runner.core.LABELS)
    proposed.insert(0, 'id', truth.id.to_numpy())
    result = runner.scorer.report(truth, proposed, n_boot=1000)
    result.update(selection_scope='tune only', validation_previously_seen=True,
                  production_refit=False, selected_candidate=chosen)
    runner.write_json(folder / 'metrics.json', result)
    proposed.to_csv(folder / 'validation_predictions.csv', index=False, encoding='utf-8')
    runner.save_scores(folder, 'validation', probabilities, df.id.iloc[valid])
    print('LỰA CHỌN TRÊN TUNE:', chosen, flush=True)
    print('ABSA-F1 VALIDATION:', result['ABSA_F1'], flush=True)
    print('CI95%:', result['ABSA_F1_CI95'], flush=True)
    comparison = []
    for index, (name, source, probs) in enumerate(zip(['PhoBERT', 'TFIDF', 'ViSoBERT'], args.sources, p_valid)):
        cfg = json.loads((source / 'config.json').read_text())
        original_cuts = np.asarray(cfg.get('thresholds', []))
        if original_cuts.shape != (21,):
            original_cuts = runner.thresholds(y[tune], p_tune[index])
        if original_cuts.shape == (21,):
            value = runner.core.absa_f1(y[valid], runner.core.binary_predictions(probs, original_cuts))
            comparison.append({'method': name, 'validation_ABSA_F1': float(value)})
    comparison.append({'method': 'Ensemble_refined', 'validation_ABSA_F1': result['ABSA_F1']})
    pd.DataFrame(comparison).to_csv(folder / 'validation_comparison.csv', index=False)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)
    print('Thư mục:', folder.resolve(), flush=True)
    if not args.no_export:
        subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('export_colab.py')),
            '--ensemble-dir', str(folder.resolve()), '--legacy-code-dir', str(args.legacy_code_dir),
            '--train', str(args.train.resolve()), '--team', 'KPW2.3',
            '--out', str(args.export_out.resolve())], check=True)


if __name__ == '__main__':
    main()
