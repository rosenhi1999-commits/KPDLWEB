"""Verify source restriction, notebook execution and validation-only selection."""
from pathlib import Path
import ast
import importlib.util
import json
import tempfile
import numpy as np
import pandas as pd
import nbformat
from export_colab import create_zip, select_best_validation, make_notebook, LiveNotebookClient
from colab_runner import clean_source


def main():
    here = Path(__file__).resolve().parent
    root = here.parent
    restricted = ''.join(map(chr, [116,101,115,116,95,98,108,105,110,100,46,99,115,118]))
    for p in here.glob('*'):
        if p.suffix in ['.py', '.md']:
            assert restricted not in p.read_text().lower(), p.name
    for name in ['neural_train.py', 'score_absa.py', 'absa_core.py', 'reference_optimization.py']:
        result = clean_source(name, (root / name).read_text(encoding='utf-8-sig'))
        ast.parse(result)
        assert restricted not in result.lower(), name
    nb = make_notebook()
    for cell in nb.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
    assert restricted not in nbformat.writes(nb).lower()
    with tempfile.TemporaryDirectory(dir='/workspace/work') as temp:
        path = Path(temp)
        for i, value in [(1, .88), (2, .89), (3, .885)]:
            folder = path / f'KPW2.3_lan{i}'
            folder.mkdir()
            (folder / 'validation_predictions.csv').write_text(f'fixture-{i}')
            (folder / 'validation_metrics.json').write_text(json.dumps({
                'ABSA_F1': value, 'train_sha256': 'same-training-data'}))
            create_zip(folder, 'KPW2.3')
            chosen = select_best_validation(path, folder, 'KPW2.3')
        result = json.loads((chosen / 'LUA_CHON.json').read_text())
        assert result['validation_ABSA_F1'] == .89
        smoke = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
            "print('Executed internal notebook: OK')")])
        LiveNotebookClient(smoke, timeout=60, kernel_name='python3',
            resources={'metadata': {'path': temp}}).execute()
        assert smoke.cells[0].execution_count == 1
        assert 'Executed internal notebook: OK' in smoke.cells[0].outputs[0].text
    # Check unchanged metric computation against the original implementation.
    original_spec = importlib.util.spec_from_file_location('original_metric', root/'score_absa.py')
    original = importlib.util.module_from_spec(original_spec); original_spec.loader.exec_module(original)
    clean = {};exec(clean_source('score_absa.py', (root/'score_absa.py').read_text()),clean)
    rng = np.random.default_rng(2026)
    a = pd.DataFrame(rng.integers(0,2,(80,21)), columns=original.COLS)
    b = pd.DataFrame(rng.integers(0,2,(80,21)), columns=original.COLS)
    a.insert(0,'id',range(80));b.insert(0,'id',range(80))
    left = original.report(a,b,n_boot=20)
    right = clean['report'](a,b,n_boot=20)
    assert left['ABSA_F1'] == right['ABSA_F1']
    assert left['ABSA_F1_CI95'] == right['ABSA_F1_CI95']
    print('Source scope, metric equivalence, validation selection, executed notebook: PASS')


if __name__ == '__main__':
    main()
