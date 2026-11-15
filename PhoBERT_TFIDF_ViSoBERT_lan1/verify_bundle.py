"""Local structural tests. Fixtures below are never deliverable submissions."""
from pathlib import Path
import json
import tempfile
import sys
import numpy as np
import pandas as pd
import nbformat
from export_colab import register_score, sha, LiveNotebookClient
from predict_bundle import LABELS, write_submission


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # Temporary fixtures test I/O and the official checker without test labels.
    with tempfile.TemporaryDirectory(dir='/workspace/work') as temp:
        root = Path(temp)
        class FixtureEngine:
            def probabilities(self, texts):
                return np.zeros((len(texts), 21))
            def frame(self, ids, probabilities):
                result = pd.DataFrame(probabilities.astype(int), columns=LABELS)
                result.insert(0, 'id', list(ids))
                return result
        ids = [f'fixture_{i:04d}' for i in range(732)]
        test = pd.DataFrame({'id': ids[::-1], 'text': ['format fixture'] * 732})
        test.to_csv(root / 'test.csv', index=False)
        demo = FixtureEngine().frame(ids, np.zeros((732, 21)))
        demo.to_csv(root / 'demo.csv', index=False)
        frame = write_submission(FixtureEngine(), root/'test.csv', root/'demo.csv', root/'format.csv')
        assert frame.id.tolist() == ids
        raw = pd.read_csv(root/'format.csv', dtype=str)
        assert raw[LABELS].isin(['0','1']).all().all()
        for n, score in [(1, .88), (2, .9), (3, .89)]:
            folder = root / f'KPW2.3_lan{n}'
            folder.mkdir()
            (folder/'submission.csv').write_text(f'fixture-{n}')
            (folder/'BUNDLE_CONFIG.json').write_text('{}')
            chosen = register_score(root, folder, score, 'KPW2.3')
        selected = json.loads((chosen/'BEST_PUBLIC_SCORE.json').read_text())
        assert selected['public_score'] == .9
        assert sha(chosen/'submission.csv') == selected['submission_sha256']
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
            "import sys; print('Executed notebook smoke:', sys.version.split()[0])")])
        LiveNotebookClient(notebook, timeout=60, kernel_name='python3',
            resources={'metadata': {'path': str(root)}}).execute()
        assert notebook.cells[0].execution_count == 1
        assert 'Executed notebook smoke:' in notebook.cells[0].outputs[0].text
        print('CSV format/order/official checker, public-score selection/hash, executed notebook: PASS')


if __name__ == '__main__':
    main()
