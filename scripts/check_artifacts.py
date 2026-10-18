"""Verify stored results against raw labels and independently recomputed metrics."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import nbformat
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_profile, TARGETS
from src.evaluation import split_indices

profile = load_profile()
results = json.loads((ROOT / 'reports/metrics.json').read_text())
assert set(results['targets']) == set(TARGETS)
for target, r in results['targets'].items():
    train, test = map(np.array, [r['train_indices'], r['test_indices']])
    assert len(train) == 1764 and len(test) == 441
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(range(2205))
    assert all(np.array_equal(a, b) for a, b in zip((train, test), split_indices(profile[target])))
    truth, pred = map(np.array, [r['test_truth'], r['test_predictions']])
    assert np.array_equal(truth, profile.iloc[test][target])
    assert np.isclose(f1_score(truth, pred, average='macro'), r['test']['macro_f1'])
    assert np.isclose(accuracy_score(truth, pred), r['test']['accuracy'])
    assert confusion_matrix(truth, pred).tolist() == r['test']['confusion_matrix']
    assert r['winner'] == max(r['cv'], key=lambda c: c['mean'])
    assert len(r['cv']) == 4
    for c in r['cv']:
        assert len(c['fold_macro_f1']) == 10
        assert np.isclose(c['mean'], np.mean(c['fold_macro_f1']))
    assert len(r['ablations']) == 24 and len(r['permutation_sensors']) == 17
    base = np.array(r['ablations'][0]['fold_macro_f1'])
    for a in r['ablations']:
        assert np.allclose(base - a['fold_macro_f1'], a['paired_drop'])
    for flag, subgroup in r['stability'].items():
        mask = profile.iloc[test].stable.to_numpy() == int(flag)
        assert subgroup['n'] == sum(mask)
        assert np.isclose(subgroup['macro_f1'], f1_score(truth[mask], pred[mask], labels=r['test']['labels'], average='macro', zero_division=0))

notebooks = {}
for path in (ROOT / 'notebooks').glob('*.ipynb'):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code)
    assert all(o.output_type != 'error' for c in code for o in c.outputs)
    notebooks[path.name] = {'executed_code_cells': len(code), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
for name in ['01_target_stability.png', '02_sensor_correlation.png', '03_confusion_matrices.png', '04_sensor_importance.png', '05_ablation.png', '06_feature_comparison.png']:
    assert (ROOT / 'reports/figures' / name).stat().st_size > 1000
manifest = json.loads((ROOT / 'reports/data_manifest.json').read_text())
for name, item in manifest['files'].items():
    assert hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest() == item['sha256']
check = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, capture_output=True, text=True)
assert check.returncode == 0, check.stdout + check.stderr
verification = {'status': 'passed', 'tests': check.stdout.strip(), 'notebooks': notebooks,
                'checks': ['raw data hashes', 'test partition separation and reproducibility', 'saved predictions versus raw labels',
                           'recomputed test and subgroup metrics', 'training-CV winner and fold counts',
                           'paired ablation drops', 'executed notebooks without errors', 'six nonempty figures'],
                'metrics_sha256': hashlib.sha256((ROOT / 'reports/metrics.json').read_bytes()).hexdigest()}
(ROOT / 'reports/verification.json').write_text(json.dumps(verification, indent=2))
print(json.dumps(verification, indent=2))
