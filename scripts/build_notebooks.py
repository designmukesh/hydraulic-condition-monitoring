"""Generate portable notebooks; execution is a separate explicit step."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def write(name, cells):
    nb = nbf.v4.new_notebook(cells=[getattr(nbf.v4, f'new_{kind}_cell')(text) for kind, text in cells])
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nbf.write(nb, ROOT / 'notebooks' / name)


setup = '''from pathlib import Path
import sys, json
ROOT = Path.cwd() if (Path.cwd() / 'src').is_dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data import *
FIG = ROOT / 'reports' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
profile = load_profile()
'''

write('01_eda.ipynb', [
('markdown', '''# Hydraulic test-rig data: provenance, stability and features

Helwig, Pignanelli & Schütze (2015), UCI, DOI [10.24432/C5CW21](https://doi.org/10.24432/C5CW21); donated 2018, CC BY 4.0.
2,205 60-second cycles, 43,680 raw values per cycle: 7×6000 + 2×600 + 8×60.
The download is about 73.1 MB compressed; extracted text is substantially larger.

`stable=0` means stable conditions; `stable=1` means static conditions may not yet have been reached. It is diagnostic metadata, never a predictor. Full-dataset descriptive EDA is not an independent test. The protocol is in reports/AUDIT.md.'''),
('code', setup),
('code', '''assert len(profile) == 2205
assert sum(SAMPLE_RATES.values()) * 60 == 43680
display(profile.describe())
display(profile['stable'].value_counts().rename(index={0: 'stable', 1: 'possibly transient'}))
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
stability = {}
for ax, target in zip(axes.flat, TARGETS):
    table = pd.crosstab(profile[target], profile.stable)
    stability[target] = table.to_dict()
    display(table)
    table.rename(columns={0: 'stable', 1: 'possibly transient'}).plot.bar(ax=ax)
    ax.set_title(target)
    ax.set_ylabel('Cycles')
fig.tight_layout()
fig.savefig(FIG / '01_target_stability.png', dpi=140)
plt.show()
(ROOT / 'reports' / 'stability_counts.json').write_text(json.dumps(stability, indent=2))
'''),
('markdown', '''## Cycle-local features
187 original statistical descriptors plus 42 FFT descriptors = 229 columns. FFT uses demeaned Hann-window spectra for PS1–PS6 and EPS1 (100 Hz); six summaries per channel. No FFT claim is made for the 1 Hz vibration summary. Energy/RMS² and range/max/min are redundant; correlated inputs complicate importance. No scaling or imputation is learned globally. Files are loaded one sensor at a time; cache is regenerated on each EDA execution.'''),
('code', '''X = build_feature_frame(spectral=True)
assert X.shape == (2205, 229)
assert np.isfinite(X.to_numpy()).all()
X.to_pickle(ROOT / 'data' / 'features.pkl')
print('Feature matrix:', X.shape)
display(X.head())
means = X[[c for c in X if c.endswith('_mean')]]
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(means.corr(), vmin=-1, vmax=1, cmap='coolwarm')
ax.set_xticks(range(len(means.columns)), means.columns, rotation=90)
ax.set_yticks(range(len(means.columns)), means.columns)
fig.colorbar(im, ax=ax)
ax.set_title('Cycle mean correlations: redundancy, not causality')
fig.tight_layout()
fig.savefig(FIG / '02_sensor_correlation.png', dpi=140)
plt.show()
'''),
('markdown', '''Stability and fault labels can be associated, so stable/transient scores must include class support. They cannot establish that transients cause errors. The modeling notebook reports both strata using the same saved final predictions and the full target label set; absent classes receive zero recall and lower fixed-label macro-F1. Regime/sequence dependence remains a limitation of random-cycle evaluation.''')])

write('02_modeling.ipynb', [
('markdown', '''# Training-only selection and final testing

Protocol fixed in reports/AUDIT.md before new results. For each target: stratified 80/20 split (seed 2026); training-only 5-fold CV repeated twice compares RF/HGB × statistics/statistics+FFT. Different targets have different test cycle IDs. Select by mean macro-F1; deterministic ties favor earlier candidates. Fold SD describes variation, not an independent-sample confidence interval.

Training-only diagnostics: feature and grouped sensor permutations on one validation fold (five repeats); paired three-fold RF ablation of every sensor and family; ordered expanding-window sensitivity with a ten-training-row gap. These diagnostics do not change the selected model. Ordered rows are a proxy for acquisition sequence, not verified timestamps or independent assets.

Only then refit the selected candidate and predict the test once. Bootstrap and stability diagnostics reuse these predictions. This public dataset was previously explored: the test is untouched during this revision's selection, not independent external validation.'''),
('code', setup + '''from src.evaluation import run_experiment
from src.reporting import plot_results, result_table
X = pd.read_pickle(ROOT / 'data' / 'features.pkl')
assert len(X) == len(profile)
'''),
('code', '''results = run_experiment(X, profile, ROOT / 'reports')
display(result_table(results))'''),
('code', '''for target, result in results['targets'].items():
    print(target, 'training-only candidate comparison')
    display(pd.DataFrame(result['cv']))
    print('Final test stable=0 / possibly transient=1, fixed label universe')
    display(pd.DataFrame(result['stability']).T)
    print('Training-only ordered sensitivity')
    display(pd.DataFrame(result['ordered_sensitivity']))
plot_results(results, FIG)
'''),
('markdown', '''## Interpretation and limits

Permutation measures sensitivity of a fitted model to disrupted inputs; retrained ablation measures the diagnostic RF's ability to compensate with remaining features. Correlation can conceal individual importance; permutation can create unrealistic combinations. Five permutation repeats and three ablation folds are exploratory. CE/CP are virtual channels and SE is derived efficiency: removing a physical sensor while retaining derived signals does not simulate a physically feasible retrofit.

The 95% intervals use 2,000 class-stratified bootstrap resamples of fixed predictions. They condition on training, class counts and independent cycles, omit selection uncertainty, and may collapse to [1,1] for perfect predictions. They are not field-performance bounds. Random cycles can share operating regimes. Ordered sensitivity may encounter unseen classes and confounds class shift with ordering.

These test-rig classification results do not establish field reliability, remaining useful life, probability calibration, economic benefit, or maintenance recommendations. Instrumentation purchasing requires physical dependency mapping, sensor cost/reliability evidence and prospective validation on new assets. See PROJECT_GUIDE.md for verified results and remaining work.''')])
