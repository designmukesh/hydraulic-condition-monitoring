"""Render documentation and tables from completed, executed results."""
import json
import markdown
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / 'reports/metrics.json').read_text())
assert len(r['targets']) == 4, 'Complete all targets first'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


summary = table(['Target', 'Selected model / features', 'CV mean ± SD', 'Test accuracy', 'Test macro-F1 (95% CI)'], [
    [t, f"{v['winner']['model']} / {v['winner']['features']}", f"{v['winner']['mean']:.4f} ± {v['winner']['std']:.4f}",
     f"{v['test']['accuracy']:.4f}", f"{v['test']['macro_f1']:.4f} ({v['test']['macro_f1_ci95'][0]:.4f}–{v['test']['macro_f1_ci95'][1]:.4f})"]
    for t, v in r['targets'].items()])
cv = table(['Target', 'Candidate', 'Features', 'CV mean', 'Fold SD'], [
    [t, c['model'], c['features'], f"{c['mean']:.4f}", f"{c['std']:.4f}"] for t, v in r['targets'].items() for c in v['cv']])
stability = table(['Target', 'Flag', 'Test cycles', 'Class support (label order)', 'Accuracy', 'Fixed-label macro-F1'], [
    [t, flag, s['n'], str(s['class_support']), f"{s['accuracy']:.4f}", f"{s['macro_f1']:.4f}"]
    for t, v in r['targets'].items() for flag, s in v['stability'].items()])
ordered = table(['Target', 'Fold', 'Missing training classes', 'Macro-F1'], [
    [t, i + 1, str(s['missing_training_classes']), f"{s['macro_f1']:.4f}"]
    for t, v in r['targets'].items() for i, s in enumerate(v['ordered_sensitivity'])])
ablations = table(['Target', 'Removed', 'Mean paired F1 decrease', 'Three paired decreases'], [
    [t, a['removed'], f"{a['mean_drop']:.4f}", ', '.join(f'{d:.4f}' for d in a['paired_drop'])]
    for t, v in r['targets'].items() for a in sorted(v['ablations'][1:], key=lambda a: -a['mean_drop'])[:5]])

citation = '''Helwig, N., Pignanelli, E., & Schütze, A. (2015). *Condition monitoring of hydraulic systems* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CW21. Donated April 2018; dataset license: CC BY 4.0.

Official dataset: https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems'''
run = '''```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
python data/download_data.py
python -m pytest -q
python scripts/build_notebooks.py
python scripts/run_notebooks.py
python scripts/build_docs.py
python scripts/check_artifacts.py
```

Python 3.12 recommended. `requirements-lock.txt` records the executed Windows/Python 3.12 environment; `requirements.txt` bounds supported dependency families. Full execution takes substantially longer than unit tests because it fits hundreds of models. Notebook cells have a four-hour timeout. Kernels use the invoking Python environment; no separately named kernel installation is needed. The runner regenerates the feature cache from raw files before modeling. Running it again repeats the fixed experiment; do not tune using the final test scores.
'''
limits = '''- This is a reproducible research/portfolio project, **not a field-deployed monitoring product**. Test-rig cycles do not establish performance on new machines, loads, sensor drift or new fault combinations.
- Cycle-local extraction avoids fitting preprocessing on test data, but random splits can share correlated regimes and neighboring cycles. No blanket “no leakage” claim is made.
- The public dataset and original holdout were already explored. The revised protocol isolates selection from final testing within this revision, but does not undo that historical exposure.
- Bootstrap intervals assume independent cycles and condition on fixed predictions and observed class counts. They omit training/selection uncertainty and can collapse for perfect predictions. Repeated-CV fold SD is descriptive, not a confidence interval.
- Ordered validation is only a sensitivity check on row order. Missing classes and regime shifts can depress its fixed-label score. It is not an asset-held-out or verified chronological deployment test.
- Importance and ablation are model diagnostics, not causal or instrumentation-cost estimates. Virtual/derived CE, CP and SE may retain information from physical sensors removed in an ablation. Procurement requires physical dependency mapping and actual cost/reliability analysis.
- No calibrated probability, remaining-useful-life estimate, maintenance policy, or financial benefit is claimed.'''

readme = f'''# Hydraulic System Condition Monitoring

Reproducible classification of cooler, valve, pump and accumulator condition from public hydraulic test-rig data. The emphasis is an auditable evaluation protocol and explicit limitations.

## Verified results

Each target uses 1,764 training and 441 final-test cycles. Training-only 5-fold CV repeated twice selects RF or HGB and statistical or statistical+FFT features. The selected candidate is refitted on training data, then predicts the test once. All subgroup metrics and intervals reuse those predictions.

{summary}

The 95% intervals are class-stratified percentile bootstrap intervals (2,000 resamples), subject to the limits below. Scores are generated by the executed modeling notebook and stored with split indices, predictions, CV folds and diagnostics in [reports/metrics.json](reports/metrics.json).

![Test confusion matrices](reports/figures/03_confusion_matrices.png)

## What changed

The original project selected models on its reported test split. [The baseline audit](reports/AUDIT.md) records that issue and the new protocol. This revision adds training-only selection, a separate final test, explicit stability analysis, permutation importance, retrained sensor/family ablations, ordered-split sensitivity and an executed FFT baseline comparison.

`stable=0` denotes stable conditions; `stable=1` denotes conditions that may not yet be static. Of 2,205 cycles, 1,449 are stable and 756 possibly transient. All possibly transient valve cycles have label 100; 732/756 possibly transient pump cycles have label 0. Therefore subgroup scores reflect different class mixes as well as possible stability effects. The flag is never used as a model input.

## Reproduce

{run}
Download: about 73.1 MiB compressed (76,601,704 bytes in this run; UCI labels it 73.1 MB), substantially larger after extraction. Content hashes are recorded in [data_manifest.json](reports/data_manifest.json). There are exactly **43,680 raw values per cycle**: seven 100 Hz channels × 6,000, two 10 Hz channels × 600, and eight 1 Hz channels × 60. Features: 187 statistical descriptors, plus 42 spectral descriptors for PS1–PS6 and EPS1 when enabled.

## Repository map

- `src/`: reusable data validation, features, evaluation and reporting.
- `notebooks/01_eda.ipynb`: executed inventory, stability/class counts, feature extraction and correlations.
- `notebooks/02_modeling.ipynb`: executed selection, final testing and diagnostic figures.
- `scripts/`: notebook execution and documentation generation; `scripts/legacy/` contains disabled original generators for provenance.
- `tests/`: synthetic-data checks; GitHub Actions runs unit tests without downloading the dataset.
- `reports/`: verified metrics, figures, data provenance and audit.
- [PROJECT_GUIDE.md](PROJECT_GUIDE.md): methodology, candidate comparisons and interpretation.

## Limits and remaining work

{limits}

Next steps: external assets and prospective acquisition; physically feasible sensor subsets; calibration and missing/drifting sensor tests; grouped evaluation with reliable asset/regime identifiers; a measured operational decision policy. FFT is implemented and compared here, not deferred or presented as a guaranteed improvement.

## Data citation and independence

{citation}

Independent portfolio work; no employer affiliation or proprietary data. Original commit history is retained.
'''
guide = f'''# Project guide: evaluating hydraulic condition classifiers

## 1. What the project measures

The task is four separate supervised classifications of the annotated state during a 60-second laboratory cycle. It is not forecasting future failure. Labels are cooler {{3,20,100}}, valve {{73,80,90,100}}, pump {{0,1,2}}, accumulator {{90,100,115,130}}. Macro-F1 gives each state equal weight; it is not a measured downtime cost.

## 2. Data and features

2,205 cycles contain 43,680 values each: 7×6000 + 2×600 + 8×60. Download is 76,601,704 bytes for the archive used here (roughly 73.1 MiB); extracted text is larger. `reports/data_manifest.json` records SHA-256 hashes.

Each of 17 channels yields mean, population SD, min, max, median, range, RMS, crest factor, skew, excess kurtosis and mean-square energy. Constant-signal skew/kurtosis are defined as zero. Empty, nonfinite, invalid-label or misaligned inputs are rejected. Redundancies such as energy=RMS² remain for continuity with the original baseline; tree models tolerate them, but importance requires caution.

Seven 100 Hz pressure/power channels additionally yield dominant frequency, spectral centroid, and relative power in [0,1), [1,5), [5,20), [20,50] Hz bands. Signals are demeaned and Hann-windowed; DC is excluded. These summarize the repeated load cycle; they are not calibrated physical power spectral density or proof of fault-frequency mechanisms. No high-frequency vibration claim is made from VS1's 1 Hz readings.

## 3. Evaluation protocol

Per-target stratified 80/20 split with seed 2026; split indices are stored. Different targets have different test IDs. Only training data enters 5-fold stratified CV repeated twice, with identical folds across all four candidates. RF uses 150 trees; HGB uses 100 iterations with early stopping disabled. Both use seed 2026 and default remaining hyperparameters. No additional hyperparameter search was performed. Highest mean macro-F1 wins; exact ties use candidate order (statistics RF, statistics HGB, FFT RF, FFT HGB).

After training-only diagnostics, refit the winner on all 1,764 training cycles and predict the 441 test cycles once. Store truth and predictions; bootstrap, subgroups and confusion matrices use the stored prediction vector. `stable` and cycle ID are not predictors. Audit and protocol were committed before results.

{summary}

### Statistical versus FFT comparison (training only)

{cv}

![Feature comparison](reports/figures/06_feature_comparison.png)

This is a controlled candidate comparison using the same folds. Small differences do not establish a statistically significant improvement; correlated fold scores are not independent replicates. The final test only scores the selected candidate, avoiding a second round of test-based feature selection.

## 4. Stability and class composition

There are 1,449 stable and 756 possibly transient cycles. The latter contain only valve class 100 and pump counts {{0:732,1:12,2:12}}. Stability is associated with the experimental condition mix; these data cannot isolate a causal effect of transients.

{stability}

Label order for support vectors is the ascending order listed in section 1. Fixed-label macro-F1 includes every target class and assigns zero to absent classes. Consequently a perfect single-class valve subgroup scores 0.25, not 1.0; consult accuracy and class support together. Sparse classes make subgroup intervals especially uncertain. Full subgroup recalls and intervals are in metrics.json. No stable-only retraining was used to select the model.

![Stability distribution](reports/figures/01_target_stability.png)

## 5. Explainability and retrained ablation

The selected candidate is fitted on the first training CV fold and evaluated on its validation fold. Feature permutation and grouped sensor permutation use five shuffles. Grouped permutation moves a sensor's columns together, retaining within-sensor correlations. Neither uses final-test rows. Validation importance is exploratory and can be optimistic after candidate selection on overlapping training folds.

Separately, a fixed RF using all 229 features is retrained on identical three-fold training splits after removing each of 17 sensors and each of six sensor families. Paired decreases compare each ablation to the full RF on the same folds. Negative values mean a small improvement after removal; they are not suppressed. Single-channel families duplicate their corresponding sensor ablation by design. This diagnostic baseline may differ from the selected final model.

Largest mean decreases (five per target):

{ablations}

![Sensor permutation](reports/figures/04_sensor_importance.png)
![Ablation](reports/figures/05_ablation.png)

Correlated substitutes may mask importance. Shuffling may create unrealistic inputs. Derived channels may preserve physical measurements supposedly removed. These results support hypotheses for follow-up engineering work, not a purchase list or a causal sensor ranking. Full signed effects and fold scores are retained in metrics.json.

## 6. Ordered sensitivity

Selected candidates are assessed on three expanding windows of training cycles sorted by original row index, with a ten-training-row gap. Final-test cycles are excluded. No decision is revised after these checks. The gap is not a verified physical time interval.

{ordered}

These scores deliberately retain the full target label universe. Missing training classes or classes absent from a validation window depress fixed-label macro-F1; lower values alone do not quantify temporal drift. The experiment demonstrates sensitivity to distribution/class coverage and the weakness of treating random-cycle scores as field evidence.

## 7. Reproduction and verification

{run}
The notebooks generate all current figures; old figures with outdated claims were retired. Documentation tables are generated from metrics.json. Unit tests cover known-signal statistics and FFT, malformed/misaligned loading, stable-label validation, fixed-label metrics and reproducible splits. The artifact check independently recomputes metrics, verifies split separation and selection, validates output notebooks and checks required figures. See reports/verification.json for executed checks.

## 8. What remains unproven

{limits}

Future work: prospective external data, known asset/regime groups, physically valid reduced instrumentation, calibration, drift/missing-sensor robustness, measured inference latency and operational cost-based validation. No dashboard or deployment service is claimed in this revision.

## Citation

{citation}
'''
(ROOT / 'README.md').write_text(readme, encoding='utf-8')
(ROOT / 'PROJECT_GUIDE.md').write_text(guide, encoding='utf-8')
(ROOT / 'PROJECT_GUIDE.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Hydraulic project guide</title><style>body{max-width:1100px;margin:40px auto;padding:20px;font:16px/1.6 system-ui;color:#18332d}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f5f3;padding:16px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cbd7d3;padding:8px;text-align:left}th{background:#e5efea}img{max-width:100%}h2{margin-top:48px}</style><body>' + markdown.markdown(guide, extensions=['tables', 'fenced_code']) + '</body></html>', encoding='utf-8')
print('Documentation generated from completed metrics')
