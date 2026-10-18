"""Training-only selection and diagnostics, followed by a single final prediction."""
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold, TimeSeriesSplit
from threadpoolctl import threadpool_limits
from .data import TARGETS, SENSORS, FAMILIES, sensor_of

SEED = 2026
CANDIDATES = [(m, f) for f in ['statistics', 'statistics_fft'] for m in ['rf', 'hgb']]


def model(name):
    if name == 'rf':
        return RandomForestClassifier(n_estimators=150, n_jobs=4, random_state=SEED)
    if name == 'hgb':
        return HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=SEED)
    raise ValueError(name)


def score(y, pred, labels):
    return float(f1_score(y, pred, labels=labels, average='macro', zero_division=0))


def split_indices(y):
    train, test = train_test_split(np.arange(len(y)), test_size=.2,
                                   stratify=y, random_state=SEED)
    return np.sort(train), np.sort(test)


def bootstrap_interval(y, pred, labels, repeats=2000):
    """Conditional on fixed predictions and observed class counts; iid-cycle assumption."""
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(y == c) for c in labels]
    samples = []
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups if len(g)])
        samples.append(score(y[idx], pred[idx], labels))
    return np.quantile(samples, [.025, .975]).tolist()


def metrics(y, pred, labels):
    return dict(n=len(y), accuracy=float(accuracy_score(y, pred)),
                macro_f1=score(y, pred, labels),
                macro_f1_ci95=bootstrap_interval(y, pred, labels),
                labels=[int(c) for c in labels],
                class_support=[int(sum(y == c)) for c in labels],
                per_class_recall=recall_score(y, pred, labels=labels, average=None, zero_division=0).tolist(),
                confusion_matrix=confusion_matrix(y, pred, labels=labels).tolist())


def select_candidate(X, y):
    """Receives training rows only. Identical folds for every candidate."""
    labels = np.unique(y)
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED).split(X, y))
    rows = []
    for name, features in CANDIDATES:
        cols = [c for c in X if features == 'statistics_fft' or '_fft_' not in c]
        values = []
        for tr, va in folds:
            fitted = model(name).fit(X.iloc[tr][cols], y[tr])
            values.append(score(y[va], fitted.predict(X.iloc[va][cols]), labels))
        rows.append(dict(model=name, features=features, n_features=len(cols),
                         fold_macro_f1=values, mean=float(np.mean(values)), std=float(np.std(values, ddof=1))))
        print(name, features, round(rows[-1]['mean'], 5), flush=True)
    winner = max(rows, key=lambda row: row['mean'])
    return rows, winner, folds[0]


def training_diagnostics(X, y, winner, first_fold):
    labels = np.unique(y)
    selected = [c for c in X if winner['features'] == 'statistics_fft' or '_fft_' not in c]
    tr, va = first_fold
    fitted = model(winner['model']).fit(X.iloc[tr][selected], y[tr])
    importance = permutation_importance(fitted, X.iloc[va][selected], y[va],
                                        scoring='f1_macro', n_repeats=5, random_state=SEED, n_jobs=1)
    permutations = [dict(feature=c, mean=float(m), std=float(s)) for c, m, s in
                    zip(selected, importance.importances_mean, importance.importances_std)]
    # Group permutation preserves within-sensor correlations, breaking alignment to other sensors.
    rng = np.random.default_rng(SEED)
    baseline = score(y[va], fitted.predict(X.iloc[va][selected]), labels)
    grouped = []
    for sensor in SENSORS:
        columns = [c for c in selected if sensor_of(c) == sensor]
        drops = []
        for _ in range(5):
            changed = X.iloc[va][selected].copy()
            changed.loc[:, columns] = changed[columns].to_numpy()[rng.permutation(len(va))]
            drops.append(baseline - score(y[va], fitted.predict(changed), labels))
        grouped.append(dict(sensor=sensor, mean=float(np.mean(drops)), std=float(np.std(drops))))
    # Fixed RF + all features: diagnostic baseline independent of candidate winner.
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, y))
    removals = {'none': []} | {f'sensor:{s}': [s] for s in SENSORS} | {f'family:{k}': v for k, v in FAMILIES.items()}
    ablations = []
    reference = None
    for name, sensors in removals.items():
        cols = [c for c in X if sensor_of(c) not in sensors]
        values = []
        for a, b in folds:
            fitted = model('rf').fit(X.iloc[a][cols], y[a])
            values.append(score(y[b], fitted.predict(X.iloc[b][cols]), labels))
        if reference is None:
            reference = np.array(values)
        ablations.append(dict(removed=name, n_features=len(cols), fold_macro_f1=values,
                              mean=float(np.mean(values)), paired_drop=(reference - values).tolist(),
                              mean_drop=float(np.mean(reference - values))))
    # Preserve original row order inside training; test cycles are absent.
    ordered = []
    for a, b in TimeSeriesSplit(n_splits=3, gap=10).split(X):
        fitted = model(winner['model']).fit(X.iloc[a][selected], y[a])
        ordered.append(dict(n_train=len(a), n_validation=len(b),
                            train_cycle_range=[int(X.index[a[0]]), int(X.index[a[-1]])],
                            validation_cycle_range=[int(X.index[b[0]]), int(X.index[b[-1]])],
                            missing_training_classes=[int(c) for c in labels if c not in y[a]],
                            macro_f1=score(y[b], fitted.predict(X.iloc[b][selected]), labels)))
    return dict(permutation_features=permutations, permutation_sensors=grouped,
                ablations=ablations, ordered_sensitivity=ordered)


def run_experiment(X, profile, report_dir):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output = dict(seed=SEED, n_cycles=len(X), n_features=X.shape[1],
                  python=platform.python_version(), sklearn=sklearn.__version__, targets={})
    # Prevent HGB/OpenMP oversubscription, particularly on Windows laptops.
    with threadpool_limits(limits=2):
        for target in TARGETS:
            print(f'\nTarget: {target}', flush=True)
            y = profile[target].to_numpy()
            train, test = split_indices(y)
            cv, winner, first = select_candidate(X.iloc[train], y[train])
            print('Training diagnostics:', winner['model'], winner['features'], flush=True)
            diagnostics = training_diagnostics(X.iloc[train], y[train], winner, first)
            cols = [c for c in X if winner['features'] == 'statistics_fft' or '_fft_' not in c]
            fitted = model(winner['model']).fit(X.iloc[train][cols], y[train])
            pred = fitted.predict(X.iloc[test][cols])  # only final test prediction
            labels = np.unique(y[train])
            subgroups = {}
            for flag in [0, 1]:
                mask = profile.iloc[test]['stable'].to_numpy() == flag
                subgroups[str(flag)] = metrics(y[test][mask], pred[mask], labels) if mask.any() else {'n': 0}
            output['targets'][target] = dict(cv=cv, winner=winner, test=metrics(y[test], pred, labels),
                stability=subgroups, train_indices=train.tolist(), test_indices=test.tolist(),
                test_truth=y[test].tolist(), test_predictions=pred.tolist(), **diagnostics)
            print('Final test:', output['targets'][target]['test']['macro_f1'], flush=True)
            (report_dir / 'metrics.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    return output
