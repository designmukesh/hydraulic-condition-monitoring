"""Figures generated from executed analysis, never hand-entered scores."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from .data import TARGETS


def plot_results(results, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (target, r) in zip(axes.flat, results['targets'].items()):
        ConfusionMatrixDisplay(np.array(r['test']['confusion_matrix']), display_labels=r['test']['labels']).plot(ax=ax, colorbar=False)
        ax.set_title(f"{target}: test macro-F1 {r['test']['macro_f1']:.3f}")
    fig.tight_layout()
    fig.savefig(directory / '03_confusion_matrices.png', dpi=140)
    plt.show()
    for key, label, filename in [('permutation_sensors', 'Validation macro-F1 decrease (permutation)', '04_sensor_importance.png'),
                                 ('ablations', 'Paired CV macro-F1 decrease (retrain RF)', '05_ablation.png')]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        for ax, (target, r) in zip(axes.flat, results['targets'].items()):
            rows = r[key]
            names = [v.get('sensor', v.get('removed')) for v in rows]
            values = [v.get('mean_drop', v.get('mean')) for v in rows]
            order = np.argsort(values)
            ax.barh(np.array(names)[order], np.array(values)[order])
            ax.axvline(0, color='black', linewidth=.5)
            ax.set_title(target)
            ax.set_xlabel(label)
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=140)
        plt.show()
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (target, r) in enumerate(results['targets'].items()):
        values = [v['mean'] for v in r['cv']]
        ax.plot(range(4), values, marker='o', label=target)
    ax.set_xticks(range(4), ['RF stats', 'HGB stats', 'RF stats+FFT', 'HGB stats+FFT'])
    ax.set_ylabel('Training repeated-CV macro-F1')
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / '06_feature_comparison.png', dpi=140)
    plt.show()


def result_table(results):
    return pd.DataFrame([{'target': t, 'model': r['winner']['model'],
        'features': r['winner']['features'], 'cv_mean': r['winner']['mean'],
        'test_macro_f1': r['test']['macro_f1'], 'test_ci95': r['test']['macro_f1_ci95'],
        'test_accuracy': r['test']['accuracy']} for t, r in results['targets'].items()])
