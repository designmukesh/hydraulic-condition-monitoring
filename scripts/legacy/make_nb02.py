raise SystemExit("Historical generator retired; use scripts/build_notebooks.py and scripts/run_notebooks.py")
"""Build notebooks/02_modeling.ipynb (unexecuted). Run, then execute with nbconvert."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "hydraulic-venv", "language": "python", "name": "hydraulic-venv"}

C = []

C.append(nbf.v4.new_markdown_cell("""# Hydraulic Condition Monitoring — Fault Classification

**Goal.** From one 60-second multi-sensor cycle, predict the health state of four components: cooler (3 states), valve (4 states), pump leakage (3 states), accumulator (4 states).

**Approach.**
1. *Feature engineering* — compress ~38k raw samples/cycle into 187 statistical descriptors (mean, std, range, RMS, crest factor, skew, kurtosis, energy …) per sensor. Transparent, explainable, and the standard approach for this benchmark.
2. *One classifier per component* — components degrade independently, so four separate models (RandomForest vs HistGradientBoosting, best by macro-F1).
3. *Honest evaluation* — stratified 80/20 split, accuracy + macro-F1 + confusion matrices. No leakage: features are computed per cycle, split is by cycle.

**Independence note:** independent portfolio project on public UCI research data; not affiliated with any employer.
"""))

C.append(nbf.v4.new_code_cell("""import os, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import sys
sys.path.insert(0, "../src")
from data import build_feature_frame, load_profile, TARGETS, TARGET_LABELS, sensor_of

FIG = "../reports/figures"
os.makedirs(FIG, exist_ok=True)

t0 = time.time()
X = build_feature_frame()
print(f"feature matrix: {X.shape}  (built in {time.time()-t0:.0f} s)")
y = load_profile()
X.head(3)
"""))

C.append(nbf.v4.new_markdown_cell("""## 1. Train one classifier per component

Stratified split keeps every fault class represented in train and test. We compare RandomForest and gradient boosting and keep the winner per component by macro-F1 (macro, because every fault state matters — missing a rare "close to failure" is the expensive error).
"""))

C.append(nbf.v4.new_code_cell("""splits, leaderboard = {}, []
for t in TARGETS:
    yt = y[t].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, yt, test_size=0.2, random_state=42, stratify=yt)
    splits[t] = (Xtr, Xte, ytr, yte)
    for name, mk in [("rf", lambda: RandomForestClassifier(
                                n_estimators=300, n_jobs=-1, random_state=42)),
                     ("hgb", lambda: HistGradientBoostingClassifier(random_state=42))]:
        m = mk().fit(Xtr, ytr)
        pred = m.predict(Xte)
        leaderboard.append({
            "component": t, "model": name,
            "accuracy": round(accuracy_score(yte, pred), 4),
            "macro_f1": round(f1_score(yte, pred, average="macro"), 4),
        })
lb = pd.DataFrame(leaderboard).sort_values(["component", "macro_f1"], ascending=[True, False])
print(lb.to_string(index=False))
"""))

C.append(nbf.v4.new_markdown_cell("""## 2. Confusion matrices — where do the errors concentrate?

For maintenance decisions, *which* states get confused matters more than a single score: confusing "reduced efficiency" with "full efficiency" delays action; confusing the two degraded states is less harmful.
"""))

C.append(nbf.v4.new_code_cell("""winners = {}
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, t in zip(axes.ravel(), TARGETS):
    Xtr, Xte, ytr, yte = splits[t]
    # re-fit winner (rf won or tied everywhere in practice; pick by leaderboard)
    name = lb[lb.component == t].iloc[0]["model"]
    mk = (lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)) \\
        if name == "rf" else (lambda: HistGradientBoostingClassifier(random_state=42))
    m = mk().fit(Xtr, ytr)
    winners[t] = (name, m)
    pred = m.predict(Xte)
    classes = sorted(np.unique(yte))
    disp_labels = [f"{c}\\n{TARGET_LABELS[t][c]}" for c in classes]
    ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=disp_labels) \\
        .plot(ax=ax, cmap="Greens", colorbar=False, values_format="d")
    ax.set_title(f"{t.capitalize()}  ({name}, macro-F1="
                f"{lb[lb.component==t].iloc[0]['macro_f1']:.3f})",
                fontsize=11, fontweight="bold")
fig.suptitle("Confusion matrices on held-out cycles (20% stratified split)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG}/03_confusion_matrices.png", dpi=120)
plt.show()
"""))

C.append(nbf.v4.new_markdown_cell("""## 3. Which sensors matter? — the product insight

Feature importance aggregated back to the *sensor* level tells an asset manager where to invest in instrumentation: if temperature sensors dominate cooler prediction, a cheaper temperature-only retrofit may suffice. (Importance is taken from the random-forest model — the interpretable member of the comparison.)
"""))

C.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sensor_rank = {}
for ax, t in zip(axes.ravel(), TARGETS):
    Xtr, Xte, ytr, yte = splits[t]
    m = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                               random_state=42).fit(Xtr, ytr)
    imp = pd.Series(m.feature_importances_, index=X.columns)
    by_sensor = imp.groupby(imp.index.map(sensor_of)).sum().sort_values(ascending=True)
    sensor_rank[t] = by_sensor
    by_sensor.plot.barh(ax=ax, color="#2a7f62", edgecolor="white")
    ax.set_title(f"{t.capitalize()} — importance by sensor (random forest)",
                fontsize=11, fontweight="bold")
    ax.set_xlabel("summed feature importance")
fig.suptitle("Sensor-level importance: physics shows up where it should",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG}/04_sensor_importance.png", dpi=120)
plt.show()

print("Top-3 sensors per component:")
for t in TARGETS:
    print(f"  {t:12s}: {list(sensor_rank[t].tail(3).index[::-1])}")
"""))

C.append(nbf.v4.new_markdown_cell("""## 4. Results, limitations, next steps

**What we can claim.** Per-cycle statistical features + off-the-shelf tree ensembles separate all four component health states on held-out cycles (see scores above). Sensor importance aligns with physics: temperature sensors dominate cooler state, pressure sensors dominate valve state — the model rediscovers the engineering.

**Limitations (stated plainly).**
- Test-rig data, not field data: real plants add sensor drift, missing data, and mixed fault modes.
- One fixed 60 s cycle framing; a streaming deployment would need windowing + latency analysis.
- Single stratified split with a fixed seed — reported scores are indicative, not a full cross-validation study.

**Next steps.**
- Add spectral (FFT-band) features for the 100 Hz pressure/power signals.
- Try sequence models (1D-CNN/LSTM) on raw signals vs. the feature baseline.
- Feed these predictions into a **maintenance decision engine** (cost of downtime vs. cost of intervention) — the subject of Project #3 in this portfolio.
"""))

C.append(nbf.v4.new_code_cell("""# persist a machine-readable summary for the README / dashboard
summary = {}
for _, r in lb.iterrows():
    summary.setdefault(r["component"], {})[r["model"]] = {
        "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
with open("../reports/metrics.json", "w") as f:
    json.dump({"leaderboard": summary,
               "n_cycles": int(len(y)),
               "n_features": int(X.shape[1])}, f, indent=2)
print(open("../reports/metrics.json").read())
"""))

nb["cells"] = C
with open("../notebooks/02_modeling.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 02_modeling.ipynb")

