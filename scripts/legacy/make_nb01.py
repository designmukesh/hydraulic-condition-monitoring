raise SystemExit("Historical generator retired; use scripts/build_notebooks.py and scripts/run_notebooks.py")
"""Build notebooks/01_eda.ipynb (unexecuted). Run, then execute with nbconvert."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "hydraulic-venv", "language": "python", "name": "hydraulic-venv"}

C = []  # cells

C.append(nbf.v4.new_markdown_cell("""# Hydraulic System Condition Monitoring — Exploratory Data Analysis

**Business problem.** Unplanned hydraulic failures stop production lines. If we can detect *which* component is degrading (cooler, valve, pump, accumulator) from routine sensor data, maintenance shifts from reactive firefighting to planned, low-cost interventions.

**What this notebook does.** Inspects the UCI hydraulic test-rig dataset (2,205 sixty-second operating cycles, 17 sensors, per-cycle component health labels) to answer: *do the sensor signals actually separate the fault classes?*

**Dataset:** Helwig et al., "Condition monitoring of hydraulic systems", UCI Machine Learning Repository (2018). Independent portfolio project — public research data, not affiliated with any employer.
"""))

C.append(nbf.v4.new_code_cell("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../src")
from data import load_profile, load_sensor, SENSORS, SENSOR_INFO, TARGETS, TARGET_LABELS

FIG = "../reports/figures"
os.makedirs(FIG, exist_ok=True)

profile = load_profile()
print("cycles:", len(profile))
profile.head()
"""))

C.append(nbf.v4.new_markdown_cell("""## 1. Target variables — how the four components degrade

Each cycle is labeled with the health state of four components. Class balance matters: a model can cheat on imbalanced targets, so we check first.
"""))

C.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, t in zip(axes.ravel(), TARGETS):
    counts = profile[t].value_counts().sort_index()
    labels = [f"{v}\\n{TARGET_LABELS[t][v]}" for v in counts.index]
    ax.bar([str(v) for v in counts.index], counts.values, color="#2a7f62", edgecolor="white")
    ax.set_title(t.capitalize(), fontsize=12, fontweight="bold")
    ax.set_ylabel("cycles")
    for i, (v, c) in enumerate(zip(counts.index, counts.values)):
        ax.text(i, c + 12, TARGET_LABELS[t][v], ha="center", fontsize=8, rotation=0)
fig.suptitle("Component health states across 2,205 test-rig cycles", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG}/01_target_distributions.png", dpi=120)
plt.show()
print(profile[TARGETS].describe().T)
"""))

C.append(nbf.v4.new_markdown_cell("""## 2. Sensor inventory

17 sensors at three sampling rates. High-rate pressure/power signals (100 Hz → 6,000 samples/cycle) carry the fast dynamics; 1 Hz temperature/vibration signals carry the slow thermal picture.
"""))

C.append(nbf.v4.new_code_cell("""rows = []
for s in SENSORS:
    a = load_sensor(s)
    rate = {6000: "100 Hz", 600: "10 Hz", 60: "1 Hz"}[a.shape[1]]
    rows.append({"sensor": s, "description": SENSOR_INFO[s], "sampling": rate,
                 "shape": f"{a.shape[0]} x {a.shape[1]}"})
inv = pd.DataFrame(rows).set_index("sensor")
print(inv.to_string())
print("\\nTotal raw values per cycle:",
      sum(load_sensor(s).shape[1] for s in SENSORS))
"""))

C.append(nbf.v4.new_markdown_cell("""## 3. Do the signals separate the fault classes?

The key EDA question. If mean signal traces already diverge by health state, a classifier has something real to learn. Two physical hypotheses:
- **Cooler degradation → higher oil temperatures** (temperature sensors TS1–TS4 should separate)
- **Valve lag → altered pressure dynamics** (pressure sensors PS1–PS6 should separate)
"""))

C.append(nbf.v4.new_code_cell("""def mean_trace(sensor, target, ax, title):
    mat = load_sensor(sensor)
    for val in sorted(profile[target].unique()):
        tr = mat[profile[target].values == val]
        m = tr.mean(axis=0)
        label = f"{val} ({TARGET_LABELS[target][val]})"
        ax.plot(m, label=label, linewidth=1.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("sample within 60 s cycle")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
# temperature sensors vs cooler state (downsampled view: every 1 s already)
mean_trace("TS3", "cooler", axes[0, 0], "TS3 oil temperature by cooler condition")
mean_trace("TS4", "cooler", axes[0, 1], "TS4 oil temperature by cooler condition")
# pressure sensors vs valve state (plot every 20th sample for speed)
for ax, sensor, title in [(axes[1, 0], "PS2", "PS2 pressure by valve condition"),
                          (axes[1, 1], "PS1", "PS1 pressure by valve condition")]:
    mat = load_sensor(sensor)[:, ::20]
    for val in sorted(profile["valve"].unique()):
        tr = mat[profile["valve"].values == val]
        ax.plot(tr.mean(axis=0), label=f"{val} ({TARGET_LABELS['valve'][val]})", linewidth=1.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("sample (downsampled 20x)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("Mean sensor traces split by component health — classes visibly separate",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG}/02_signal_separation.png", dpi=120)
plt.show()
"""))

C.append(nbf.v4.new_markdown_cell("""## 4. Takeaways → modeling

1. **Class balance is workable.** Cooler is near-balanced; valve/pump/accumulator are imbalanced but every class has ≥360 cycles — stratified splits will handle it.
2. **Physics is visible in the raw signals.** Cooler degradation lifts the whole temperature trace; valve lag reshapes pressure dynamics. This is exactly what a feature-based classifier should exploit.
3. **Dimensionality is the challenge:** ~38k raw values per cycle → we need compact per-cycle features (statistics + spectral descriptors) before any classifier sees the data.

Next: `02_modeling.ipynb` — feature engineering, one classifier per component, honest evaluation with confusion matrices and sensor-level feature importance.
"""))

nb["cells"] = C
with open("../notebooks/01_eda.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 01_eda.ipynb")

