# Hydraulic System Condition Monitoring

**Predict which component is degrading — cooler, valve, pump, or accumulator — from a single 60-second multi-sensor cycle.**

## The problem

Unplanned hydraulic failures stop production lines and inflate maintenance cost. But "something is wrong" isn't actionable — a maintenance planner needs to know **which component** is degrading and **how badly**, so the right spare, technician, and shutdown window can be planned.

This project answers: *can routine sensor data distinguish component-level fault states?*

## What this repo contains

| Notebook | What it does |
|---|---|
| `notebooks/01_eda.ipynb` | Inspects 2,205 test-rig cycles: class balance, sensor inventory, and whether raw signals separate fault states |
| `notebooks/02_modeling.ipynb` | Feature engineering → one classifier per component → honest evaluation (confusion matrices, sensor-level importance) |

`src/data.py` holds the reusable loading + feature-engineering code. `reports/` holds the evaluation figures and `metrics.json`.

## Results (held-out 20% stratified split, 441 cycles)

| Component | States | Best model | Accuracy | Macro-F1 |
|---|---|---|---|---|
| Cooler | 3 (full / reduced / near-failure) | RandomForest / HGB | 1.000 | 1.000 |
| Valve | 4 (optimal → near-failure) | RandomForest | 0.986 | 0.983 |
| Pump leakage | 3 (none / weak / severe) | RandomForest | 0.998 | 0.997 |
| Accumulator | 4 (optimal → near-failure) | HistGradientBoosting | 0.986 | 0.985 |

**The product insight** (from `reports/figures/04_sensor_importance.png`): sensor importance rediscovers the physics — temperature/cooling sensors dominate cooler prediction, pressure sensors dominate valve prediction, flow + efficiency dominate pump and accumulator. That tells an asset manager *where instrumentation money is best spent*.

## How to run

```bash
# 1. Get the data (public UCI dataset, ~500 MB)
python data/download_data.py

# 2. Install
pip install -r requirements.txt

# 3. Open the notebooks
jupyter notebook notebooks/
```

`src/data.py:build_feature_frame()` compresses ~43,700 raw samples per cycle into 187 statistical descriptors (mean, std, range, RMS, crest factor, skew, kurtosis, energy, …) per sensor — transparent and explainable, the standard baseline for this benchmark.

## Methodology notes (stated plainly)

- **No leakage:** features are computed per cycle; the split is by cycle (stratified 80/20, seed 42).
- **Macro-F1 is the headline metric**, because missing a rare "close to failure" state is the expensive error.
- Test-rig data, not field data: real plants add sensor drift, missing data, and mixed fault modes. Scores are indicative, from a single split — not a full cross-validation study.

## Independence note

Independent portfolio project built on public research data (Helwig et al., UCI Machine Learning Repository, 2018). Not affiliated with any employer; no proprietary data used.

## What's next

These predictions become far more valuable inside a **maintenance decision engine** — converting "pump: weak leakage" into *repair now / inspect later / continue operation* using downtime cost, intervention cost, and risk. That's Project #3 in this portfolio series.
