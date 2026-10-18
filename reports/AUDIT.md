# Baseline audit and fixed analysis protocol

Audited baseline: `9ac7f95` (original history retained).

- RF/HGB selection used the same 20% split reported as held-out performance.
- Stability was loaded but unused; random-cycle separation does not establish independent operating regimes.
- Impurity importance was presented as instrumentation purchasing evidence without ablation.
- Raw sample counts and download-size wording were inconsistent; citation year confused donation with publication.
- Notebook generators contained hard-coded paths; no tests or bounded dependencies existed.

Protocol fixed before new model results: per-target stratified 80/20 split, seed 2026; identical training-only 5-fold CV repeated twice for RF/HGB and statistical/statistical+FFT candidates. Choose highest mean macro-F1 (deterministic candidate-order tie break). Refit selected candidate on all training rows and predict the test once. Store cycle indices and predictions. Existing exploration of the full public dataset means this is a procedural holdout, not a newly collected independent test.

Diagnostics use training data only: feature permutation on the first validation fold, and paired 3-fold RF ablations for each sensor and each family using statistical+FFT features. These are exploratory, not inputs to candidate selection. Ordered expanding-window validation on training cycles is a sensitivity check, not a validated temporal/asset split. No asset identifiers are available.

Final test diagnostics reuse the saved predictions: fixed-label macro-F1, accuracy, confusion matrices, per-class recall and stable/transient subgroups. 2,000 class-stratified bootstrap resamples give conditional cycle-level 95% percentile intervals. They assume independent cycles and omit model-selection/training uncertainty; perfect predictions can yield a degenerate interval. CV fold standard deviation is descriptive, not a confidence interval.

FFT is pre-specified for PS1–PS6 and EPS1 only: demeaned Hann-window power, dominant frequency, spectral centroid, and relative power in 0–1, 1–5, 5–20 and 20–50 Hz bands. No tuning after inspecting test scores.
