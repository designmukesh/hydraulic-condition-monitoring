raise SystemExit("Historical generator retired; use scripts/build_notebooks.py and scripts/run_notebooks.py")
"""Enrich notebooks/02_modeling.ipynb with: per-class report cards, a prediction
walkthrough on real held-out cycles, top-12 individual features. Renumbers
sections. Saves unexecuted."""
import nbformat as nbf

P = "../notebooks/02_modeling.ipynb"
nb = nbf.read(P, as_version=4)
cells = nb.cells

def find_code(fragment):
    for i, c in enumerate(cells):
        if c.cell_type == "code" and fragment in c.source:
            return i
    raise ValueError(fragment)

i_conf = find_code("ConfusionMatrixDisplay")
i_imp = find_code("sensor_rank")

report_cells = [
    nbf.v4.new_markdown_cell(
        "## 3. Per-class report card\n\n"
        "Macro-F1 summarizes, but a maintenance planner needs the per-state detail: "
        "*precision* (when we raise a flag, how often are we right?) and *recall* "
        "(of all real faults, how many did we catch?) for every health state."),
    nbf.v4.new_code_cell(
        'from sklearn.metrics import classification_report\n'
        'for t in TARGETS:\n'
        '    name, m = winners[t]\n'
        '    Xtr, Xte, ytr, yte = splits[t]\n'
        '    labels = sorted(np.unique(yte))\n'
        '    print(f"--- {t}  [{name}] ---")\n'
        '    print(classification_report(\n'
        '        yte, m.predict(Xte),\n'
        '        target_names=[f"{c} ({TARGET_LABELS[t][c]})" for c in labels],\n'
        '        digits=3))'),
]

walk_cells = [
    nbf.v4.new_markdown_cell(
        "## 4. Prediction walkthrough: real held-out cycles\n\n"
        "Scores are abstract — let's look at actual cycles. For two components we show "
        "correctly classified cycles (with the sensor evidence) and misclassified ones, "
        "comparing each cycle's top-3 feature values against the class means. "
        "This is how you'd debug a model with a domain engineer."),
    nbf.v4.new_code_cell(
        'print("=== prediction walkthrough ===\\n")\n'
        'for t in ["cooler", "valve"]:\n'
        '    name, m = winners[t]\n'
        '    Xtr, Xte, ytr, yte = splits[t]\n'
        '    pred = m.predict(Xte)\n'
        '    idx = np.arange(len(y))\n'
        '    _, te_idx = train_test_split(idx, test_size=0.2, random_state=42,\n'
        '                                 stratify=y[t].values)\n'
        '    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1,\n'
        '                                random_state=42).fit(Xtr, ytr)\n'
        '    imp3 = pd.Series(rf.feature_importances_, index=X.columns).nlargest(3)\n'
        '    ok = pred == yte\n'
        '    show = []\n'
        '    for cls in sorted(np.unique(yte)):\n'
        '        ii = np.where(ok & (yte == cls))[0]\n'
        '        if len(ii):\n'
        '            show.append(("correct", ii[0]))\n'
        '        if len(show) == 2:\n'
        '            break\n'
        '    for j in np.where(~ok)[0][:2]:\n'
        '        show.append(("WRONG", j))\n'
        '    print(f"--- {t} ---")\n'
        '    for kind, j in show:\n'
        '        mark = "✓" if kind == "correct" else "✗ MISCLASSIFIED"\n'
        '        print(f"cycle {te_idx[j]:4d} | true={yte[j]:3d} ({TARGET_LABELS[t][yte[j]]})"\n'
        '              f" -> pred={pred[j]:3d} ({TARGET_LABELS[t][pred[j]]})  {mark}")\n'
        '        for f in imp3.index:\n'
        '            cms = "  ".join(f"{c}:{X[y[t].values == c][f].mean():.1f}"\n'
        '                           for c in sorted(np.unique(yte)))\n'
        '            print(f"    {f:12s} cycle={Xte.iloc[j][f]:8.1f}   class means -> {cms}")\n'
        '        print()'),
    nbf.v4.new_markdown_cell(
        "## 5. Top individual features\n\n"
        "Section 3 showed *sensor*-level importance (good for instrumentation budgets). "
        "Here is the *feature*-level view — useful when an engineer asks exactly "
        "which statistic the model leans on."),
    nbf.v4.new_code_cell(
        'fig, axes = plt.subplots(2, 2, figsize=(13, 9))\n'
        'for ax, t in zip(axes.ravel(), TARGETS):\n'
        '    Xtr, Xte, ytr, yte = splits[t]\n'
        '    m = RandomForestClassifier(n_estimators=300, n_jobs=-1,\n'
        '                               random_state=42).fit(Xtr, ytr)\n'
        '    top = pd.Series(m.feature_importances_, index=X.columns).nlargest(12).sort_values()\n'
        '    top.plot.barh(ax=ax, color="#2a7f62", edgecolor="white")\n'
        '    ax.set_title(f"{t.capitalize()} — top 12 individual features",\n'
        '                 fontweight="bold", fontsize=11)\n'
        '    ax.set_xlabel("importance")\n'
        'fig.suptitle("Top individual features per component (random forest)",\n'
        '             fontsize=14, fontweight="bold")\n'
        'fig.tight_layout()\n'
        'fig.savefig(f"{FIG}/10_top_features.png", dpi=120)\n'
        'plt.show()'),
]

cells[i_conf + 1:i_conf + 1] = report_cells
# i_imp shifted by inserted cells
i_imp2 = find_code("sensor_rank")
cells[i_imp2 + 1:i_imp2 + 1] = walk_cells

for c in cells:
    if c.cell_type == "markdown":
        s = c.source
        if s.strip().startswith("## 3. Which sensors"):
            c.source = s.replace("## 3.", "## 6.", 1)
        elif s.strip().startswith("## 4. Results, limitations"):
            c.source = s.replace("## 4.", "## 7.", 1)

nbf.write(nb, P)
print("enriched 02_modeling.ipynb:", len(nb.cells), "cells")

