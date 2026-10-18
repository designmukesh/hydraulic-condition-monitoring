raise SystemExit("Historical generator retired; use scripts/build_notebooks.py and scripts/run_notebooks.py")
"""Enrich notebooks/01_eda.ipynb with: data-quality checks, raw-cycle overlays,
sensor correlation heatmap. Inserts cells, renumbers sections, saves unexecuted."""
import nbformat as nbf

P = "../notebooks/01_eda.ipynb"
nb = nbf.read(P, as_version=4)
cells = nb.cells

# locate anchors
def find_md(prefix):
    for i, c in enumerate(cells):
        if c.cell_type == "markdown" and c.source.strip().startswith(prefix):
            return i
    raise ValueError(prefix)

i_inv_code = None
for i, c in enumerate(cells):
    if c.cell_type == "code" and "inv = pd.DataFrame" in c.source:
        i_inv_code = i
assert i_inv_code is not None, "inventory cell not found"

new_cells = [
    nbf.v4.new_markdown_cell(
        "## 3. Data quality: trust, but verify\n\n"
        "Before any analysis: missing values, infinities, duplicate cycles, "
        "and whether the labels match the documented health states. "
        "Real industrial data is never this clean — checking is a habit, not a formality."),
    nbf.v4.new_code_cell(
        'print("=== sensor files ===")\n'
        'for s in SENSORS:\n'
        '    a = load_sensor(s)\n'
        '    print(f"{s:5s} shape={str(a.shape):16s} NaNs={np.isnan(a).sum():,}  infs={np.isinf(a).sum():,}")\n'
        'print("\\nprofile NaNs:", int(profile.isna().sum().sum()))\n'
        'print("duplicate profile rows:", int(profile.duplicated().sum()),\n'
        '      "(expected >0: the rig repeats conditions by design)")\n'
        'ok = all(set(profile[t].unique()) == set(TARGET_LABELS[t].keys()) for t in TARGETS)\n'
        'print("labels match documentation:", ok)'),
    nbf.v4.new_markdown_cell(
        "## 4. What one raw cycle looks like\n\n"
        "Before aggregating anything, look at the raw material: six random 60-second "
        "cycles of oil temperature (TS3) and system pressure (PS1). Every feature we "
        "engineer later is a compression of curves like these."),
    nbf.v4.new_code_cell(
        'rng = np.random.default_rng(7)\n'
        'idx = rng.choice(len(profile), 6, replace=False)\n'
        'fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n'
        'ts3 = load_sensor("TS3")\n'
        'for i in idx:\n'
        '    axes[0].plot(ts3[i], lw=1.4, alpha=0.85)\n'
        'axes[0].set_title("TS3 oil temperature — 6 random raw cycles", fontweight="bold")\n'
        'axes[0].set_xlabel("second within cycle"); axes[0].set_ylabel("°C")\n'
        'axes[0].grid(alpha=0.3)\n'
        'ps1 = load_sensor("PS1")[:, ::50]\n'
        'for i in idx:\n'
        '    axes[1].plot(ps1[i], lw=1.4, alpha=0.85)\n'
        'axes[1].set_title("PS1 pressure — same 6 cycles (downsampled 50×)", fontweight="bold")\n'
        'axes[1].set_xlabel("sample"); axes[1].set_ylabel("bar")\n'
        'axes[1].grid(alpha=0.3)\n'
        'fig.suptitle("Raw cycles: the 60-second operating signature behind every feature",\n'
        '             fontsize=13, fontweight="bold")\n'
        'fig.tight_layout()\n'
        'fig.savefig(f"{FIG}/08_raw_cycles.png", dpi=120)\n'
        'plt.show()'),
    nbf.v4.new_markdown_cell(
        "## 5. Sensor redundancy map\n\n"
        "Per-cycle sensor means, correlated against each other. Strong blocks tell us "
        "which sensors watch the same physical loop (expected: the four temperature "
        "sensors move together) — useful context when we later ask *where to spend "
        "instrumentation budget*. Tree models tolerate this redundancy fine."),
    nbf.v4.new_code_cell(
        'means = pd.DataFrame({s: load_sensor(s).mean(axis=1) for s in SENSORS})\n'
        'corr = means.corr()\n'
        'fig, ax = plt.subplots(figsize=(9, 7.5))\n'
        'im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)\n'
        'ax.set_xticks(range(len(SENSORS))); ax.set_yticks(range(len(SENSORS)))\n'
        'ax.set_xticklabels(SENSORS, rotation=45, ha="right")\n'
        'ax.set_yticklabels(SENSORS)\n'
        'ax.set_title("Correlation of per-cycle sensor means — redundancy map",\n'
        '            fontweight="bold", pad=12)\n'
        'fig.colorbar(im, ax=ax, label="Pearson r")\n'
        'fig.tight_layout()\n'
        'fig.savefig(f"{FIG}/09_sensor_corr.png", dpi=120)\n'
        'plt.show()\n'
        'print("Strongest off-diagonal pairs:")\n'
        'c = corr.where(~np.eye(len(corr), dtype=bool)).stack().sort_values(ascending=False)\n'
        'print(c.head(6).to_string())'),
]

# insert after inventory code cell
cells[i_inv_code + 1:i_inv_code + 1] = new_cells

# renumber old ## 3. and ## 4. markdown headers
for c in cells:
    if c.cell_type == "markdown":
        s = c.source
        if s.strip().startswith("## 3. Do the signals"):
            c.source = s.replace("## 3.", "## 6.", 1)
        elif s.strip().startswith("## 4. Takeaways"):
            c.source = s.replace("## 4.", "## 7.", 1)

nbf.write(nb, P)
print("enriched 01_eda.ipynb:", len(nb.cells), "cells")

