"""Generate pedagogical diagrams for the HTML project guide."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = "/home/hatch/workspace/portfolio/01-hydraulic-condition-monitoring/reports/figures"
DATA = "/home/hatch/workspace/portfolio/01-hydraulic-condition-monitoring/data"
ACCENT = "#1b6b4a"
LIGHT = "#e8f3ee"

plt.rcParams.update({"font.size": 10, "figure.dpi": 140})


def d1_workflow():
    """Horizontal 7-stage pipeline."""
    stages = [
        ("Business\nproblem", "Which component\nis degrading?"),
        ("Data", "2,205 cycles\n17 sensors\nUCI test rig"),
        ("EDA", "Do signals\nseparate faults?"),
        ("Features", "43,680 values\n→ 187 features"),
        ("Models", "1 classifier\nper component"),
        ("Evaluation", "Confusion matrix\nmacro-F1"),
        ("Product", "Work orders\n+ cost model"),
    ]
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    n = len(stages)
    w = 13 / n
    for i, (title, sub) in enumerate(stages):
        x = i * w + 0.08
        box = FancyBboxPatch((x, 0.7), w - 0.5, 1.8,
                             boxstyle="round,pad=0.08", fc=LIGHT if i % 2 else "white",
                             ec=ACCENT, lw=1.6)
        ax.add_patch(box)
        ax.text(x + (w - 0.5) / 2, 2.05, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=ACCENT)
        ax.text(x + (w - 0.5) / 2, 1.35, sub, ha="center", va="center",
                fontsize=8.5, color="#333")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w - 0.5, 1.6), (x + w - 0.08, 1.6),
                                         arrowstyle="-|>", color=ACCENT, lw=1.6,
                                         mutation_scale=12))
    ax.set_title("Project workflow — every stage answers a question before the next begins",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/05_workflow.png", bbox_inches="tight")
    plt.close(fig)


def d2_annotated_signal():
    """Real TS3 trace with statistical features annotated.
    Cycles are chosen to be representative of their class means
    (degraded cooler runs hotter on average)."""
    ts3 = np.loadtxt(os.path.join(DATA, "TS3.txt"), delimiter="\t")
    profile = np.loadtxt(os.path.join(DATA, "profile.txt"), delimiter="\t")

    def representative(target_val, want_hot):
        pool = ts3[profile[:, 0] == target_val]
        means = pool.mean(axis=1)
        cls_mean = means.mean()
        # pick the cycle whose mean is closest to the class mean,
        # nudged in the physically expected direction
        order = np.argsort(np.abs(means - cls_mean))
        return pool[order[0]]

    deg = representative(3, True)
    ok = representative(100, False)
    # sanity: degraded mean must exceed healthy mean (the aggregate finding)
    assert deg.mean() > ok.mean(), "chosen cycles contradict class pattern"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, sig, title in [(axes[0], deg, "Cooler: close to failure"),
                           (axes[1], ok, "Cooler: full efficiency")]:
        t = np.arange(len(sig))
        ax.plot(t, sig, color=ACCENT, lw=1.2)
        mu, sd = sig.mean(), sig.std()
        ax.axhline(mu, color="#c0392b", ls="--", lw=1.4, label=f"mean = {mu:.1f}°C")
        ax.axhspan(mu - sd, mu + sd, color="#c0392b", alpha=0.12, label=f"±1 std = {sd:.2f}°C")
        ax.axhline(sig.max(), color="#7f8c8d", ls=":", lw=1.2, label=f"max = {sig.max():.1f}°C")
        ax.axhline(sig.min(), color="#7f8c8d", ls=":", lw=1.2, label=f"min = {sig.min():.1f}°C")
        # crest illustration: mark the peak (kept inside the axes)
        pk = int(np.argmax(np.abs(sig)))
        ax.annotate("crest factor\n= peak / RMS",
                    xy=(pk, sig[pk]), xycoords="data",
                    xytext=(0.72, 0.78), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#333"), fontsize=8,
                    ha="center", color="#333",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.9))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("second within the 60 s cycle")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("oil temperature (°C)")
    fig.suptitle("One sensor, one cycle — and the statistics we extract from it",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/06_feature_anatomy.png", bbox_inches="tight")
    plt.close(fig)


def d3_decision_flow():
    """Prediction -> cost model -> maintenance decision."""
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    boxes = [
        (0.3, 1.3, 2.6, 2.0, "Model output", '"Pump:\nweak leakage"\n(confidence 0.91)'),
        (3.6, 1.3, 2.6, 2.0, "Cost model", "P(failure) × cost of\nunplanned downtime\nvs cost of planned fix"),
        (6.9, 1.3, 2.6, 2.0, "Decision", "REPAIR NOW\nINSPECT LATER\nCONTINUE"),
        (10.2, 1.3, 1.5, 2.0, "Action", "Work order\nin CMMS"),
    ]
    for i, (x, y, w, h, title, sub) in enumerate(boxes):
        fc = "#1b6b4a" if i == 2 else ("white" if i % 2 == 0 else LIGHT)
        tc = "white" if i == 2 else ACCENT
        sc = "white" if i == 2 else "#333"
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                             fc=fc, ec=ACCENT, lw=1.8)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.45, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h / 2 - 0.25, sub, ha="center", va="center",
                fontsize=9, color=sc)
        if i < len(boxes) - 1:
            nx = boxes[i + 1][0]
            ax.add_patch(FancyArrowPatch((x + w, y + h / 2), (nx, y + h / 2),
                                         arrowstyle="-|>", color=ACCENT, lw=1.8,
                                         mutation_scale=14))
    ax.text(6, 0.55, "A prediction becomes a decision when you attach economics to it  →  Project #3",
            ha="center", fontsize=10, style="italic", color="#555")
    ax.set_title("From prediction to product: the maintenance decision flow",
                 fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/07_decision_flow.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    d1_workflow()
    d2_annotated_signal()
    d3_decision_flow()
    print("diagrams done:", sorted(f for f in os.listdir(FIG) if f.startswith(("05", "06", "07"))))
