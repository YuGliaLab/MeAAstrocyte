"""
Subject-mouse syllable pie charts, byte-for-byte the same style as the Colab
notebook's cell 142 (`subject_syllable_pie_*.pdf`).

Every plotting parameter below is copied from that cell: figsize (8,8), dpi 100,
tab20 colormap, 0.5%-of-frames threshold with an `Other` bucket, sort descending,
autopct '%1.1f%%', startangle 140, pctdistance 0.82, explode 0.02, donut width 0.4,
white wedge edges, and the same title wording and font sizes.

Difference from the notebook: the source is data/model/results.h5 — the OLD model
(2025_05_15-04_13_34) applied to all 8 recordings — so BL and FST come from the SAME
inference pass and are directly comparable. The notebook's version read the old
model's own training results, which contained 8 FST recordings and no BL.

Usage:
    python scripts/pie_charts_colab_format.py
"""
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_H5 = os.path.join(ROOT, "data", "model", "results.h5")
GROUPS_CSV = os.path.join(ROOT, "data", "groups.csv")
OUT = os.path.join(ROOT, "results", "pie_charts")


def wedges_for(syllables):
    """Exactly the notebook's filtering: counts >= 0.5% of frames, rest -> 'Other'."""
    duration_counts = pd.Series(syllables).value_counts().sort_index()
    total_frames = len(syllables)
    threshold = 0.005 * total_frames
    filtered = duration_counts[duration_counts >= threshold].copy()
    other = duration_counts[duration_counts < threshold].sum()
    if other > 0:
        filtered["Other"] = other
    return filtered.sort_values(ascending=False)


def draw(ax, filtered):
    """The notebook's exact pie() call."""
    colors = plt.cm.tab20(np.linspace(0, 1, len(filtered)))
    wedges, texts, autotexts = ax.pie(
        filtered,
        labels=[f"Syllable {idx}" if isinstance(idx, (int, np.integer)) else idx
                for idx in filtered.index],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.82,
        explode=[0.02] * len(filtered),
        wedgeprops={"width": 0.4, "edgecolor": "w"},
    )
    plt.setp(autotexts, size=8, weight="bold", color="black")
    plt.setp(texts, size=9)


def main():
    os.makedirs(OUT, exist_ok=True)
    groups = pd.read_csv(GROUPS_CSV)
    subj = groups[groups["role"] == "subject"].copy()
    # BL block first, then FST, so the numbering reads in a sensible order
    subj["_o"] = subj["condition"].map({"BL": 0, "FST": 1})
    subj = subj.sort_values(["_o", "animal"]).reset_index(drop=True)

    panels = []
    with h5py.File(RESULTS_H5, "r") as f:
        for i, rec in subj.iterrows():
            syllables = f[rec["name"]]["syllable"][:]
            filtered = wedges_for(syllables)
            panels.append((rec["condition"], rec["animal"], filtered))

            short_name = rec["name"].replace(".analysis_track0", "").split("/")[-1]
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
            draw(ax, filtered)
            ax.set_title(f"Subject Behavioral Distribution:\n{short_name}",
                         fontsize=12, pad=15, weight="bold")
            fig.tight_layout()
            safe = short_name.replace(".", "_").replace(" ", "_")
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(OUT, f"subject_syllable_pie_{i}_{safe}.{ext}"),
                            bbox_inches="tight",
                            transparent=(ext == "pdf"),
                            facecolor="white" if ext == "png" else "none")
            plt.close(fig)
            print(f"Generated chart {i+1}/{len(subj)}: "
                  f"subject_syllable_pie_{i}_{safe}.pdf   [{rec['condition']}]")

    # ---- 2x4 contact sheet: BL row on top, FST row below, paired by animal ----
    fig, axes = plt.subplots(2, 4, figsize=(26, 14), dpi=100)
    for ax, (cond, animal, filtered) in zip(axes.ravel(), panels):
        draw(ax, filtered)
        pct = 100 * filtered.get(5, 0) / filtered.sum()
        ax.set_title(f"{cond} — {animal}\nSyllable 5 = {pct:.1f}%",
                     fontsize=15, pad=12, weight="bold")
    fig.suptitle("Subject mice — syllable distribution, BL (top) vs FST (bottom)",
                 fontsize=20, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"ALL_BL_vs_FST_subject.{ext}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nContact sheet: ALL_BL_vs_FST_subject.png")
    print(f"All files in {OUT}")


if __name__ == "__main__":
    main()
