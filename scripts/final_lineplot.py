"""
The one line plot to hand over.

Visual format follows keypoint_moseq's `plot_syll_stats_with_sem` (syllable-index
ordering, seaborn palette, SEM bars, despined axes, star markers below the axis), but
the significance marking reflects how this design should actually be analysed:

  * syllable 5 is the PRE-SPECIFIED primary outcome (labelled allogrooming before this
    comparison existed) -> tested on its own, paired, no multiple-comparison correction
  * every other syllable is exploratory -> FDR-BH corrected across the family; none
    survive, so none are starred

Usage:
    python scripts/final_lineplot.py                 # time share (matches pie charts)
    python scripts/final_lineplot.py --metric bouts  # bout share (kpms convention)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PRESPECIFIED = 5

METRICS = {
    "time": ("frame_fraction", "frequency (fraction of time)"),
    "bouts": ("frequency", "frequency (fraction of bouts)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=list(METRICS), default="time")
    ap.add_argument("--min-frequency", type=float, default=0.005)
    args = ap.parse_args()
    col, ylabel = METRICS[args.metric]

    df = pd.read_csv(os.path.join(RES, "syllable_table_all_videos.csv"))
    sub = df[df["role"] == "subject"].copy()

    pooled = sub.groupby("syllable")[col].mean()
    ordering = sorted(pooled[pooled > args.min_frequency].index)
    sub = sub[sub["syllable"].isin(ordering)]

    mat = (sub.pivot_table(index=["condition", "animal"], columns="syllable",
                           values=col).reindex(columns=ordering).fillna(0.0))
    bl = mat.loc["BL"]
    fst = mat.loc["FST"].reindex(bl.index)          # same animal order -> paired

    # primary outcome: one pre-specified test, uncorrected
    t_p, p_p = stats.ttest_rel(fst[PRESPECIFIED], bl[PRESPECIFIED])

    # exploratory family: everything else, FDR-BH corrected
    others = [s for s in ordering if s != PRESPECIFIED]
    praw = [stats.ttest_rel(fst[s], bl[s])[1] for s in others]
    rej, padj, _, _ = multipletests(praw, alpha=0.05, method="fdr_bh")
    sig_expl = [s for s, r in zip(others, rej) if r]

    long = sub.replace({"condition": {"BL": "BL_subject", "FST": "FST_subject"}})
    colors = sns.color_palette(n_colors=2)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.pointplot(data=long, x="syllable", y=col, hue="condition", order=ordering,
                  errorbar=("ci", 68), ax=ax, palette=colors,
                  hue_order=["BL_subject", "FST_subject"])
    handles, _ = ax.get_legend_handles_labels()
    plt.xticks(range(len(ordering)), [str(s) for s in ordering], rotation=90)

    y0 = -0.05 * ax.get_ylim()[1] / 0.3            # keep the star row just under 0
    k = ordering.index(PRESPECIFIED)
    ax.plot([k], [y0], marker="*", color="red", markersize=13, linestyle="none",
            clip_on=False)
    ax.get_xticklabels()[k].set_color("red")
    ax.get_xticklabels()[k].set_weight("bold")
    if sig_expl:
        ax.plot([ordering.index(s) for s in sig_expl], [y0] * len(sig_expl),
                marker="*", color="grey", markersize=10, linestyle="none",
                clip_on=False)

    handles.append(Line2D([], [], color="red", marker="*", markersize=9, linestyle="none",
                          label=f"Syllable {PRESPECIFIED} (allogrooming)\n"
                                f"pre-specified, paired t p={p_p:.3f}"))
    handles.append(Line2D([], [], color="none",
                          label=f"{len(sig_expl)} of {len(others)} exploratory syllables\n"
                                f"significant after FDR-BH"))
    ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1, 1), fontsize=8)
    plt.xlabel("Syllable", fontsize=12)
    plt.ylabel(ylabel)
    ax.set_title("BL vs FST — subject mice (n=4, within-animal paired)", fontsize=11)
    sns.despine()

    out = os.path.join(RES, f"FINAL_lineplot_{args.metric}.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200,
                facecolor="white")
    plt.close(fig)

    print(f"metric            : {ylabel}")
    print(f"syllables plotted : {ordering}")
    print(f"syllable {PRESPECIFIED} (primary): BL {bl[PRESPECIFIED].mean():.4f} -> "
          f"FST {fst[PRESPECIFIED].mean():.4f},  paired t={t_p:.2f}, p={p_p:.4f}")
    print(f"exploratory       : {len(sig_expl)}/{len(others)} survive FDR-BH  {sig_expl}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
