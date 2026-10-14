"""
Faithful replica of keypoint-MoSeq's `plot_syll_stats_with_sem`, plus the full
per-video syllable table for downstream use.

Every detail below was taken from the keypoint-moseq source
(keypoint_moseq/analysis.py, jax_moseq/utils/utils.py):

  * frequency = get_frequencies(..., runlength=True)  -> fraction of syllable
    INSTANCES (bouts), not frames. This is what stats_df["frequency"] holds and is
    a different quantity from the frame-share shown in the pie charts.
  * ordering for stat="frequency" is sorted(syllable.unique()) - ASCENDING SYLLABLE
    INDEX, despite the axis label reading "Syllables sorted by frequency". This is
    what gives the plot its zigzag shape.
  * syllables kept where pooled frequency > min_frequency (0.005)
  * sns.pointplot(errorbar=("ci", 68)), default seaborn palette
  * significance stars drawn at y = -0.05 with marker "*", colour red
  * text "{ctrl} vs. {exp} - Total {n} S.S." at (xlim[1], -0.05)
  * legend frameon=False, bbox_to_anchor=(1, 1), with a Line2D star handle
  * xlabel fontsize 12, sns.despine(), figsize (8, 4)

Statistics: kpms runs Kruskal-Wallis + Dunn post-hoc with FDR-BH correction, which
is UNPAIRED. This design is paired (each animal has both a BL and an FST recording),
so a paired test is also reported — see --paired.

Usage:
    python scripts/kpms_stats_replica.py            # kpms-identical figure
    python scripts/kpms_stats_replica.py --paired   # same look, paired test
"""
import argparse
import os

import h5py
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
RESULTS_H5 = os.path.join(ROOT, "data", "model", "results.h5")
GROUPS_CSV = os.path.join(ROOT, "data", "groups.csv")
OUT = os.path.join(ROOT, "results")

CTRL, EXP = "BL_subject", "FST_subject"


def get_frequencies(seq, num_states, runlength=True):
    """Port of jax_moseq.utils.get_frequencies."""
    seq = np.asarray(seq).astype(int)
    if runlength:
        onsets = np.pad(np.diff(seq).nonzero()[0] + 1, (1, 0))
        seq = seq[onsets]
    counts = np.bincount(seq, minlength=num_states)
    return counts / counts.sum()


def build_table():
    """Per-recording syllable frequencies, both instance- and frame-based."""
    groups = pd.read_csv(GROUPS_CSV)
    with h5py.File(RESULTS_H5, "r") as f:
        seqs = {k: f[k]["syllable"][:] for k in f.keys()}
    num_states = max(s.max() for s in seqs.values()) + 1

    rows = []
    for _, g in groups.iterrows():
        s = seqs[g["name"]]
        inst = get_frequencies(s, num_states, runlength=True)
        frame = get_frequencies(s, num_states, runlength=False)
        n_bouts = len(np.pad(np.diff(s).nonzero()[0] + 1, (1, 0)))
        for syl in range(num_states):
            if inst[syl] == 0 and frame[syl] == 0:
                continue
            rows.append({
                "name": g["name"], "animal": g["animal"],
                "condition": g["condition"], "role": g["role"],
                "group": f"{g['condition']}_{g['role']}",
                "syllable": syl,
                "frequency": inst[syl],            # kpms definition (bouts)
                "frame_fraction": frame[syl],      # pie-chart definition
                "n_frames": int((s == syl).sum()),
                "total_frames": len(s), "total_bouts": n_bouts,
            })
    # pooled frequency across every recording, used for the min_frequency filter
    pooled = get_frequencies(np.concatenate(list(seqs.values())), num_states)
    return pd.DataFrame(rows), pooled


def kpms_figure(df, pooled, min_frequency, thresh, paired, path):
    sub = df[df["role"] == "subject"].copy()
    keep = np.where(pooled > min_frequency)[0]
    sub = sub[sub["syllable"].isin(keep)]

    # kpms: for stat="frequency" the ordering is the sorted syllable index
    ordering = sorted(sub["syllable"].unique())

    # ---- significance -------------------------------------------------------
    # Pivot to animal x syllable per condition and fill absent syllables with 0,
    # otherwise a syllable missing from one recording yields unequal-length groups
    # and the paired test cannot line the animals up.
    mat = (sub.pivot_table(index=["condition", "animal"], columns="syllable",
                           values="frequency")
              .reindex(columns=ordering).fillna(0.0))
    bl, fst = mat.loc["BL"], mat.loc["FST"]
    fst = fst.reindex(bl.index)                 # identical animal order -> paired

    pvals = []
    for s in ordering:
        a, b = bl[s].values, fst[s].values
        if paired:
            _, p = stats.ttest_rel(b, a)
        else:
            _, p = stats.kruskal(a, b)
        pvals.append(p)
    rej, p_adj, _, _ = multipletests(pvals, alpha=thresh, method="fdr_bh")
    sig_sylls = [s for s, r in zip(ordering, rej) if r]
    sig_raw = [s for s, p in zip(ordering, pvals) if p < thresh]

    # ---- figure -------------------------------------------------------------
    colors = sns.color_palette(n_colors=2)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.pointplot(data=sub, x="syllable", y="frequency", hue="group",
                  order=ordering, errorbar=("ci", 68), ax=ax,
                  hue_order=[CTRL, EXP], palette=colors)
    handles, _ = ax.get_legend_handles_labels()
    plt.xticks(range(len(ordering)), [str(s) for s in ordering], rotation=90)

    marks = [ordering.index(s) for s in sig_sylls]
    if marks:
        plt.scatter(marks, [-0.05] * len(marks), color="r", marker="*")
    plt.text(plt.xlim()[1], -0.05,
             f"{CTRL} vs. {EXP} - Total {len(sig_sylls)} S.S.")
    handles.append(Line2D([], [], color="red", marker="*", markersize=9,
                          label="Significant Syllable"))

    legend = ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1, 1))
    plt.xlabel("Syllables sorted by frequency", fontsize=12)
    plt.ylabel("frequency")
    sns.despine()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200,
                facecolor="white")
    plt.close(fig)

    return pd.DataFrame({"syllable": ordering, "p_raw": pvals,
                         "p_fdr_bh": p_adj, "significant_fdr": rej}), sig_sylls, sig_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-frequency", type=float, default=0.005)
    ap.add_argument("--thresh", type=float, default=0.05)
    ap.add_argument("--paired", action="store_true",
                    help="paired t-test instead of kpms's unpaired Kruskal-Wallis")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    df, pooled = build_table()
    df.to_csv(os.path.join(OUT, "syllable_table_all_videos.csv"), index=False)

    # Wide tables: one row per syllable, one column per recording — easiest to hand
    # off. Two files because "frequency" is ambiguous: kpms means share of BOUTS,
    # the pie charts mean share of TIME. Both columns sum to 1.0 per recording.
    subject = df[df["role"] == "subject"]
    for value, fname in [("frequency", "syllable_frequency_wide_BOUTS.csv"),
                         ("frame_fraction", "syllable_frequency_wide_TIME.csv")]:
        (subject.pivot_table(index="syllable", columns=["condition", "animal"],
                             values=value)
                .fillna(0.0).sort_index()
                .to_csv(os.path.join(OUT, fname)))
    # keep the original filename as an alias for the kpms (bout-share) definition
    (subject.pivot_table(index="syllable", columns=["condition", "animal"],
                         values="frequency")
            .fillna(0.0).sort_index()
            .to_csv(os.path.join(OUT, "syllable_frequency_wide.csv")))

    tag = "paired" if args.paired else "kpms"
    path = os.path.join(OUT, f"syllable_stats_{tag}_format.pdf")
    ptab, sig_fdr, sig_raw = kpms_figure(df, pooled, args.min_frequency,
                                         args.thresh, args.paired, path)
    ptab.to_csv(os.path.join(OUT, f"pvalues_{tag}.csv"), index=False)

    test = "paired t-test" if args.paired else "Kruskal-Wallis (kpms default)"
    print(f"\n=== {test} ===")
    print(f"syllables kept (pooled freq > {args.min_frequency}): "
          f"{len(ptab)}  -> {ptab['syllable'].tolist()}")
    print(f"significant, raw p < {args.thresh}      : {sig_raw}")
    print(f"significant, after FDR-BH correction: {sig_fdr}\n")
    pd.set_option("display.float_format", lambda v: f"{v:9.4f}")
    print(ptab.to_string(index=False))
    print(f"\nWrote {path}")
    print(f"Wrote {os.path.join(OUT, 'syllable_table_all_videos.csv')}")
    print(f"Wrote {os.path.join(OUT, 'syllable_frequency_wide.csv')}")


if __name__ == "__main__":
    main()
