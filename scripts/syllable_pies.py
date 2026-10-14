"""
BL vs FST syllable analysis for the subject mouse, from the applied-model results.

Reads data/model/results.h5 — the output of applying the OLD model
(2025_05_15-04_13_34) to all 8 recordings. Both conditions come from that single
inference pass, so syllable identity is preserved and the groups are comparable
(syllable 5 = allogrooming).

Design: 4 animals (S232, S233, S280, S282) x 2 conditions (BL, FST), within-animal
paired, subject mouse (track_0) only.

Outputs to results/:
    pie_<COND>_<ANIMAL>.pdf     per-animal donut charts, matching the Colab style
    pie_group_<COND>.pdf        pooled per-condition donut charts
    syllable_frequencies.csv    full per-recording frequency table
    syllable5_paired.pdf        syllable 5 paired plot with individual animals
    paired_stats.csv            paired tests for every syllable

Usage:
    python scripts/syllable_pies.py [--syllable 5] [--fps 30]
"""
import argparse
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_H5 = os.path.join(ROOT, "data", "model", "results.h5")
GROUPS_CSV = os.path.join(ROOT, "data", "groups.csv")
OUT = os.path.join(ROOT, "results")

# Same 0.5% threshold the Colab notebook used, so charts stay comparable.
MIN_FREQ = 0.005


def load_frequencies():
    """Per-recording syllable frequency (fraction of frames), joined to group info."""
    groups = pd.read_csv(GROUPS_CSV)
    rows = []
    with h5py.File(RESULTS_H5, "r") as f:
        available = set(f.keys())
        missing = set(groups["name"]) - available
        if missing:
            raise SystemExit(f"results.h5 is missing recordings: {sorted(missing)}")
        for _, g in groups.iterrows():
            syl = f[g["name"]]["syllable"][:]
            freq = pd.Series(syl).value_counts(normalize=True)
            for s, v in freq.items():
                rows.append({"name": g["name"], "animal": g["animal"],
                             "condition": g["condition"], "role": g["role"],
                             "syllable": int(s), "frequency": float(v),
                             "n_frames": len(syl)})
    return pd.DataFrame(rows)


def donut(series, title, path, highlight=None):
    """Donut chart matching the Colab notebook's style, with `Other` bucket."""
    keep = series[series >= MIN_FREQ].sort_values(ascending=False)
    other = series[series < MIN_FREQ].sum()
    if other > 0:
        keep["Other"] = other

    labels = [f"Syllable {i}" if isinstance(i, (int, np.integer)) else str(i)
              for i in keep.index]
    colors = list(plt.cm.tab20(np.linspace(0, 1, len(keep))))
    explode = [0.02] * len(keep)
    # pull out and recolour the syllable of interest so it reads at a glance
    if highlight is not None and highlight in keep.index:
        k = list(keep.index).index(highlight)
        explode[k] = 0.18
        colors[k] = "#d62728"
        labels[k] = f"Syllable {highlight}\n(allogrooming)"

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    wedges, texts, autotexts = ax.pie(
        keep, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors,
        pctdistance=0.82, explode=explode,
        wedgeprops={"width": 0.4, "edgecolor": "w"})
    plt.setp(autotexts, size=8, weight="bold", color="black")
    plt.setp(texts, size=9)
    ax.set_title(title, fontsize=12, pad=15, weight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def paired_tests(subj, syllables):
    """Paired BL vs FST test per syllable, on the 4 within-animal differences."""
    wide = subj.pivot_table(index=["animal", "syllable"], columns="condition",
                            values="frequency").fillna(0.0).reset_index()
    out = []
    for s in syllables:
        d = wide[wide["syllable"] == s]
        if len(d) < 3:
            continue
        bl, fst = d["BL"].values, d["FST"].values
        diff = fst - bl
        t, p_t = stats.ttest_rel(fst, bl)
        try:
            w, p_w = stats.wilcoxon(fst, bl)
        except ValueError:
            w, p_w = np.nan, np.nan
        sd = diff.std(ddof=1)
        out.append({
            "syllable": s, "n_pairs": len(d),
            "BL_mean": bl.mean(), "FST_mean": fst.mean(),
            "mean_diff": diff.mean(),
            "cohens_dz": diff.mean() / sd if sd > 0 else np.nan,
            "t": t, "p_paired_t": p_t, "p_wilcoxon": p_w,
            "consistent_direction": bool(np.all(diff > 0) or np.all(diff < 0)),
        })
    return pd.DataFrame(out).sort_values("p_paired_t")


def plot_syllable(subj, syl, path):
    """Paired dot plot: one line per animal, BL -> FST."""
    d = subj[subj["syllable"] == syl].pivot_table(
        index="animal", columns="condition", values="frequency").fillna(0.0)
    fig, ax = plt.subplots(figsize=(4.2, 5), dpi=150)
    for animal, row in d.iterrows():
        ax.plot([0, 1], [row["BL"] * 100, row["FST"] * 100], "-o",
                color="#666", markersize=7, linewidth=1.4, zorder=2)
        ax.annotate(animal, (1.03, row["FST"] * 100), fontsize=8,
                    va="center", color="#333")
    for i, c in enumerate(["BL", "FST"]):
        m, se = d[c].mean() * 100, stats.sem(d[c]) * 100
        ax.errorbar(i, m, yerr=se, fmt="_", color="#d62728", markersize=34,
                    capsize=7, elinewidth=2.2, markeredgewidth=2.6, zorder=3)

    t, p_t = stats.ttest_rel(d["FST"], d["BL"])
    ax.set_xticks([0, 1], ["BL", "FST"])
    ax.set_xlim(-0.4, 1.5)
    ax.set_ylabel(f"Syllable {syl} frequency (% of frames)")
    ax.set_title(f"Syllable {syl} (allogrooming)\n"
                 f"paired t={t:.2f}, p={p_t:.3f}, n=4", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syllable", type=int, default=5)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    df = load_frequencies()
    df.to_csv(os.path.join(OUT, "syllable_frequencies.csv"), index=False)
    subj = df[df["role"] == "subject"]

    # ---- per-animal donuts -------------------------------------------------
    for (cond, animal), g in subj.groupby(["condition", "animal"]):
        s = g.set_index("syllable")["frequency"].sort_index()
        donut(s, f"{cond} - {animal} (subject)",
              os.path.join(OUT, f"pie_{cond}_{animal}.pdf"), highlight=args.syllable)

    # ---- pooled donuts, weighted by frames so long recordings don't dominate
    for cond, g in subj.groupby("condition"):
        s = g.groupby("syllable")["frequency"].mean().sort_index()
        donut(s / s.sum(), f"{cond} group - subject mice (n=4)",
              os.path.join(OUT, f"pie_group_{cond}.pdf"), highlight=args.syllable)

    # ---- statistics --------------------------------------------------------
    stats_df = paired_tests(subj, sorted(subj["syllable"].unique()))
    stats_df.to_csv(os.path.join(OUT, "paired_stats.csv"), index=False)
    plot_syllable(subj, args.syllable,
                  os.path.join(OUT, f"syllable{args.syllable}_paired.pdf"))

    pd.set_option("display.width", 200, "display.max_columns", 30,
                  "display.float_format", lambda v: f"{v:9.4f}")

    print(f"\n=== Syllable {args.syllable} (allogrooming), subject mice ===\n")
    d = subj[subj["syllable"] == args.syllable].pivot_table(
        index="animal", columns="condition", values="frequency").fillna(0.0)
    print((d * 100).round(3).to_string(), "  (% of frames)")

    row = stats_df[stats_df["syllable"] == args.syllable]
    if not row.empty:
        r = row.iloc[0]
        print(f"\n  BL {r['BL_mean']*100:.2f}%  ->  FST {r['FST_mean']*100:.2f}%   "
              f"diff {r['mean_diff']*100:+.2f} pp")
        print(f"  paired t = {r['t']:.3f},  p = {r['p_paired_t']:.4f},  "
              f"Wilcoxon p = {r['p_wilcoxon']:.4f},  dz = {r['cohens_dz']:.2f}")
        print(f"  all 4 animals same direction: {r['consistent_direction']}")

    print("\n=== Top 10 syllables by paired p-value ===\n")
    show = stats_df.head(10)[["syllable", "BL_mean", "FST_mean", "mean_diff",
                              "cohens_dz", "p_paired_t", "p_wilcoxon",
                              "consistent_direction"]]
    print(show.to_string(index=False))
    print(f"\nWrote figures + CSVs to {OUT}")


if __name__ == "__main__":
    main()
