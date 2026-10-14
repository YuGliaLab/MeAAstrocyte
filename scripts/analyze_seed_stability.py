"""
How much does apply_model's output move between inference runs?

Compares 3 independent apply_model passes (seeds 0/1/2, identical input) against each
other and against the original run stored in results.h5, then re-tests BL vs FST using
seed-averaged values so the effect carries inference uncertainty.

Usage:
    python scripts/analyze_seed_stability.py
"""
import os

import h5py
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS_CSV = os.path.join(ROOT, "data", "seed_test", "seed_stability.csv")
RESULTS_H5 = os.path.join(ROOT, "data", "model", "results.h5")
GROUPS = os.path.join(ROOT, "data", "groups.csv")
TARGET = 5


def main():
    if not os.path.exists(SEEDS_CSV):
        raise SystemExit(
            f"Not found: {SEEDS_CSV}\n\n"
            "This script analyses the output of scripts/colab_seed_stability.py, which "
            "runs in Colab.\nPlace its seed_stability.csv at that path and re-run.")
    seeds = pd.read_csv(SEEDS_CSV)
    groups = pd.read_csv(GROUPS)[["name", "animal", "condition", "role"]]
    df = seeds.merge(groups, left_on="recording", right_on="name")

    # original single run, for reference
    with h5py.File(RESULTS_H5, "r") as f:
        orig = {k: float((f[k]["syllable"][:] == TARGET).mean()) for k in f.keys()}
    df["original"] = df["recording"].map(orig)

    pd.set_option("display.width", 200, "display.max_columns", 30)

    # ── run-to-run spread ────────────────────────────────────────────────
    g = df.groupby(["condition", "animal"])["time_share"]
    tab = pd.DataFrame({
        "seed0": df[df.seed == 0].set_index(["condition", "animal"])["time_share"],
        "seed1": df[df.seed == 1].set_index(["condition", "animal"])["time_share"],
        "seed2": df[df.seed == 2].set_index(["condition", "animal"])["time_share"],
    })
    tab["mean"] = g.mean()
    tab["sd"] = g.std(ddof=1)
    tab["range_pp"] = (g.max() - g.min()) * 100
    tab["original"] = df.groupby(["condition", "animal"])["original"].first()
    tab["orig_minus_mean_pp"] = (tab["original"] - tab["mean"]) * 100

    print(f"\n=== syllable {TARGET} time share, 3 independent apply_model runs ===")
    print("(values in %, last two columns in percentage points)\n")
    show = tab.copy()
    for c in ["seed0", "seed1", "seed2", "mean", "sd", "original"]:
        show[c] = (show[c] * 100).round(2)
    print(show.round(2).to_string())

    print(f"\nlargest run-to-run spread : {tab['range_pp'].max():.2f} pp "
          f"({tab['range_pp'].idxmax()})")
    print(f"median spread             : {tab['range_pp'].median():.2f} pp")
    print(f"mean within-recording SD  : {tab['sd'].mean()*100:.2f} pp")

    # ── does the BL vs FST effect survive? ───────────────────────────────
    m = tab["mean"].unstack(0) if isinstance(tab.index, pd.MultiIndex) else None
    wide = tab.reset_index().pivot(index="animal", columns="condition", values="mean")
    diff = (wide["FST"] - wide["BL"]) * 100
    t, p = stats.ttest_rel(wide["FST"], wide["BL"])
    dz = diff.mean() / diff.std(ddof=1)

    print(f"\n=== BL vs FST using seed-averaged values (n=4 paired) ===\n")
    out = (wide * 100).round(2)
    out["diff_pp"] = diff.round(2)
    print(out.to_string())
    print(f"\n  BL {wide['BL'].mean()*100:.2f}%  ->  FST {wide['FST'].mean()*100:.2f}%"
          f"   ({wide['FST'].mean()/wide['BL'].mean():.1f}x)")
    print(f"  paired t({len(wide)-1}) = {t:.2f},  p = {p:.4f},  dz = {dz:.2f}")

    # ── the key ratio ────────────────────────────────────────────────────
    effect = diff.mean()
    noise = tab["range_pp"].max()
    print(f"\n=== signal vs inference noise ===\n")
    print(f"  BL -> FST effect          : {effect:.2f} pp")
    print(f"  worst run-to-run spread   : {noise:.2f} pp")
    print(f"  ratio                     : {effect/noise:.1f}x")
    lo_fst, hi_bl = tab.loc["FST", "mean"].min(), tab.loc["BL", "mean"].max()
    print(f"\n  lowest FST animal  : {lo_fst*100:.2f}%")
    print(f"  highest BL animal  : {hi_bl*100:.2f}%")
    print(f"  separation         : {'COMPLETE - no overlap' if lo_fst > hi_bl else 'OVERLAP'}")


if __name__ == "__main__":
    main()
