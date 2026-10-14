"""
Comparison A — CalEx vs Control, from the original model's own results.

Reads data/model/results_old_model.h5, i.e. `extract_results` output for model
2025_05_15-04_13_34. Verified to reproduce the original pie charts to within 0.05
percentage points, so this is the same labelling the earlier figures came from.

Design note: unlike the baseline-vs-forced-swim comparison, this contrast is
BETWEEN-subject and UNPAIRED — each animal contributes one recording. n = 4 vs 4.

Usage:
    python scripts/calex_vs_control.py [--syllable 5]
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "data", "model", "results_old_model.h5")
GROUPS = os.path.join(ROOT, "data", "groups_calex.csv")
OUT = os.path.join(ROOT, "results")
MIN_FREQ = 0.005
CTRL, EXP = "Control_Subject", "CalEX_Subject"


def frequencies(seq, num_states, runlength):
    seq = np.asarray(seq).astype(int)
    if runlength:
        seq = seq[np.pad(np.diff(seq).nonzero()[0] + 1, (1, 0))]
    c = np.bincount(seq, minlength=num_states)
    return c / c.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syllable", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    meta = pd.read_csv(GROUPS)
    with h5py.File(RESULTS, "r") as f:
        seqs = {n: f[n]["syllable"][:] for n in meta["name"]}
    num_states = max(s.max() for s in seqs.values()) + 1

    rows = []
    for _, r in meta.iterrows():
        s = seqs[r["name"]]
        inst = frequencies(s, num_states, True)
        frame = frequencies(s, num_states, False)
        onsets = np.pad(np.diff(s.astype(int)).nonzero()[0] + 1, (1, 0))
        for syl in range(num_states):
            if inst[syl] == 0 and frame[syl] == 0:
                continue
            nb = int((s[onsets] == syl).sum())
            rows.append({"animal": r["animal"], "group": r["group"],
                         "role": r["role"], "syllable": syl,
                         "bout_share": inst[syl], "time_share": frame[syl],
                         "n_bouts": nb,
                         "mean_bout_frames": int((s == syl).sum()) / nb if nb else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "calex_syllable_table.csv"), index=False)

    subj = df[df["role"] == "subject"]
    pooled = frequencies(np.concatenate([seqs[n] for n in meta["name"]]),
                         num_states, True)
    keep = sorted(set(np.where(pooled > MIN_FREQ)[0]) & set(subj["syllable"]))

    for metric in ("time_share", "bout_share"):
        mat = (subj.pivot_table(index=["group", "animal"], columns="syllable",
                                values=metric).reindex(columns=keep).fillna(0.0))
        a, b = mat.loc[CTRL], mat.loc[EXP]
        praw, res = [], []
        for s in keep:
            x, y = a[s].values, b[s].values
            _, p_mw = stats.mannwhitneyu(y, x, alternative="two-sided")
            _, p_t = stats.ttest_ind(y, x)
            praw.append(p_mw)
            res.append({"syllable": s, "Control": x.mean(), "CalEX": y.mean(),
                        "diff": y.mean() - x.mean(),
                        "Control_sd": x.std(ddof=1), "CalEX_sd": y.std(ddof=1),
                        "p_mannwhitney": p_mw, "p_ttest": p_t})
        rej, padj, _, _ = multipletests(praw, alpha=0.05, method="fdr_bh")
        t = pd.DataFrame(res)
        t["p_fdr_bh"] = padj
        t["sig_fdr"] = rej
        t.to_csv(os.path.join(OUT, f"calex_stats_{metric}.csv"), index=False)

        pd.set_option("display.width", 200, "display.float_format",
                      lambda v: f"{v:9.4f}")
        print(f"\n{'='*78}\n=== CalEx vs Control — {metric} (subject, n=4 vs 4) ===\n")
        row = t[t["syllable"] == args.syllable]
        if not row.empty:
            r = row.iloc[0]
            print(f"syllable {args.syllable}:  Control {r['Control']*100:.2f}% "
                  f"± {r['Control_sd']*100:.2f}   vs   "
                  f"CalEX {r['CalEX']*100:.2f}% ± {r['CalEX_sd']*100:.2f}")
            print(f"            diff {r['diff']*100:+.2f} pp,  "
                  f"Mann-Whitney p = {r['p_mannwhitney']:.4f},  "
                  f"t-test p = {r['p_ttest']:.4f}")
            print(f"\nper-animal {metric}:")
            print((pd.concat([a[args.syllable], b[args.syllable]]) * 100).round(2).to_string())
        print(f"\nraw p < 0.05     : {t[t.p_mannwhitney < 0.05]['syllable'].tolist()}")
        print(f"survives FDR-BH  : {t[t.sig_fdr]['syllable'].tolist()}")

    print(f"\nsyllables included (pooled bout freq > {MIN_FREQ}): {keep}")
    print(f"\nWrote calex_syllable_table.csv, calex_stats_time_share.csv, "
          f"calex_stats_bout_share.csv to {OUT}")


if __name__ == "__main__":
    main()
