"""
Frame-by-frame behavioural metrics for the SUBJECT mouse (track_0) from SLEAP
.analysis.h5 exports.

Model-free: reads keypoints directly, so it needs no keypoint-MoSeq checkpoint.
Everything is reported in body-length units where possible so it is invariant to
camera zoom differences between recording sessions.

Usage:
    python scripts/subject_metrics.py [--fps 30]
"""
import argparse
import glob
import os
import re

import h5py
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# node order as stored in the SLEAP export
NOSE, L_EAR, R_EAR, BODY, L_HIP, R_HIP, TAIL = range(7)

BATCHES = {
    "A_BL_Ctl": os.path.join(ROOT, "data", "batch_A_BL_Ctl"),
    "B_FST": os.path.join(ROOT, "data", "batch_B_FST"),
}

# Condition is a property of the RECORDING, not the animal: every animal has both
# a BL and an FST session (within-subject design, n=4 pairs).
# Confirmed from the user's own interactive_group_setting annotation:
#   BL  = "S282 BL FST", "S280 BL FST", "S232-FST Ctl", "S233 FST Ctl"  (batch A)
#   FST = "S232 FST", "S233-FST", "S282-FST", "280-FST"                 (batch B)
# NB "Ctl" in the batch-A filenames is part of the baseline condition, NOT a
# control group.
CONDITION = {"A_BL_Ctl": "BL", "B_FST": "FST"}


def animal_id(filename):
    """Pull the animal ID out of the inconsistent filenames (e.g. '280-FST' -> S280).

    Strips the SLEAP 'labels.vNNN.NNN_' prefix first, otherwise the video index
    digits (and any '_save_2024-..' timestamp) get picked up instead of the ID.
    """
    name = re.sub(r"^labels\.v\d+\.\d+_", "", os.path.basename(filename))
    m = re.search(r"S(\d{3})", name) or re.match(r"(\d{3})", name)
    return f"S{m.group(1)}" if m else "UNKNOWN"


def load(fp):
    with h5py.File(fp, "r") as f:
        tracks = f["tracks"][:]          # (n_tracks, 2, n_nodes, T)
        scores = f["point_scores"][:]
        video = f["video_path"][()]
    return tracks, scores, (video.decode() if isinstance(video, bytes) else video)


def interpolate_gaps(xy):
    """Linearly fill NaN gaps in a (2, T) array; leaves leading/trailing NaNs."""
    out = xy.copy()
    for d in range(out.shape[0]):
        v = out[d]
        ok = ~np.isnan(v)
        if ok.sum() < 2:
            continue
        idx = np.arange(v.size)
        out[d] = np.interp(idx, idx[ok], v[ok])
    return out


def metrics_for(fp, fps):
    tracks, scores, video = load(fp)
    T = tracks.shape[-1]

    with np.errstate(invalid="ignore"):
        subj = tracks[0]                      # (2, nodes, T) -- track_0 = subject
        part = tracks[1]                      # partner
        c_subj = interpolate_gaps(np.nanmean(subj, axis=1))
        c_part = interpolate_gaps(np.nanmean(part, axis=1))

        # body length = nose -> tail_base, used to normalise for camera zoom
        bodylen = np.nanmedian(np.linalg.norm(subj[:, NOSE, :] - subj[:, TAIL, :], axis=0))

        # --- locomotion -------------------------------------------------
        step = np.linalg.norm(np.diff(c_subj, axis=1), axis=0)     # px / frame
        speed_bl_s = (step / bodylen) * fps                        # bodylengths / s
        # immobility: < 0.25 body lengths per second, smoothed over ~0.5 s
        win = max(1, int(round(0.5 * fps)))
        smooth = pd.Series(speed_bl_s).rolling(win, center=True, min_periods=1).mean().values
        immobile = smooth < 0.25

        # --- social proximity -------------------------------------------
        inter = np.linalg.norm(c_subj - c_part, axis=0) / bodylen   # in body lengths
        close = inter < 2.0

        # nose of subject -> body centre of partner (investigation proxy)
        nose_subj = interpolate_gaps(subj[:, NOSE, :])
        nose_to_partner = np.linalg.norm(nose_subj - c_part, axis=0) / bodylen
        investigating = nose_to_partner < 1.0

        # --- bout structure ---------------------------------------------
        def bouts(mask, min_frames):
            d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
            starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
            keep = (ends - starts) >= min_frames
            return starts[keep], ends[keep]

        cs, ce = bouts(close, int(round(0.5 * fps)))
        ins, ine = bouts(investigating, int(round(0.3 * fps)))

        # keypoint confidence for the subject
        conf = np.nanmean(scores[0])

    return {
        "animal": animal_id(fp),
        "video": os.path.basename(video),
        "frames": T,
        "duration_min": T / fps / 60,
        "bodylen_px": bodylen,
        "mean_conf": conf,
        "speed_bl_per_s": np.nanmean(speed_bl_s),
        "dist_travelled_bl": np.nansum(step) / bodylen,
        "immobile_pct": 100 * np.nanmean(immobile),
        "proximity_pct": 100 * np.nanmean(close),
        "median_inter_bl": np.nanmedian(inter),
        "investigate_pct": 100 * np.nanmean(investigating),
        "n_proximity_bouts": len(cs),
        "mean_prox_bout_s": np.mean((ce - cs) / fps) if len(cs) else 0.0,
        "n_investigate_bouts": len(ins),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=30.0,
                    help="frame rate of the source videos (VERIFY THIS - see README)")
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    missing = [d for d in BATCHES.values() if not glob.glob(os.path.join(d, "*.h5"))]
    if missing:
        raise SystemExit(
            "No SLEAP .analysis.h5 files found in:\n  "
            + "\n  ".join(missing)
            + "\n\nThis script reads the SLEAP exports directly (it does not use the "
              "keypoint-MoSeq results).\nCopy the .analysis.h5 files into those "
              "directories and re-run.")

    rows = []
    for batch, d in BATCHES.items():
        for fp in sorted(glob.glob(os.path.join(d, "*.h5"))):
            r = metrics_for(fp, args.fps)
            r["condition"] = CONDITION[batch]
            rows.append(r)

    df = pd.DataFrame(rows)
    cols = ["condition", "animal", "duration_min", "bodylen_px", "mean_conf",
            "speed_bl_per_s", "dist_travelled_bl", "immobile_pct", "proximity_pct",
            "median_inter_bl", "investigate_pct", "n_proximity_bouts",
            "mean_prox_bout_s", "n_investigate_bouts"]
    df = df[cols].sort_values(["animal", "condition"])

    out = os.path.join(ROOT, "results", f"subject_metrics_fps{int(args.fps)}.csv")
    df.to_csv(out, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 50,
                  "display.float_format", lambda v: f"{v:8.2f}")
    print(f"\n=== Subject (track_0) metrics @ {args.fps:g} fps ===\n")
    print(df.to_string(index=False))

    # ---- within-animal paired comparison, BL -> FST -------------------
    from scipy import stats as st

    metrics = ["immobile_pct", "proximity_pct", "investigate_pct",
               "speed_bl_per_s", "dist_travelled_bl", "median_inter_bl"]
    wide = df.pivot(index="animal", columns="condition", values=metrics)

    print("\n=== Within-animal paired: BL -> FST (n=4 pairs) ===\n")
    res = []
    for m in metrics:
        bl, fst = wide[(m, "BL")], wide[(m, "FST")]
        d = fst - bl
        t, p_t = st.ttest_rel(fst, bl)
        # exact Wilcoxon signed-rank: with n=4 the smallest two-sided p is 0.125
        try:
            w, p_w = st.wilcoxon(fst, bl)
        except ValueError:
            w, p_w = float("nan"), float("nan")
        res.append({
            "metric": m,
            "BL_mean": bl.mean(), "FST_mean": fst.mean(),
            "mean_diff": d.mean(), "sd_diff": d.std(ddof=1),
            "cohen_dz": d.mean() / d.std(ddof=1) if d.std(ddof=1) else float("nan"),
            "consistency": (f"{int((d > 0).sum())}/4 up" if d.mean() > 0
                            else f"{int((d < 0).sum())}/4 down"),
            "paired_t_p": p_t, "wilcoxon_p": p_w,
        })
    rdf = pd.DataFrame(res)
    print(rdf.to_string(index=False))
    rdf.to_csv(os.path.join(ROOT, "results", "paired_BL_vs_FST.csv"), index=False)

    print("\nNOTE  n=4 pairs. Exact Wilcoxon cannot go below p=0.125 two-sided,")
    print("      so read the paired t-test and Cohen's dz, not the Wilcoxon p.")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
