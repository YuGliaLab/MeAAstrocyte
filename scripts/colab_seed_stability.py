"""
PASTE THIS INTO COLAB — inference-stability test for apply_model.

Question it answers: when the same old model is applied to the same recordings more
than once, how much do the syllable proportions move? That number tells us whether the
BL-vs-FST effect sits far above inference noise, and it is the error bar reviewers will
ask for.

Runtime: apply_model took ~16 min for 500 iterations on this dataset, so 3 seeds is
~50 min and 5 seeds is ~80 min. Start with 3.

IMPORTANT: this writes nothing into demo_project/ — save_results=False, so the existing
results_mice_track_2.h5 is untouched. Everything lands in one CSV at the end.

On a fresh runtime the install below takes a few minutes. If Colab asks you to restart
the session after it finishes, restart and then re-run this cell — the pip step will be
a no-op the second time.
"""

# ── prerequisites (same as cell 3 of the analysis notebook) ───────────────
get_ipython().system("pip install -q -U keypoint-moseq[cuda]")

from google.colab import drive
drive.mount("/content/drive")

# ══════════════════════════════════════════════════════════════════════════
import os
import numpy as np
import pandas as pd
import jax.random as jr
import keypoint_moseq as kpms

PROJECT_DIR = "/content/drive/MyDrive/demo_project/"
BASE_MODEL = "2025_05_15-04_13_34"          # the old model with the labelled syllables
DATA_DIR = "/content/drive/MyDrive/mice_track_2/videos/"
SEEDS = [0, 1, 2]                            # add 3, 4 if you have the time
TARGET = 5                                   # allogrooming
OUT_CSV = os.path.join(PROJECT_DIR, "seed_stability.csv")

config = lambda: kpms.load_config(PROJECT_DIR)

# Load the keypoints ONCE — identical input for every seed, so any difference in the
# output is inference variability and nothing else.
coordinates, confidences, bodyparts = kpms.load_keypoints(DATA_DIR, "sleap")
data, metadata = kpms.format_data(coordinates, confidences, **config())
print(f"loaded {len(coordinates)} recordings\n")

rows = []
for seed in SEEDS:
    # reload the checkpoint each time so no state leaks between runs
    model = kpms.load_checkpoint(PROJECT_DIR, BASE_MODEL)[0]
    model["seed"] = jr.PRNGKey(seed)

    results = kpms.apply_model(
        model, data, metadata, PROJECT_DIR, BASE_MODEL,
        save_results=False,                  # <- nothing written to Drive
        overwrite=True,
        **config(),
    )

    for rec, r in results.items():
        if not rec.endswith("track0"):       # subject mouse only
            continue
        s = np.asarray(r["syllable"])
        onsets = np.pad(np.diff(s).nonzero()[0] + 1, (1, 0))
        starts = s[onsets]
        n_bouts = int((starts == TARGET).sum())
        n_frames = int((s == TARGET).sum())
        rows.append({
            "seed": seed,
            "recording": rec,
            "time_share": n_frames / len(s),
            "bout_share": n_bouts / len(starts),
            "n_bouts": n_bouts,
            "mean_bout_frames": n_frames / n_bouts if n_bouts else np.nan,
        })
    print(f"seed {seed} done")

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

# ── summary ───────────────────────────────────────────────────────────────
piv = df.pivot_table(index="recording", columns="seed", values="time_share")
piv["mean"] = piv.mean(axis=1)
piv["sd"] = piv[SEEDS].std(axis=1, ddof=1)
piv["range_pp"] = (piv[SEEDS].max(axis=1) - piv[SEEDS].min(axis=1)) * 100

print(f"\n=== syllable {TARGET} time share across seeds ===\n")
print((piv * 100).round(2).to_string())

print(f"\nlargest spread across seeds: {piv['range_pp'].max():.2f} percentage points")
print(f"observed BL -> FST difference: 13.23 percentage points")
print(f"\nIf the spread is well under ~2 pp, inference noise cannot explain the")
print(f"BL/FST effect and the drop from the old extract_results numbers is structural.")
print(f"If the spread approaches 5+ pp, the syllable-5 result needs re-examining.")
print(f"\nSaved {OUT_CSV} — copy it to data/seed_test/ locally for "
      f"scripts/analyze_seed_stability.py.")
