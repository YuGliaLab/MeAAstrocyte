"""
PASTE INTO COLAB — one apply_model pass over all twelve recordings.

Why: Comparison A (CalEx vs Control) currently uses `extract_results` labels while
Comparison B (baseline vs forced swim) uses `apply_model` labels. The four Control /
forced-swim recordings are shared between them, and the two label sources disagree by
about 10 percentage points on syllable 5. Running every recording through one
`apply_model` pass gives both comparisons a single labelling regime.

The four CalEx recordings have never been through apply_model; the other eight have.

Runtime: ~16-20 min for 500 iterations over 24 tracks.
Writes ONE new file and touches nothing that already exists.
"""

get_ipython().system("pip install -q -U keypoint-moseq[cuda]")

from google.colab import drive
drive.mount("/content/drive")

import os
import numpy as np
import keypoint_moseq as kpms

PROJECT_DIR = "/content/drive/MyDrive/demo_project/"
BASE_MODEL = "2025_05_15-04_13_34"      # the model carrying the annotated syllables
OUT_NAME = "2025_05_15-04_13_34_all12"  # new results folder, nothing overwritten

# The two source directories holding the SLEAP exports. Adjust if the paths differ.
DATA_DIRS = [
    "/content/drive/MyDrive/Mice_track/videos/videos/",   # original 8 (CalEx + Control)
    "/content/drive/MyDrive/mice_track_2/videos/",         # the 8 used for BL vs FST
]

config = lambda: kpms.load_config(PROJECT_DIR)

# ── collect every unique .analysis.h5 across both directories ───────────
files = []
for d in DATA_DIRS:
    if not os.path.isdir(d):
        print(f"!! not found, skipping: {d}")
        continue
    for f in sorted(os.listdir(d)):
        if f.endswith(".analysis.h5"):
            files.append(os.path.join(d, f))

# de-duplicate by basename — the four shared recordings appear in both directories
seen, unique = set(), []
for f in files:
    b = os.path.basename(f)
    if b not in seen:
        seen.add(b)
        unique.append(f)

print(f"\n{len(unique)} unique recordings:")
for f in unique:
    print("  ", os.path.basename(f))

assert len(unique) == 12, f"expected 12 recordings, found {len(unique)} — check DATA_DIRS"

# ── load, apply, save ───────────────────────────────────────────────────
coordinates, confidences, bodyparts = kpms.load_keypoints(unique, "sleap")
data, metadata = kpms.format_data(coordinates, confidences, **config())

model = kpms.load_checkpoint(PROJECT_DIR, BASE_MODEL)[0]

os.makedirs(os.path.join(PROJECT_DIR, OUT_NAME), exist_ok=True)
results = kpms.apply_model(
    model, data, metadata, PROJECT_DIR, BASE_MODEL,
    results_path=os.path.join(PROJECT_DIR, OUT_NAME, "results.h5"),
    save_results=True, overwrite=True, **config(),
)

# ── report ──────────────────────────────────────────────────────────────
n_syll = len(np.unique(np.concatenate([r["syllable"] for r in results.values()])))
print(f"\nDone. {len(results)} tracks, {n_syll} syllables.")
print(f"Saved to {PROJECT_DIR}{OUT_NAME}/results.h5")

print("\nsyllable 5 time share, subject mice:")
for k in sorted(results):
    if k.endswith("track0"):
        s = np.asarray(results[k]["syllable"])
        print(f"  {k[:52]:54s} {100 * (s == 5).mean():6.2f}%")

print("\nCopy that results.h5 to data/model/ locally; both comparisons are then\n"
      "recomputed from it by the analysis scripts.")
