"""
PASTE INTO COLAB — fit the keypoint-MoSeq model and apply it to new recordings.

This is the upstream half of the pipeline: SLEAP .analysis.h5 exports in, a fitted
AR-HMM checkpoint and a results.h5 out. Every parameter below is the one recorded in
METHODS.md and data/config.yml. The downstream scripts in this folder read the
results.h5 this produces.

Steps, in the order they were run:
  1. set up the project and write the anatomy / hyperparameters into config.yml
  2. load keypoints, fit PCA, fix latent_dim = 4
  3. initialise the model and run 50 AR-only iterations
  4. set kappa = 1e4 and run 500 full-model iterations
  5. reindex syllables by frequency, extract training-set labels (extract_results)
  6. apply the fitted model, parameters frozen, to a second set of recordings
     (apply_model) — one pass covering every recording in that comparison
  7. generate grid movies for syllable annotation

Reproducibility note: keypoint-MoSeq infers state sequences by Gibbs sampling, so a
fresh fit yields a new syllable vocabulary whose indices do not correspond to the ones
reported in the paper. To reuse the reported labels, syllables in a refitted model must
be re-annotated from their grid movies. The reported numbers come from the checkpoint
named BASE_MODEL below; nothing in this script overwrites an existing checkpoint
unless you set MODEL_NAME to one that exists.

Runtime on a Colab GPU: PCA + 50 AR-only iterations a few minutes; 500 full-model
iterations roughly 30–40 min on 16 tracks; apply_model ~16 min per 16 tracks.
"""

get_ipython().system("pip install -q -U keypoint-moseq[cuda]")

from google.colab import drive
drive.mount("/content/drive")

import os
import numpy as np
import keypoint_moseq as kpms

# ── paths ─────────────────────────────────────────────────────────────────
PROJECT_DIR = "/content/drive/MyDrive/demo_project/"
TRAIN_DATA_DIR = "/content/drive/MyDrive/Mice_track/videos/videos/"   # 8 CalEx / Control
APPLY_DATA_DIR = "/content/drive/MyDrive/mice_track_2/videos/"         # 8 BL / FST
VIDEO_DIR = TRAIN_DATA_DIR                                             # for grid movies

BASE_MODEL = "2025_05_15-04_13_34"      # the checkpoint the paper reports; leave as is
MODEL_NAME = None                        # None -> kpms names the new fit by timestamp
APPLIED_SUFFIX = "_mice_track_2"         # results folder for the apply_model pass

NUM_AR_ITERS = 50
NUM_FULL_ITERS = 500
KAPPA_FULL = 1e4
LATENT_DIM = 4

# ── 1. project setup ──────────────────────────────────────────────────────
if not os.path.exists(os.path.join(PROJECT_DIR, "config.yml")):
    sleap_file = sorted(f for f in os.listdir(TRAIN_DATA_DIR) if f.endswith(".analysis.h5"))[0]
    kpms.setup_project(PROJECT_DIR, sleap_file=os.path.join(TRAIN_DATA_DIR, sleap_file))

kpms.update_config(
    PROJECT_DIR,
    video_dir=VIDEO_DIR,
    use_bodyparts=["nose", "left_ear", "right_ear", "body_center",
                   "right_hip", "left_hip", "tail_base"],
    anterior_bodyparts=["nose", "left_ear", "right_ear"],
    posterior_bodyparts=["tail_base"],
    fps=30,
    # hyperparameters as stored in data/config.yml
    added_noise_level=0.1,
    PCA_fitting_num_frames=1000000,
    conf_threshold=0.5,
    conf_pseudocount=0.001,
    whiten=True,
    fix_heading=False,
    error_estimator={"intercept": 0.25, "slope": -0.5},
    obs_hypparams={"nu_s": 5, "nu_sigma": 1e5, "sigmasq_0": 0.1, "sigmasq_C": 0.1},
    ar_hypparams={"K_0_scale": 10.0, "S_0_scale": 0.01, "latent_dim": LATENT_DIM, "nlags": 3},
    trans_hypparams={"alpha": 5.7, "gamma": 1000.0, "kappa": 1e6, "num_states": 100},
    cen_hypparams={"sigmasq_loc": 0.5},
)
config = lambda: kpms.load_config(PROJECT_DIR)

# ── 2. keypoints and PCA ──────────────────────────────────────────────────
coordinates, confidences, bodyparts = kpms.load_keypoints(TRAIN_DATA_DIR, "sleap")
data, metadata = kpms.format_data(coordinates, confidences, **config())
print(f"training set: {len(coordinates)} tracks")

pca = kpms.fit_pca(**data, **config())
kpms.save_pca(pca, PROJECT_DIR)
kpms.print_dims_to_explain_variance(pca, 0.9)   # 6 components reached 90% in the reported fit
kpms.plot_scree(pca, project_dir=PROJECT_DIR)
kpms.plot_pcs(pca, project_dir=PROJECT_DIR, **config())

kpms.update_config(PROJECT_DIR, latent_dim=LATENT_DIM)

# ── 3. initialise and AR-only stage ───────────────────────────────────────
model = kpms.init_model(data, pca=pca, **config())
model, model_name = kpms.fit_model(
    model, data, metadata, PROJECT_DIR, MODEL_NAME,
    ar_only=True, num_iters=NUM_AR_ITERS,
    parallel_message_passing=False,
)
print(f"AR-only stage done, model_name = {model_name}")

# ── 4. full model with kappa = 1e4 ────────────────────────────────────────
model, data, metadata, current_iter = kpms.load_checkpoint(
    PROJECT_DIR, model_name, iteration=NUM_AR_ITERS)
model = kpms.update_hypparams(model, kappa=KAPPA_FULL)
model = kpms.fit_model(
    model, data, metadata, PROJECT_DIR, model_name,
    ar_only=False, start_iter=current_iter, num_iters=current_iter + NUM_FULL_ITERS,
    parallel_message_passing=False,
)[0]

# ── 5. reindex and extract training-set labels ────────────────────────────
# Syllables are renumbered by descending frequency (0 = most frequent). The reported
# model's indices are frequency-ordered, so this step was applied.
kpms.reindex_syllables_in_checkpoint(PROJECT_DIR, model_name)
model, data, metadata, current_iter = kpms.load_checkpoint(PROJECT_DIR, model_name)
results = kpms.extract_results(model, metadata, PROJECT_DIR, model_name)
kpms.save_results_as_csv(results, PROJECT_DIR, model_name)
n_syll = len(np.unique(np.concatenate([r["syllable"] for r in results.values()])))
print(f"extract_results: {len(results)} tracks, {n_syll} syllables (reported fit: 22)")

# ── 6. apply the frozen model to the second recording set ─────────────────
# One pass over every recording in the comparison, so both conditions share one
# labelling regime. Parameters are held fixed; only state sequences are resampled.
apply_coords, apply_confs, _ = kpms.load_keypoints(APPLY_DATA_DIR, "sleap")
apply_data, apply_meta = kpms.format_data(apply_coords, apply_confs, **config())

applied_name = model_name + APPLIED_SUFFIX
os.makedirs(os.path.join(PROJECT_DIR, applied_name), exist_ok=True)
applied = kpms.apply_model(
    model, apply_data, apply_meta, PROJECT_DIR, model_name,
    results_path=os.path.join(PROJECT_DIR, applied_name, "results.h5"),
    save_results=True, overwrite=True, num_iters=NUM_FULL_ITERS,
    **config(),
)
print(f"apply_model: {len(applied)} tracks -> {applied_name}/results.h5")

# ── 7. grid movies for annotation ─────────────────────────────────────────
kpms.generate_grid_movies(results, PROJECT_DIR, model_name,
                          coordinates=coordinates, **config())

print("\nCopy these to the local repository:")
print(f"  {PROJECT_DIR}{model_name}/results.h5      -> data/model/results_old_model.h5")
print(f"  {PROJECT_DIR}{applied_name}/results.h5    -> data/model/results.h5")
print(f"  {PROJECT_DIR}config.yml                   -> data/config.yml")
