> These are the supplied methods notes. See [code review notes](docs/REVIEW_NOTES.md) for discrepancies requiring author confirmation.

# Methods — computational analysis

> Scope: pose tracking, behavioural segmentation, and how the quantities were computed.
> Wet-lab procedures, syllable annotation and results interpretation are covered elsewhere.

## Data

Recordings are 512 x 384 px H.264 video, 22,232–23,583 frames each. Two mice were tracked
per recording: `track_0` is the subject, introduced first, and `track_1` the partner.
All analyses use the subject mouse only.

Twelve recordings were analysed. Eight (`S232 FST`, `S233-FST`, `S238-FST-rerun`,
`S241-FST`, `S240-FST-Stress`, `S239-FST`, `S282-FST`, `280-FST`) constitute the
CalEx/Control set; four (`S282 BL FST`, `S280 BL FST`, `S232-FST Ctl`,
`S233 FST Ctl`) are the baseline sessions of animals S282, S280, S232 and S233. Group
membership is recorded in `index.csv` within the project directory. Note that "Ctl" in two
baseline filenames denotes the baseline session, not a control group.

Frame rate: `config.yml` records `fps: 30`, while `ffprobe` on the video files reports
36 fps. All durations in this analysis are therefore reported in **frames** rather than
seconds.

## Pose tracking (SLEAP)

DeepLabCut was evaluated first and tracked this dataset poorly, so SLEAP was used.

Training proceeded iteratively. In each round, additional frames were manually annotated
and a model was trained **from random initialisation on the full accumulated label set**
(`base_checkpoint: null`) — not by continuing to train the previous round's weights.
Tracking was inspected on the study videos and failure frames were prioritised for
annotation in the next round. The loop was repeated until tracking was accurate throughout.
Because every round restarted from scratch, only the final label set and training
configuration are needed to reproduce the result.

Annotated frames were drawn from the study recordings themselves, as is standard for
SLEAP. A single final model (`n=562`, below) was used to track all twelve recordings;
inference output was written to two separate project files, `test_25_4_27.slp` for the
eight CalEx / Control recordings and `sleap_track_26_6_7.slp` for the four baseline
recordings. This is a file-organisation split only — both sets were tracked by the same
model with the same skeleton.

### Final training configuration

Reproduced from `training_config.json` (included as `data/sleap_training_config.json`).

```
SLEAP                   1.4.1a2
pipeline                bottom-up (multi_instance head; no centroid stage)
run                     250428_233307.multi_instance.n=562

labelled frames         562 total — 506 training, 56 validation
                        (validation_fraction 0.1, random split)

--- preprocessing ------------------------------------------------------
input_scaling           1.0
resize_and_pad_to_target  512 x 384
pad_to_stride           32
grayscale conversion    none (frames used as stored)

--- backbone -----------------------------------------------------------
architecture            UNet, 1 stack, middle block, bilinear upsampling
max_stride              32
output_stride           4
filters                 16, growth rate 2.0
base_checkpoint         none (trained from random initialisation)

--- heads --------------------------------------------------------------
confidence maps         7 parts, sigma 2.5, output_stride 4
part affinity fields    6 edges (body_center to each other node),
                        sigma 75.0, output_stride 8

--- optimisation -------------------------------------------------------
optimiser               Adam, initial learning rate 1e-4
batch_size              8
batches_per_epoch       200
epochs                  200 (early stopping on plateau, patience 25,
                        min_delta 1e-8)
lr schedule             reduce on plateau, factor 0.5, patience 5,
                        cooldown 3, floor 1e-8
hard keypoint mining    disabled
checkpoint kept         best validation model

--- augmentation (all others disabled) ---------------------------------
rotation                enabled, uniform in [-180, 180] degrees
random flip             enabled, vertical
```

### Track curation

Instance association across frames used SLEAP's default tracker. Automatic tracking
fragmented each recording into multiple short track segments — a median of roughly 11 to
29 per video across the two output projects — rather than two continuous identities.
**Every recording was therefore reviewed manually in the SLEAP GUI and the fragments were
merged across breakpoints**, yielding exactly two continuous tracks per recording
corresponding to the two animals. `track_0` is the subject and `track_1` the partner.

The merged assignment was then verified quantitatively: no frame in any recording showed
both tracks displacing by more than three body lengths while separated by less than three
body lengths, the signature of an identity swap. Subject detection rate after curation was
95.8–99.9% of frames (median 99.6%), with median keypoint confidence 0.91.

Seven keypoints were tracked, in a star topology anchored on the body centre:

```
nodes   nose, left_ear, right_ear, body_center, left_hip, right_hip, tail_base
edges   body_center -> each of the other six
```

Inference produced one `.analysis.h5` per recording containing coordinates
(`tracks`: tracks x 2 x nodes x frames), per-keypoint confidence (`point_scores`),
per-instance confidence, and track occupancy. The parallel `.analysis.csv` export leaves
confidence columns empty, so confidence values were read from the HDF5 files.
keypoint-MoSeq consumes the `.analysis.h5` files directly.

Missing keypoints were filled by linear interpolation.

## Behavioural segmentation (keypoint-MoSeq 0.6.8)

Keypoint sequences were egocentrically aligned, projected by PCA into a low-dimensional
space, and modelled with an autoregressive hidden Markov model, which segments continuous
movement into discrete recurrent syllables. Only the subject mouse was modelled.

Final configuration:

```
use_bodyparts          nose, left_ear, right_ear, body_center,
                       right_hip, left_hip, tail_base
anterior_bodyparts     nose, left_ear, right_ear          (heading initialisation)
posterior_bodyparts    tail_base

PCA_fitting_num_frames 1000000
whiten                 true
latent_dim             4

ar_hypparams           nlags 3, K_0_scale 10.0, S_0_scale 0.01
trans_hypparams        num_states 100, alpha 5.7, gamma 1000.0, kappa 1e4
obs_hypparams          nu_s 5, nu_sigma 1e5, sigmasq_0 0.1, sigmasq_C 0.1
cen_hypparams          sigmasq_loc 0.5

added_noise_level      0.1
conf_threshold         0.5
conf_pseudocount       0.001
error_estimator        intercept 0.25, slope -0.5
fix_heading            false

fitting                50 AR-only iterations, then 500 full-model iterations
                       parallel_message_passing = False
```

Two points are needed to reproduce this exactly:

- **kappa.** The stored `config.yml` records `1e6`, the initialisation value.
  `kpms.update_hypparams(model, kappa=1e4)` was applied after the AR-only stage, so the
  fitted model used **1e4**.
- **latent_dim.** PCA required six components to reach 90% of explained variance;
  `latent_dim` was set to 4 to keep the latent space compact.

`use_bodyparts` lists `right_hip` before `left_hip`, reversing their order in the SLEAP
export. keypoint-MoSeq indexes by name, so this is a consistent permutation and does not
affect modelling, but left and right are transposed in the model's internal pose
representation and laterality should not be read from pose or trajectory plots.

keypoint-MoSeq generates a grid movie of representative clips for each syllable. These were
reviewed and assigned behavioural labels by a collaborator (K.S.); the annotation procedure
is described with the behavioural results.

## Applying the model to the baseline recordings

The baseline recordings were labelled with `kpms.apply_model`, which holds all model
parameters fixed (`states_only=True`) and resamples only the latent state sequences,
for 500 iterations. The model was not refitted: syllable identity is defined by the fitted
AR-HMM parameters, so refitting would produce a different syllable vocabulary with
non-corresponding indices.

Baseline and forced-swim recordings were labelled in a single `apply_model` pass, so both
arms of that comparison share one labelling regime. Values obtained this way are not
interchangeable with the training-time `extract_results` output used for the CalEx/Control
comparison: because states and parameters co-adapt during fitting, `extract_results` labels
sit systematically higher for the syllable of interest on recordings that were in the
training set. Each comparison is therefore quantified end-to-end within one regime, and
absolute values are not compared across the two.

To quantify inference variability, `apply_model` was repeated three times on identical
input with different random seeds (0, 1, 2).

Note for the baseline-versus-forced-swim comparison: its two arms were tracked by
different SLEAP projects (see Pose tracking) and only the forced-swim arm was inside the
keypoint-MoSeq training set. The tracking check above indicates the two SLEAP projects are
geometrically consistent, but the training-set asymmetry remains.

## Quantification

Syllables were included where their pooled bout frequency across recordings exceeded
0.005.

Two quantities are reported, since "syllable frequency" is ambiguous:

- **time share** — fraction of frames assigned to a syllable
- **bout share** — fraction of syllable instances, a bout being one continuous run of a
  single syllable. This is `get_frequencies(..., runlength=True)`, the keypoint-MoSeq
  default.

Their ratio gives **mean bout duration**, in frames.

## Statistics

The allogrooming syllable was designated the primary outcome before the group comparisons
were run, on the basis of prior annotation, and tested without multiple-comparison
correction. All other syllables above threshold were treated as exploratory and corrected
across the family with the Benjamini–Hochberg false-discovery-rate procedure.

- **CalEx vs Control** — between-subject, one recording per animal, n = 4 vs 4.
  Mann–Whitney U, with an unpaired t-test reported alongside.
- **Baseline vs forced swim** — within-animal, n = 4 pairs. Paired two-tailed t-test on
  the within-animal difference.

At these sample sizes non-parametric tests are floor-limited: Mann–Whitney U cannot return
p below 2/C(8,4) = 0.029 for 4 versus 4, and Wilcoxon signed-rank cannot return p below
0.125 for four pairs, irrespective of effect size. Effect sizes and consistency of
direction are reported alongside p-values for this reason.

## Transition analysis

First-order transition matrices were computed per group:

1. each recording's frame-wise syllable sequence was compressed to a **bout sequence** by
   collapsing consecutive identical labels and taking the label at each change point;
2. bigram counts were accumulated over that bout sequence, maximum state index 100;
3. count matrices were summed across recordings within a group, then normalised once by
   the total matrix sum (`normalize="bigram"`);
4. the normalised matrix was subset to the included syllables.

Normalisation precedes subsetting. Node area in the graphs is proportional to that group's
bout share for the syllable (`usage / sum(usage) * 2000 + 1000`).

Graphs were drawn with a circular layout in four variants: all included syllables; two
behaviour-specific variants keeping only edges incident on the syllables of interest; and
a difference graph in which edge colour encodes which group has the higher transition
probability and node ring colour the same for syllable usage. In the behaviour-specific
variants, line widths are normalised against the complete matrix rather than the subset
maximum, so thickness is comparable across all variants. Two line-width scalings were
generated for each graph; the scaling used in each published panel is stated in its
legend.

## Software

```
SLEAP            1.4.1a2
keypoint-moseq   0.6.8
jax              0.6.2
Python           3.12 (Colab, model fitting and apply_model)
                 3.12+ (local quantification; see requirements.txt)
```

## Code availability

Analysis code is in `scripts/` of this repository: `colab_fit_model.py` reproduces the
keypoint-MoSeq fitting and `apply_model` stages with the parameters above, and the
remaining scripts compute every table and figure from the resulting `results.h5` files
(see `README.md`). Because state sequences are inferred by Gibbs sampling, a fresh fit
yields a new syllable vocabulary whose indices do not correspond to those reported here;
reproducing the behavioural labels requires re-annotating the new model's grid movies.
The fitted checkpoint used for the reported results is available from the authors on
request.
