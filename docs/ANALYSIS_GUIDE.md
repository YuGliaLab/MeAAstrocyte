# Behavioural syllable analysis — SLEAP → keypoint-MoSeq

Analysis code and figures for the mouse allogrooming study.

Model fitting and `apply_model` run in Google Colab against
`keypoint-moseq==0.6.8`. The three `colab_` scripts cover upstream fitting and inference. Local downstream scripts do not fit the model and do not require a GPU or JAX. Most need model outputs that are not included. Paths and commands below are relative to the repository root. See [review notes](REVIEW_NOTES.md) for unresolved reproducibility issues.

## Layout

```
METHODS.md              methods text, with the final parameters
requirements.txt        local Python dependencies
scripts/                analysis code (see below)
figures/                generated figures (PDF, vector)
data/                   generated tables (CSV)
```

## Reproducing from raw data

Model outputs are **not distributed** with this repository. The pipeline is reproduced
from the SLEAP exports in three stages:

1. **Pose tracking (SLEAP 1.4.1a2).** Train with `data/sleap_training_config.json`,
   track each video, merge track fragments to two continuous identities in the SLEAP GUI
   (`track_0` = subject), and export one `*.analysis.h5` per recording. The procedure is
   described in `METHODS.md`.
2. **Behavioural segmentation (keypoint-MoSeq 0.6.8, Colab).** Run
   `scripts/colab_fit_model.py`: it writes the parameters in `data/config.yml`, fits the
   model (50 AR-only iterations, then 500 full iterations at kappa = 1e4), extracts
   training-set labels, applies the frozen model to the second recording set in one
   pass, and generates grid movies. Copy the two `results.h5` files it names into
   `data/model/`.
3. **Quantification (this repository, local).** Run the scripts below.

Because keypoint-MoSeq samples state sequences, a fresh fit produces a new syllable
vocabulary: indices will not match the ones in the paper, and syllables must be
re-annotated from their grid movies before any behavioural label is attached. This
repository therefore supports reproduction of the *method* end to end; the specific
checkpoint behind the reported numbers (`2025_05_15-04_13_34`) is available from the
authors on request.

### Files expected by the local scripts

| file | produced by | used by |
|---|---|---|
| `data/model/results.h5` | `apply_model` pass over the 8 baseline / forced-swim recordings | every BL-vs-FST script |
| `data/model/results_old_model.h5` | `extract_results` of the fitted model on its 8 CalEx / Control training recordings | `calex_vs_control.py` |
| `data/batch_A_BL_Ctl/*.h5`, `data/batch_B_FST/*.h5` | SLEAP exports, 4 per condition | `subject_metrics.py` only |
| `data/seed_test/seed_stability.csv` | `colab_seed_stability.py` | `analyze_seed_stability.py` |

`data/config.yml` and `data/sleap_training_config.json` are included for reference; no
local script reads them. Recording-to-group mappings: `data/groups.csv` (BL vs FST,
with animal / condition / role columns) and `data/groups_calex.csv` (CalEx vs Control).

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Scripts

Run from the repository root.

| script | what it does | outputs |
|---|---|---|
| `subject_metrics.py` | Model-free metrics straight from SLEAP keypoints — locomotion, immobility, social proximity, nose-to-partner investigation. Normalised to body length, so camera zoom differences between sessions do not confound. Needs no model output. | `subject_metrics_fps*.csv` |
| `syllable_pies.py` | Per-recording and pooled syllable frequency; per-animal donut charts; paired statistics for every syllable; the syllable-5 paired dot plot. | `syllable_frequencies.csv`, `paired_stats.csv`, `pie_*.pdf`, `syllable5_paired.pdf` |
| `pie_charts_colab_format.py` | Donut charts reproducing the original Colab styling parameter-for-parameter, plus a 2x4 BL/FST contact sheet. | `pie_charts/` |
| `kpms_stats_replica.py` | Faithful replica of `plot_syll_stats_with_sem`, including the bout-share frequency definition and syllable-index ordering. Emits both frequency tables. | `syllable_frequency_wide_{TIME,BOUTS}.csv`, `syllable_stats_kpms_format.pdf` |
| `final_lineplot.py` | The frequency line plot used in the paper: pre-specified syllable 5 marked separately from FDR-corrected exploratory syllables. | `FINAL_lineplot_{time,bouts}.pdf` |
| `transition_graphs.py` | First-order transition matrices and circular transition graphs — complete, allogrooming-focused, self-grooming-focused, and red/blue difference versions, each at two line widths. | `transition_graphs/` |
| `calex_vs_control.py` | Comparison A: CalEx vs Control, between-subject, n = 4 vs 4, from the fitted model's own `extract_results` labels. Mann–Whitney U and unpaired t-test per syllable, FDR-BH across the exploratory family. | `calex_syllable_table.csv`, `calex_stats_{time,bout}_share.csv` |
| `colab_apply_all12.py` | **Runs in Colab, not locally.** Optional: one `apply_model` pass over all twelve recordings, so both comparisons could share a single labelling regime. Not used for the reported numbers; provided so the alternative can be reproduced. | a new `results.h5` under `demo_project/` |
| `colab_fit_model.py` | **Runs in Colab, not locally.** The model-fitting stage: project setup, PCA, AR-only and full-model fitting with the reported hyperparameters, syllable reindexing, `extract_results`, a single `apply_model` pass over the second recording set, grid movies. | checkpoint, two `results.h5`, grid movies under `demo_project/` |
| `colab_seed_stability.py` | **Runs in Colab, not locally.** Repeats `apply_model` under several random seeds to quantify inference variability. Writes nothing into the project directory. | `seed_stability.csv` |
| `analyze_seed_stability.py` | Analyses that CSV: per-recording spread across seeds, and the paired test recomputed on seed-averaged values. | console summary |

Typical order:

```bash
./venv/bin/python scripts/subject_metrics.py --fps 30
./venv/bin/python scripts/syllable_pies.py --syllable 5
./venv/bin/python scripts/pie_charts_colab_format.py
./venv/bin/python scripts/kpms_stats_replica.py
./venv/bin/python scripts/final_lineplot.py
./venv/bin/python scripts/transition_graphs.py
./venv/bin/python scripts/calex_vs_control.py --syllable 5
./venv/bin/python scripts/analyze_seed_stability.py
```

Local file-producing scripts write into `results/`. The seed-stability analysis prints to the console. Colab scripts write to the configured Google Drive project. On Windows, use `venv\Scripts\python` and `venv\Scripts\pip` in place of the Unix environment paths above.

## Shipped tables (`data/`)

The CSVs in `data/` are the script outputs that back the figures and the numbers in
`METHODS.md`, under the names used in the hand-over. Mapping to the generating script:

| file in `data/` | written by | as |
|---|---|---|
| `syllable_full_table_long.csv` | `kpms_stats_replica.py` | `syllable_table_all_videos.csv` |
| `syllable_TIME_share.csv` | `kpms_stats_replica.py` | `syllable_frequency_wide_TIME.csv` |
| `syllable_BOUT_share.csv` | `kpms_stats_replica.py` | `syllable_frequency_wide_BOUTS.csv` |
| `pvalues_kruskal_wallis.csv` | `kpms_stats_replica.py` | `pvalues_kpms.csv` |
| `syllable_bout_duration_frames.csv` | derived from the long table (time share / bout share × total frames / total bouts) | — |
| `paired_stats.csv` | `syllable_pies.py` | same name |
| `subject_metrics_fps30.csv`, `paired_BL_vs_FST.csv` | `subject_metrics.py --fps 30` | same names |
| `transition_strengths.csv` | `transition_graphs.py` | same name |
| `calex_syllable_table.csv`, `calex_stats_{time,bout}_share.csv` | `calex_vs_control.py` | same names |
| `seed_test/seed_stability.csv` | `colab_seed_stability.py` (Colab) | same name |

Figures in `figures/` are the vector versions of the published panels. The scripts also
emit PNG previews and further variants (per-animal donuts, bout-share line plot,
keypoint-MoSeq-style stats plot) that are not included here.

## Notes on the analysis

**Two meanings of "frequency."** Time share is the fraction of frames; bout share is the
fraction of syllable instances, where a bout is one continuous run of a syllable.
keypoint-MoSeq's `get_frequencies` returns bout share by default (`runlength=True`). The
pie charts use time share, the default final line plot uses time share, while `--metric bouts` and the replica plot use bout share, and both tables are exported —
they are not interchangeable.

**One inference pass for both conditions.** Labels from a training checkpoint
(`extract_results`) are not comparable with labels from `apply_model`, because state
sequences are sampled rather than decoded deterministically. BL/FST uses one `apply_model` pass covering its eight recordings. CalEx/Control instead uses the fitted model’s `extract_results` output. Keep these sources separate.

**Transition matrices** are normalised before being subset to the included syllables.
Reversing that order changes the values.

**`--fps`** defaults to 30 but the true frame rate is unconfirmed; syllable bout durations are reported in frames. The model-free metrics do include seconds, minutes, and speed, and require a verified frame rate. METHODS reports a conflicting 36-fps video check.

## Regenerate the line plot from the supplied table

This limited example does not need raw/model HDF5 files. After installing the local dependencies, run these commands from the repository root:

```bash
./venv/bin/python -c "from pathlib import Path; import shutil; Path('results').mkdir(exist_ok=True); shutil.copyfile('data/syllable_full_table_long.csv', 'results/syllable_table_all_videos.csv')"
./venv/bin/python scripts/final_lineplot.py
```

The output is `results/FINAL_lineplot_time.pdf` and a PNG preview. Bootstrap intervals may vary between runs. This does not validate the upstream model or reproduce all results.
