# Medial amygdala astrocytes and social behavior

Analysis scripts for the Yu Lab medial amygdala astrocyte project. This collection covers mouse movement tracking and behavioral analysis: **SLEAP** follows body landmarks, and **keypoint-MoSeq** groups movements into recurring units called *syllables*. Researchers then assign behavioral meanings, such as allogrooming, by reviewing video clips.

## Start here

- **To view results:** request the accompanying PDF figures and CSV tables from the authors. They are not included in this code-only upload.
- **To understand the procedure:** read [`METHODS.md`](METHODS.md).

## What is included?

| Folder or file | Contents |
|---|---|
| `scripts/` | All 11 supplied Python scripts; see the guide below |
| `data/` (obtain separately) | Recording/group assignments, saved summary tables, and tracking/model settings |
| `figures/` (obtain separately) | PDF figures supplied with the original analysis |
| `requirements.txt` | Python packages for the local analyses |
| `results/` | New outputs appear here when analyses run |

The comparisons are **baseline (BL) versus forced swim (FST)** in the same four subject mice, and **CalEx versus Control** in separate groups of four subject mice. The mapping tables identify the recordings and animals.

## Which script should I use?

All filenames below are in `scripts/`.

| Purpose | Script |
|---|---|
| Train a movement model and label recordings in Google Colab | `colab_fit_model.py` |
| Optional alternative: label all 12 recordings with the existing model | `colab_apply_all12.py` |
| Repeat labeling with three random seeds in Colab | `colab_seed_stability.py` |
| Measure movement, proximity, and investigation from SLEAP tracks | `subject_metrics.py` |
| Make behavioral pie charts and paired BL/FST comparisons | `syllable_pies.py` |
| Make pie charts in the original Colab style | `pie_charts_colab_format.py` |
| Export time/bout tables and make a keypoint-MoSeq-style plot | `kpms_stats_replica.py` |
| Make the BL/FST line plot from those exported tables | `final_lineplot.py` |
| Plot transitions between behavioral syllables | `transition_graphs.py` |
| Compare CalEx and Control | `calex_vs_control.py` |
| Summarize variation between repeated labeling runs | `analyze_seed_stability.py` |

## What is needed to rerun?

**The archive does not contain raw videos, SLEAP tracking exports, training labels, the fitted checkpoint, or the two model-output HDF5 files.** Obtain the relevant files from the authors before following the full workflow. The original submission also contains summary tables and figures; these are not included in this code-only upload. Once obtained, they can be inspected without Python, and the line plot can be regenerated from the long-form table.

Local analyses use Python 3.12 and the packages in `requirements.txt`. Scripts beginning `colab_` are intended for Google Colab and need the original project files plus updated Google Drive paths. They currently install the latest keypoint-MoSeq, although the supplied methods name version 0.6.8; confirm the environment with the authors before using them.

**Two terms to keep separate:** *time share* is the fraction of video frames assigned to a behavior; *bout share* is the fraction of continuous behavior episodes. Pie charts and the default final line plot show time share. Syllable numbers are specific to the fitted model: a newly trained model needs fresh behavioral annotation.

The original analysis calculations and supplied results are retained. Two scripts now create their output folder automatically. Syntax checks and regeneration of the time-share line plot from the supplied table passed; the full pipeline has not been validated against the missing raw/model files.

## License

The software and associated documentation are available under the [MIT License](LICENSE).

## Running the local scripts

Install Python 3.12. Download this repository with **Code → Download ZIP**, extract it, and open a terminal in that folder. On macOS/Linux:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python scripts/syllable_pies.py
```

On Windows, use `venv\Scripts\pip` and `venv\Scripts\python` instead. Before running, obtain the missing inputs from the authors: `data/groups.csv` and `data/model/results.h5` for BL/FST analyses; `data/groups_calex.csv` and `data/model/results_old_model.h5` for CalEx/Control. Filenames in the group tables must match the recording names inside the HDF5 files.

Run `kpms_stats_replica.py` before `final_lineplot.py`; the first creates the table the second reads. `subject_metrics.py` instead needs curated SLEAP exports in `data/batch_A_BL_Ctl/` and `data/batch_B_FST/`. The local seed check needs `data/seed_test/seed_stability.csv` as well as the BL/FST inputs. New files go into `results/`; the local seed check prints to the terminal.

Confirm the source-video frame rate before using time-dependent movement measurements. The source notes disagree between 30 and 36 fps. The review also identified assumptions about absent syllables, plotting, and model-training tracks that need author confirmation. Original calculations were preserved, except that two scripts now create their output folder automatically.

