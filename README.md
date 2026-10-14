# Medial amygdala astrocytes and social behavior

Analysis scripts for the Yu Lab medial amygdala astrocyte project. **SLEAP** tracks mouse body landmarks; **keypoint-MoSeq** groups movements into recurring units called *syllables*. Researchers assign behavioral meanings, such as allogrooming, by reviewing video clips.

## Scripts

All 11 Python files are in [`scripts/`](scripts/).

| Purpose | Script |
|---|---|
| Fit a movement model and label recordings in Google Colab | `colab_fit_model.py` |
| Optional labeling of all 12 recordings with an existing model | `colab_apply_all12.py` |
| Repeat labeling with different random seeds in Colab | `colab_seed_stability.py` |
| Measure movement, proximity, and investigation from SLEAP tracks | `subject_metrics.py` |
| Make pie charts and paired baseline/forced-swim comparisons | `syllable_pies.py` |
| Make pie charts in the original Colab style | `pie_charts_colab_format.py` |
| Export behavioral frequency tables and make a comparison plot | `kpms_stats_replica.py` |
| Make the final baseline/forced-swim line plot | `final_lineplot.py` |
| Plot transitions between behavioral syllables | `transition_graphs.py` |
| Compare CalEx and Control groups | `calex_vs_control.py` |
| Summarize variation across repeated labeling runs | `analyze_seed_stability.py` |

## Getting started

Download the repository using **Code → Download ZIP** and extract it. Local scripts use Python 3.12; obtain the dependency list and supporting files from the authors before running. This release contains scripts and documentation only: data tables, figures, settings, detailed methods, raw tracking files, and model outputs are not included.

Run local scripts from the repository folder, for example `python scripts/syllable_pies.py`. Files beginning `colab_` instead run in Google Colab and require updated Google Drive paths and the original project files. Confirm the software environment with the authors: the Colab scripts install an unpinned keypoint-MoSeq version, while the supplied methods name version 0.6.8.

For baseline (BL) versus forced swim (FST), place the group assignments at `data/groups.csv` and applied-model labels at `data/model/results.h5`. For CalEx versus Control, use `data/groups_calex.csv` and `data/model/results_old_model.h5`. Recording names in the tables must match names inside the HDF5 files.

Run `kpms_stats_replica.py` before `final_lineplot.py`, which reads its exported table. `subject_metrics.py` instead needs SLEAP exports in `data/batch_A_BL_Ctl/` and `data/batch_B_FST/`. The local seed analysis also needs `data/seed_test/seed_stability.csv`. Generated files go into `results/`; the local seed analysis prints to the terminal.

## Interpretation and validation

**Time share** means the fraction of video frames assigned to a behavior. **Bout share** means the fraction of continuous behavior episodes. Pie charts and the default final line plot use time share. Syllable numbers belong to a particular fitted model; a new fit requires fresh behavioral annotation.

Confirm the source-video frame rate before using time-dependent movement measures: the source notes disagree between 30 and 36 fps. Review assumptions about absent syllables, plotting, and training tracks before reusing the analyses.

All 11 scripts passed syntax checks. The time-share line plot ran successfully with a separately supplied summary table; the complete pipeline was not validated because raw/model inputs were absent. Original calculations were preserved. Two scripts were adjusted only to create their output folder automatically.

## License

Software and associated documentation: [MIT License](LICENSE), copyright 2026 YuGliaLab.
