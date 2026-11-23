# Risk-based sampling versus spatial inference — Northern Ireland river NH₄-N

Code, data and results for the manuscript *Risk-based sampling versus spatial inference for detecting river ammonium exceedances under reduced monitoring budgets*.

## Contents

| Path | What it is |
|---|---|
| `Complete_Study_NI_Ammonium_Reduced_Sampling.ipynb` | The complete analysis, executed top to bottom (Restart & Run All) with outputs saved |
| `build_full_notebook.py` | Builds the notebook from the cell sources below and the commentary in `full_notebook_comments.json` |
| `exp4_network_cell.py` … `exp9_robustness_cells.py`, `flow_cells_nb.py`, `fig_distance_cell.py`, `revision_cells.py` | Cell sources used by the builder |
| `EXPERIMENT7_PLAN.md`, `EXPERIMENT8_PLAN.md`, `EXPERIMENT8_ADDENDUM.md`, `EXPERIMENT9_PLAN.md` | Dated analysis plans for the additional analyses (see the manuscript, Section 2.7) |
| `get_haduk_rainfall.py` | Extracts monthly HadUK-Grid rainfall at the station locations |
| `data/raw/` | DAERA open data file, its SHA-256 checksum and source note |
| `data/hydrorivers_ni.json` | HydroRIVERS v1.0 reaches clipped to Northern Ireland |
| `data/haduk_rainfall_stations_monthly.csv` | Monthly rainfall at the stations (output of `get_haduk_rainfall.py`) |
| `results/complete_study/` | All result tables (`T*.csv`), figures (`figures/`), exported data and `manifest.json` |
| `requirements.txt` | Exact package versions |
| `tools/run_notebook.py` | Executes the notebook top to bottom in a fresh kernel |
| `CHANGES.md` | What changed in this revision of the repository |

## Reproducing the results

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name python3
python build_full_notebook.py            # optional: regenerates the notebook from the sources
python tools/run_notebook.py Complete_Study_NI_Ammonium_Reduced_Sampling.ipynb executed.ipynb
```

Run from the repository root (the notebook looks for `data/raw/`). The reported run took 143 minutes on 4 CPU cores (`results/complete_study/manifest.json`); `NB_FAST=1` gives a quick smoke test with fewer blocks and bootstrap replicates (its numbers are not the reported ones).

## Where each manuscript item comes from

| Manuscript | Notebook section | Output |
|---|---|---|
| Table 1 | 4, 5.2 | `T00_*`, `T01_exceedance_counts.csv` |
| Table 2 | 10 (Experiment 1), 14b | `T06_fixed_cutoffs.csv`, `T07_*`, `T23_*` |
| Table 3 | 9 | `T04_threshold_free.csv`, `T05_*`, `T05b_*` |
| Table 4 | 14b (Experiment 6a) | `T23_baselines_climatology_station_risk.csv`, `T23b_*` |
| Table 5 | 14 (Experiment 5c) | `T20_fair_allocation_comparison.csv`, `T20b_*` |
| Table 6 | 14b (Experiment 6b), 14.4 | `T24b_*`, `T24c_*`, `T22_allocation_tradeoffs.csv` |
| Table 7 | 14c, 14d | `T28_alarm_triggered_followup.csv`, `T29*` |
| Tables S2–S3 | 9, 12 | `T05*`, `T12d_*` |
| Table S4 (ranking stability) | 14 | `T19_*`, `T19b_ranking_overlap.csv` |
| Table S5 (sampling effort) | 14.6 | `T32_sampling_effort.csv` |
| Table S6 (basin bootstrap) | 14e | `T30_*` |
| Table S7 (data preparation and cohort) | 15 | `T13_*` |
| Table S8 (detection levels) | 14.6 | `T33_detection_levels.csv`, `T33b_*` |
| Repository tables cited in the text | 14b, 14c, 14d, 14e | `T23c`, `T27`, `T29d–T29g`, `T31` |
| Figures 1–8, S1 | 15b | `figures/Fig01_*` … `Fig08_*`, `FigS1_*` |

Columns named `PR-AUC` in the tables are scikit-learn `average_precision_score` (AP, a step-wise summary), reported as AP in the manuscript. Concentrations are NH₄-N in mg N L⁻¹ (written `mg/l N` in working figures).

## Data licence

DAERA river water quality data: Open Government Licence v3.0 (contains public sector information licensed under the OGL v3.0). HydroRIVERS: HydroSHEDS licence. HadUK-Grid: Open Government Licence (Met Office, via the CEDA Archive).
