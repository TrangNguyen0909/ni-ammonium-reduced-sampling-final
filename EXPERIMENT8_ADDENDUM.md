# Experiment 8 – addendum (pre-declared), 4 Nov 2026

**Status at the time of writing.**
- The official v6 run (Experiment 8 as in `EXPERIMENT8_PLAN.md`) has not been executed.
- A fast-mode development run (2 training blocks, few bootstrap replicates) had been executed to test the code. Its numbers suggested that rainfall changed PR-AUC by about +0.005.
- This addendum is therefore **written after seeing preliminary fast-mode results** and is labelled accordingly. Its analyses are reported as sensitivity and exploratory analyses.
- `EXPERIMENT8_PLAN.md` is unchanged and remains the primary plan.

## Analyses added (exactly these five; nothing else)

**(a) Timing assumption of rainfall (sensitivity).** Model `HistGB_rainlag_nb` = own + neighbour + four rainfall features computed from month **t−1** instead of month t:
- r(t−1);
- raindays10mm(t−1);
- r(t−1) minus the expanding same-calendar-month mean up to t₀;
- r(t−1) − r(t₀).

Rationale: rainfall of month t is complete only at the end of month t, and rainfall after the sampling date cannot influence the sampled concentration. Same hyper-parameters, seen design, block retraining and development cut-offs as in the main plan.

**(b) Ablation of `raindays10mm`.** Model `HistGB_rain_noRD_nb` = own + neighbour + `rain_t`, `rain_anom_t`, `rain_change` (no `raindays10_t`). Same protocol.

**(c) Shared rainfall grid cells.** The primary comparison (`HistGB_rain_nb − HistGB_nb`, PR-AUC and network-level recall, 4 visits per year, > 0.3 mg N/L) is repeated with a **grid-cell cluster bootstrap**: whole 5 km cells (187 cells) are resampled, 1,000 replicates, seed 42.

**(d) Wet, normal and dry months (mechanism, no new model).**
- Skipped months are classified by the rainfall of month t relative to the station's own 1995–2007 distribution for that calendar month: lower, middle and upper tercile.
- No 2008–2024 information is used to define the classes.
- Within each class, report PR-AUC and network-level recall of persistence, seasonal climatology, HistGB own, HistGB + nb and HistGB + nb + rain.
- Also report the neighbour benefit (HistGB + nb − HistGB own) with a station bootstrap.

**(e) Threshold-selection table (presentation only).** For persistence, seasonal climatology, HistGB own, HistGB + nb and HistGB + nb + rain at 4/6/8 visits per year and > 0.3 mg N/L, report:
- the development-period recall and false alarms per station-year for every candidate cut-off;
- the selected cut-off;
- the evaluation-period recall and false alarms per station-year at that frozen cut-off.

The main analysis is unchanged.

## Not done (explicitly)

- No hyper-parameter tuning.
- No class weighting. The models are regressions of the log change, so there is no class to weight.
- No other rainfall lags, cumulative windows or interactions.
- No change of the primary model (HistGB + neighbours) or of the primary outcomes.

## Decision rule

The primary model and conclusions remain those of the main analysis. If rainfall (in any of these forms) does not improve recall at the false-alarm budget, with an interval above zero, the manuscript states that rainfall did not improve detection.
