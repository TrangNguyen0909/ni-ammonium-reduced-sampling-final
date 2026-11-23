# Experiment 9 – pre-declared robustness checks (v6.2), 4 Nov 2026

**Status:**
- Written before these analyses were run.
- The primary results (v4/v6.1) are not changed.
- No model, cut-off or setting is re-selected.
- Both analyses are reported whatever their outcome.

## 9a. Basin-level cluster bootstrap

**Question.** Are the main conclusions robust when whole river basins, rather than single stations or years, are resampled?

**Units:** the 40 named primary basins of the cohort (`PrimaryBasin`). Each of the 3 stations with a blank basin is treated as its own unit, giving 43 units.

**Method:** paired cluster bootstrap with 1,000 replicates and seed 42, using the same predictions, cut-offs and definitions as the primary analyses.
- PR-AUC is recomputed on the pooled resample.
- Recall is the ratio of summed detections to summed recorded exceedances.
- The station-bootstrap interval is recomputed with the same code and reported next to it for comparison.

**Comparisons (> 0.3 mg N/L):**
1. One-at-a-time, 8/6/4 visits per year: `HistGB_nb − Persistence` (PR-AUC, network-level recall); `HistGB_nb − Clim_freq_station` (PR-AUC, network-level recall); `HistGB_nb − HistGB_own` (PR-AUC).
2. Network-wide, 6 and 4 visits per year, HistGB + neighbours: staggered − synchronised (PR-AUC, network-level recall).
3. Allocation, budget ≤ 4 visits per station-year:
   - risk-based 56 monthly + 2/yr − uniform staggered, both with HistGB + neighbours (network-level recall);
   - risk-based 56 with observation only − uniform staggered with HistGB + neighbours (network-level recall).

**Decision rule:** a conclusion is described as robust to basin-level dependence if its basin-bootstrap 95% CI excludes 0 in the same direction as the primary result. Otherwise, the manuscript states that the conclusion is not robust at basin level.

## 9b. Hyper-parameter choice evaluated by development-period PR-AUC

**Question.** The configuration was chosen by development-period MAE of log concentration. Would a detection-oriented criterion have chosen a similar configuration?

**Method:** for each of the 6 configurations in `HGB_GRID` and for both HistGB models:
- fit on training pairs before 2005;
- predict the development skipped months (2005–2007) for 8/6/4 visits per year;
- report the development PR-AUC at > 0.3 mg N/L, together with the existing development MAE.

**No evaluation-period (2008–2024) data are used. The selected configuration is not changed.**

**Reporting:** for each model, the rank of the selected configuration by development PR-AUC, and the difference between the selected configuration and the best one.
