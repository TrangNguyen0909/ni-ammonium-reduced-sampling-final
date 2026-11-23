# Experiment 8 – pre-declared analysis plan: rainfall covariates

Written on 3 Nov 2026, **before any model with rainfall features was fitted or evaluated** (one wording clarification of feature 3 made before implementation: "up to and including t₀" instead of "up to the year of t₀", which would have contradicted "no month after t₀"). The rainfall file had been checked only for completeness: 281 stations × 420 months, no missing values, no duplicates. The v4/v5 results remain the primary results; Experiment 8 is an extension, reported whatever its outcome.

## Data

- **Source:** HadUK-Grid v1.3.2.0 (Met Office), 5 km monthly grids, variables `rainfall` (mm) and `raindays10mm` (days with ≥ 10 mm), 1990–2024. Licence: Open Government Licence.
- **Extraction:** value of the nearest land cell to each station, after conversion from Irish Grid (EPSG:29903) to British National Grid (EPSG:27700); maximum station–cell distance 3.9 km.
- **File:** `data/haduk_rainfall_stations_monthly.csv`.
- **Availability assumption:** monthly rainfall of month t is available at the end of month t. This is consistent with the end-of-month inference of the study; provisional HadUK-Grid values are released within about a week. This is stated as an assumption.

## Features (fixed now; no other rainfall feature will be tried)

For target month t and origin month t₀ of station s, with r = ln(1 + rainfall):

1. `rain_t`: r at month t;
2. `raindays10_t`: days with ≥ 10 mm in month t;
3. `rain_anom_t`: r at month t minus the mean of r for the same calendar month over all months up to and including t₀ (expanding climatology; no month after t₀ is used; missing if no earlier month exists);
4. `rain_change`: r at month t minus r at month t₀.

Neighbour rainfall is **not** used, because 281 stations fall in only 187 grid cells, so it would largely duplicate the station value.

## Models and protocol (identical to Experiment 1)

| Element | Specification |
|---|---|
| New models | `HistGB_rain_own` = own-history features + rainfall; `HistGB_rain_nb` = own-history + neighbour + rainfall features |
| Hyper-parameters | the configuration already selected for HistGB (learning rate 0.1, 400 iterations, 31 leaves, ≥ 50 samples per leaf, L2 1.0). **No re-tuning.** |
| Training | same pairs as Exp. 1 (h = 1–3 months), seen design, 3-year blocks × 10 spatial clusters |
| Development | models trained on pairs before 2005; FA-budget cut-off (≤ 0.1 false alarms per station-year) selected on 2005–2007 and frozen |
| Evaluation | one-at-a-time design, 8/6/4 visits per year, thresholds 0.2/0.3/0.4/0.6 mg N/L, same skipped months and labels |
| Not applied | network-wide and allocation designs (Exp. 4–5); stated as a limitation |

## Outcomes

**Primary (decided now):** ΔPR-AUC `HistGB_rain_nb − HistGB_nb`, 4 visits per year, > 0.3 mg N/L, with two-way (station × year) bootstrap 95% CI (1,000 replicates, seed 42).

**Secondary:**
- the same comparison under station and year bootstrap;
- network-level and skipped-month recall at the FA-budget cut-off (station and year bootstrap);
- `HistGB_rain_own − HistGB_own`;
- `HistGB_rain_nb − seasonal climatology (station)`;
- results at 6 and 8 visits per year and other thresholds;
- results by station group (pre-2008 high-risk, other, emerging).

## Leakage checks (automated)

- Recompute the rainfall features for 300 random skipped months after deleting all rainfall values after month t: they must be identical.
- The climatology component must not change when rainfall after t₀ is deleted.

## Commitments

- The feature set, hyper-parameters, cut-off rule and outcomes above are not changed after results are seen.
- All outcomes are reported.
- Implementation errors are fixed, then recorded in `INCONSISTENCY_LOG.md`.
