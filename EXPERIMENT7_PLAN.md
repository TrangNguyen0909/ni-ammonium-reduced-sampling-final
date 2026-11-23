# Experiment 7 – pre-declared analysis plan

Written on 3 Nov 2026, **before any Experiment 7 result was computed**. The current v4 results remain the primary results of the study. Experiment 7 is reported as an extension, whatever its outcome.

## 7a. Rolling recalibration of alarm cut-offs

**Motivation.** Cut-offs fixed on 2005–2007 for ≤ 0.1 false alarms per station-year produced only 0.046–0.079 in 2008–2024, because concentrations and reporting practice changed. This is documented in v4.

**Design (one-at-a-time reduction only).**
- For evaluation block B (2008, 2011, …, 2023) and spatial cluster g, the cut-off is chosen by the **same FA-budget rule** (≤ 0.1 false alarms per station-year, lowest such cut-off).
- It is chosen on the skipped-month predictions of **the other clusters in the three years before B** (B−3 … B−1). Under the one-at-a-time design those stations keep monthly sampling, so their outcomes are known at that time. This is the same assumption as the seen training design.
- For B = 2008 the cut-off is the frozen 2005–2007 cut-off, identical to v4.
- Predictions are unchanged (v4 models). Only the cut-off changes.
- Candidate cut-offs: the same grid as Experiment 1 for concentration predictions; 400 quantiles for probability scores.
- Methods: persistence, neighbour-ratio rule, HistGB own, HistGB + neighbours, seasonal climatology (station).
- Visit frequencies: 8, 6, 4 per year. Thresholds: 0.2 and 0.3 mg N/L.

**Not applicable:** network-wide and allocation designs. Uniform network-wide reduction leaves no station sampled monthly, so post-2007 outcomes for recalibration do not exist. Risk-based designs would require additional assumptions. Both are stated as limitations; no surrogate is used.

**Primary outcome (decided now):** network-level detection recall and realised false alarms per station-year of HistGB + neighbours, at 4 visits per year and > 0.3 mg N/L, rolling vs frozen. Paired station and year bootstrap (1,000 replicates, seed 42) of both differences.

**Secondary outcomes:**
- the same comparison for all other methods, frequencies and thresholds;
- skipped-month recall and predictive precision;
- whether the ranking of methods changes (in particular HistGB + nb vs seasonal climatology).

## 7b. Alarm-triggered follow-up sampling

**Rule.**
- When a trigger fires for station s in month t, one **extra visit** is made in month t + 1, provided t + 1 is not already a scheduled visit.
- The trigger is known at the end of month t.
- The extra visit detects a recorded exceedance in month t + 1 if one exists and it was not already detected (observed or alarmed).
- The extra visit does not feed back into later model features. This is a conservative simplification.
- If the record has no value for (s, t + 1), the visit is counted as cost with no detection.

**Triggers compared:**
1. HistGB + neighbours alarm in a skipped month (frozen v4 cut-off);
2. a recorded exceedance observed in a visited month;
3. either trigger;
4. **budget-matched random control:** the same number of extra visits placed at random skipped months of the same rotation (200 random draws, seed 42).

**Designs:** uniform 4/yr staggered and risk-based 56 monthly + 2/yr (the Experiment 5 designs, identical items and alarms). Thresholds 0.2 and 0.3 mg N/L.

**Primary outcome (decided now):** additional recorded exceedances detected per 100 extra visits, trigger 1 vs random control, uniform staggered design, > 0.3 mg N/L.

**Secondary outcomes:**
- network-level recall with follow-up;
- extra visits per station-year (cost);
- the same for triggers 2–3, the risk-based design and 0.2 mg N/L;
- recall gain at the 225 low-risk and the 12 emerging stations.

**Uncertainty:** station bootstrap (1,000 replicates) of the recall gain for each trigger. For the random control, the 2.5–97.5 percentile range across draws.

## Commitments

- No cut-off, budget, rule or design element will be changed after Experiment 7 results are seen.
- All outcomes above are reported, favourable or not.
- If an implementation error is found, it is fixed, and the error and the fix are recorded in `INCONSISTENCY_LOG.md`.
