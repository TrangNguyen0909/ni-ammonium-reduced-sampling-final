# ==== CELL 1 ==== Experiment 9a - basin-level cluster bootstrap (pre-declared in EXPERIMENT9_PLAN.md); no model or cut-off is changed
# Units: 40 named primary basins; each station with a blank basin is its own unit. Paired bootstrap, CFG["n_boot"] replicates, seed SEED.
bname = info.basin.fillna("").astype(str).str.strip().values
BASIN = np.empty(S, int); _u = {}
for i_, b_ in enumerate(bname):
    key_ = b_ if b_ else f"__blank_{i_}"
    BASIN[i_] = _u.setdefault(key_, len(_u))
NB_UNITS = len(_u); print(f"basin bootstrap units: {NB_UNITS} ({sum(1 for k in _u if not k.startswith('__blank'))} named basins + {sum(1 for k in _u if k.startswith('__blank'))} stations with blank basin)")
def boot_ap_units(IA, a, IB, b, thr, unitA, unitB, K, B=None, seed=SEED):
    # paired cluster bootstrap of PR-AUC(a on IA) - PR-AUC(b on IB); IA and IB may be different item sets (e.g. two schedules)
    B = CFG["n_boot"] if B is None else B; ya, yb_ = labels(IA, thr), labels(IB, thr)
    ma = [np.flatnonzero(unitA == k) for k in range(K)]; mb = [np.flatnonzero(unitB == k) for k in range(K)]
    r = np.random.RandomState(seed); out, sk = [], 0
    for _ in range(B):
        dd = r.randint(0, K, K); ia = np.concatenate([ma[k] for k in dd]); ib = np.concatenate([mb[k] for k in dd])
        if ya[ia].sum() == 0 or yb_[ib].sum() == 0: sk += 1; continue
        out.append(average_precision_score(ya[ia], IA[a].values[ia]) - average_precision_score(yb_[ib], IB[b].values[ib]))
    BOOT_LOG.append({"call": f"9a PR-AUC {a} - {b}", "replicates": B, "discarded": sk})
    return average_precision_score(ya, IA[a]) - average_precision_score(yb_, IB[b]), np.percentile(out, 2.5), np.percentile(out, 97.5)
to_basin = lambda x: np.bincount(BASIN, x, NB_UNITS)                       # station-level sums -> basin-level sums
rows9 = []
def add9(exp, f, comp, metric, est, ci_station, ci_basin):
    rows9.append({"experiment": exp, "visits/yr": f, "comparison": comp, "metric": metric, "difference": est,
                  "station CI low": ci_station[0], "station CI high": ci_station[1], "basin CI low": ci_basin[0], "basin CI high": ci_basin[1],
                  "robust at basin level (CI excludes 0, same sign)": bool((ci_basin[0] > 0 and est > 0) or (ci_basin[1] < 0 and est < 0))})
set_thr(PRIMARY)
# (1) one-at-a-time design
for f in CFG["sched"]:
    IT, RET = pooled(ITEMS, f); us = IT.s.values
    for a_, b_ in (("HistGB_nb", "Persistence"), ("HistGB_nb", "Clim_freq_station"), ("HistGB_nb", "HistGB_own")):
        e, lo, hi = boot_ap_units(IT, a_, IT, b_, PRIMARY, us, us, S); e2, lo2, hi2 = boot_ap_units(IT, a_, IT, b_, PRIMARY, BASIN[us], BASIN[us], NB_UNITS)
        add9("Exp 1 (one-at-a-time)", f, f"{a_} - {b_}", "PR-AUC", e, (lo, hi), (lo2, hi2))
        if b_ == "HistGB_own": continue
        ta, ea = tp_ev(IT, RET, PRIMARY, alarms(IT, f, PRIMARY, a_)); tb, eb_ = tp_ev(IT, RET, PRIMARY, alarms(IT, f, PRIMARY, b_))
        e, lo, hi = boot_ratio_diff(ta, ea, tb, eb_, tag=f"9a station {a_}-{b_} {f}")
        _, lo2, hi2 = boot_ratio_diff(to_basin(ta), to_basin(ea), to_basin(tb), to_basin(eb_), tag=f"9a basin {a_}-{b_} {f}")
        add9("Exp 1 (one-at-a-time)", f, f"{a_} - {b_}", "network-level recall", e, (lo, hi), (lo2, hi2))
# (2) network-wide schedules, HistGB + neighbours
for f in NET_FREQS:
    SA, SB = "network: staggered (graph colouring)", "network: synchronised"
    EA = pd.concat([p[0] for p in NET[(f, SA)]], ignore_index=True); EB = pd.concat([p[0] for p in NET[(f, SB)]], ignore_index=True)
    e, lo, hi = boot_ap_units(EA, "HistGB_nb", EB, "HistGB_nb", PRIMARY, EA.s.values, EB.s.values, S)
    _, lo2, hi2 = boot_ap_units(EA, "HistGB_nb", EB, "HistGB_nb", PRIMARY, BASIN[EA.s.values], BASIN[EB.s.values], NB_UNITS)
    add9("Exp 4 (network-wide)", f, "staggered - synchronised (HistGB_nb)", "PR-AUC", e, (lo, hi), (lo2, hi2))
    oA, EVa, RETa, reva = evaluate_scheme(NET[(f, SA)], f, PRIMARY); oB, EVb, RETb, revb = evaluate_scheme(NET[(f, SB)], f, PRIMARY)
    ta, ea = station_recall(EVa, RETa, reva, PRIMARY, "HistGB_nb"); tb, eb_ = station_recall(EVb, RETb, revb, PRIMARY, "HistGB_nb")
    e, lo, hi = boot_ratio_diff(ta, ea, tb, eb_, tag=f"9a station network {f}")
    _, lo2, hi2 = boot_ratio_diff(to_basin(ta), to_basin(ea), to_basin(tb), to_basin(eb_), tag=f"9a basin network {f}")
    add9("Exp 4 (network-wide)", f, "staggered - synchronised (HistGB_nb)", "network-level recall", e, (lo, hi), (lo2, hi2))
# (3) allocation at the same budget
RN = f"Risk-based: {N_HI[2]} monthly, others 2/yr"
for (na, da), (nb_, db) in (((RN, "HistGB + neighbours"), ("Uniform 4/yr, staggered", "HistGB + neighbours")), ((RN, "Observation only"), ("Uniform 4/yr, staggered", "HistGB + neighbours"))):
    ta, ea = st_counts(na, PRIMARY, da); tb, eb_ = st_counts(nb_, PRIMARY, db)
    e, lo, hi = boot_ratio_diff(ta, ea, tb, eb_, tag=f"9a station alloc {da}")
    _, lo2, hi2 = boot_ratio_diff(to_basin(ta), to_basin(ea), to_basin(tb), to_basin(eb_), tag=f"9a basin alloc {da}")
    add9("Exp 5 (allocation)", 4, f"{na} [{da}] - Uniform 4/yr, staggered [{db}]", "network-level recall", e, (lo, hi), (lo2, hi2))
BASINB = pd.DataFrame(rows9); save_table(BASINB, "T30_basin_bootstrap"); display(BASINB.round(4))
print("conclusions robust at basin level:", int(BASINB.iloc[:, -1].sum()), "of", len(BASINB))

# ==== CELL 2 ==== Experiment 9b - hyper-parameter configurations evaluated by development-period PR-AUC (no evaluation data; choice not changed)
dtr = PAIRS[PAIRS.year < CFG["dev"][0]]; rows9b = []
for n_, cols_ in MODELS.items():
    for ci, cfg in enumerate(HGB_GRID):
        m_ = fit(cols_, cfg, dtr); r_ = {"model": n_, "config": ci, **cfg, "selected": ci == CHOSEN[n_],
                                          "dev MAE (log)": float(TUNE[(TUNE.model == n_) & (TUNE.config == ci)]["dev MAE (log)"].iloc[0])}
        for f in CFG["sched"]:
            DV, _ = pooled(DEV_ITEMS, f); r_[f"dev PR-AUC {f}/yr"] = average_precision_score(labels(DV, PRIMARY), predict(m_, cols_, DV))
        rows9b.append(r_)
HP = pd.DataFrame(rows9b)
for f in CFG["sched"]:
    HP[f"rank by dev PR-AUC {f}/yr"] = HP.groupby("model")[f"dev PR-AUC {f}/yr"].rank(ascending=False, method="min").astype(int)
    HP[f"selected minus best dev PR-AUC {f}/yr"] = HP[f"dev PR-AUC {f}/yr"] - HP.groupby("model")[f"dev PR-AUC {f}/yr"].transform("max")
HP["rank by dev MAE"] = HP.groupby("model")["dev MAE (log)"].rank(method="min").astype(int)
save_table(HP, "T31_hyperparameter_dev_prauc"); display(HP.round(4))
print(HP[HP.selected][["model"] + [c for c in HP.columns if c.startswith("rank") or c.startswith("selected minus")]].round(4).to_string(index=False))
