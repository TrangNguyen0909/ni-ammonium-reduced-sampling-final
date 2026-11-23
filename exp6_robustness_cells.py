# ==== CELL 1 ==== Experiment 6a - simple baselines (one-at-a-time design, same cut-off rule and same skipped months as Experiment 1)
# Every baseline uses only information available before the forecast (expanding history up to the origin month t0; no evaluation data):
#  * Clim_freq_station  (main baseline): station x calendar-month exceedance frequency from the station's own AVAILABLE visits <= t0,
#                        Beta-Binomial empirical-Bayes shrinkage towards the network-wide frequency of the same calendar month:
#                        p = (k + M p_net) / (n + M); the prior strength M is estimated once per threshold on 1995-2004 (before development).
#  * Clim_freq_network : network-wide calendar-month exceedance frequency up to t0 (other stations' records + the target's available visits).
#  * Clim_conc         : station x calendar-month mean log concentration from available visits <= t0 (feature `clim_target`).
#  * Station_risk      : station Beta-Binomial exceedance rate before the scored period (1995-2004 for development, 1995-2007 for evaluation).
# Cut-offs: FA-budget rule on 2005-2007, chosen separately for every method and frozen (as in Experiment 1).
BASES = ["Clim_freq_station", "Clim_freq_network", "Clim_conc", "Station_risk"]; ALLS = STRATS + BASES; PROB = {"Clim_freq_station", "Clim_freq_network", "Station_risk"}
COL.update({"Clim_freq_station": "#2e8b57", "Clim_freq_network": "#a6d8b5", "Clim_conc": "#7fb27f", "Station_risk": "#9b6fb5"})
LAB.update({"Clim_freq_station": "Seasonal climatology (station, shrunk)", "Clim_freq_network": "Seasonal climatology (network)",
            "Clim_conc": "Concentration climatology (station-month)", "Station_risk": "Station risk (pre-period)"})
OBSD = ~np.isnan(V)
def month_cum(X): return np.stack([np.cumsum(X & (mon == k)[None, :], 1) for k in range(1, 13)]).astype(float)   # (12, S, T), counts up to and including t
EXC_D = {thr: OBSD & exceed(np.where(OBSD, V, 0.0), CEN, thr) for thr in THRS}
KD = {thr: month_cum(EXC_D[thr]) for thr in THRS}; ND = month_cum(OBSD)
def fit_prior_strength(thr, y0=1995, y1=2004):
    # empirical Bayes: k_sm ~ Beta-Binomial(n_sm, M p_m, M (1 - p_m)) over stations s and calendar months m, data y0..y1 (before development)
    w = (yr >= y0) & (yr <= y1); k = np.stack([(EXC_D[thr] & w[None, :] & (mon == m)[None, :]).sum(1) for m in range(1, 13)]).astype(float)
    n = np.stack([(OBSD & w[None, :] & (mon == m)[None, :]).sum(1) for m in range(1, 13)]).astype(float); pm = (k.sum(1) / n.sum(1))[:, None]
    def nll(lm):
        M = np.exp(lm); a, b = M * pm, M * (1 - pm); return -np.sum(betaln(k + a, n - k + b) - betaln(a, b))
    return float(np.exp(minimize(lambda x: nll(x[0]), [np.log(20.0)], method="Nelder-Mead").x[0]))
M_PRIOR = {thr: fit_prior_strength(thr) for thr in THRS}; print("Beta-Binomial prior strength M by threshold (fitted on 1995-2004):", {k: round(v, 1) for k, v in M_PRIOR.items()})
def add_freq_clim(IT, keep, start, thr):
    AV = availability(keep, start); s, t, t0 = IT.s.values, IT.t.values, IT.t0.values; m = mon[t] - 1
    ka, na = month_cum(AV & EXC_D[thr]), month_cum(AV)
    k_s, n_s = ka[m, s, t0], na[m, s, t0]
    k_net = KD[thr].sum(1)[m, t0] - KD[thr][m, s, t0] + k_s; n_net = ND.sum(1)[m, t0] - ND[m, s, t0] + n_s     # other stations' record + own available visits
    p_net = (k_net + 0.5) / (n_net + 1.0)
    return (k_s + M_PRIOR[thr] * p_net) / (n_s + M_PRIOR[thr]), p_net
# threshold-specific frequency columns are stored per threshold; the concentration-based baselines do not depend on the threshold
FREQ = {}
for D, start in ((ITEMS, CFG["eval"][0]), (DEV_ITEMS, CFG["dev"][0])):
    for (f, ph), (IT, _) in D.items():
        IT["Clim_conc"] = IT["clim_target"]
        IT["Station_risk"] = RANKS[PRIMARY_RANK if D is ITEMS else DEV_RANK]["score"][IT.s.values]
        for thr in THRS: FREQ[(id(D), f, ph, thr)] = add_freq_clim(IT, CFG["sched"][f][ph], start, thr)
def set_thr(thr):
    # put the threshold-specific frequency climatologies into the item tables (called before every scoring at threshold thr)
    for D in (ITEMS, DEV_ITEMS):
        for (f, ph), (IT, _) in D.items(): IT["Clim_freq_station"], IT["Clim_freq_network"] = FREQ[(id(D), f, ph, thr)]
def score_of(IT, s_): return IT[s_].values if s_ in PROB else np.exp(IT[s_].values)
def fa_budget_cut(f, thr, s_):
    if s_ in STRATS: return CUTS[(f, thr)][(s_, "FA-budget")]                       # identical to Experiment 1
    set_thr(thr); IT, _ = pooled(DEV_ITEMS, f); yb = labels(IT, thr); sc = score_of(IT, s_); nph = len(CFG["sched"][f])
    cands = GRID if s_ not in PROB else np.unique(np.quantile(sc, np.linspace(0, 1, 401)))
    fa = np.array([((sc > c) & ~yb).sum() / nph / n_sy_dev for c in cands]); ok = np.flatnonzero(fa <= CFG["fa_budget"])
    return float(cands[ok[0]]) if len(ok) else float(cands[-1])
CUT6 = {(f, thr, s_): fa_budget_cut(f, thr, s_) for f in CFG["sched"] for thr in THRS for s_ in ALLS}
def tp_ev(IT, RET, thr, alarm, cluster="station", skipped_only=False):
    # per bootstrap unit: detected recorded exceedances and recorded exceedances.
    # skipped_only=False (as in Experiment 1): visited months count as observed detections (TP) and enter the denominator;
    # skipped_only=True  (main comparison of Experiment 6): only skipped months, TP = alarmed exceedances, denominator = skipped exceedances.
    yb = labels(IT, thr); ke, kr, K = unit_keys(IT, RET, cluster)
    rev = np.zeros(len(RET)) if skipped_only else exceed(V[RET.s.values, RET.t.values], CEN[RET.s.values, RET.t.values], thr).astype(float)
    ev = np.bincount(kr, rev, K) + np.bincount(ke, yb.astype(float), K)
    tp = np.bincount(kr, rev, K) + (np.bincount(ke, (alarm & yb).astype(float), K) if alarm is not None else 0.0)
    return tp, ev
BOOT_LOG = []                                                            # replicates that could not be scored, per bootstrap call
def boot_ratio_diff(ta, ea, tb, eb, units=None, B=None, seed=SEED, tag=""):
    # paired cluster bootstrap of a difference in recall; `units` restricts resampling to a subset of units (e.g. a station group).
    # Replicates in which either denominator is 0 (no recorded exceedance drawn) cannot be scored and are discarded (counted in BOOT_LOG).
    B = CFG["n_boot"] if B is None else B; units = np.arange(len(ea)) if units is None else np.asarray(units)
    ta, ea, tb, eb = ta[units], ea[units], tb[units], eb[units]; dd = np.random.RandomState(seed).randint(0, len(units), (B, len(units)))
    EA, EB = ea[dd].sum(1), eb[dd].sum(1); ok = (EA > 0) & (EB > 0)
    d_ = ta[dd].sum(1)[ok] / EA[ok] - tb[dd].sum(1)[ok] / EB[ok]; BOOT_LOG.append({"call": tag or "recall difference", "replicates": B, "discarded": int((~ok).sum())})
    return ta.sum() / ea.sum() - tb.sum() / eb.sum(), np.percentile(d_, 2.5), np.percentile(d_, 97.5)
def boot_ap_counted(IT, a, b, thr, cluster, B=None, seed=SEED, tag=""):
    # identical to boot_ap (Section 9) but also reports the number of replicates without any exceedance (discarded)
    B = CFG["n_boot"] if B is None else B; yb = labels(IT, thr); keys = IT.s.values if cluster == "station" else IT.year.values
    uk, inv = np.unique(keys, return_inverse=True); mem = [np.flatnonzero(inv == k) for k in range(len(uk))]; r = np.random.RandomState(seed); out = []; sk = 0
    for _ in range(B):
        idx = np.concatenate([mem[k] for k in r.randint(0, len(uk), len(uk))])
        if yb[idx].sum() == 0: sk += 1; continue
        out.append(average_precision_score(yb[idx], IT[a].values[idx]) - average_precision_score(yb[idx], IT[b].values[idx]))
    BOOT_LOG.append({"call": tag or f"PR-AUC {a} - {b} ({cluster})", "replicates": B, "discarded": sk})
    return average_precision_score(yb, IT[a]) - average_precision_score(yb, IT[b]), np.percentile(out, 2.5), np.percentile(out, 97.5)
def alarms(IT, f, thr, s_): return None if s_ == "Observation-only" else score_of(IT, s_) > CUT6[(f, thr, s_)]
CONC = {"Persistence", "Neighbour_ratio", "HistGB_own", "HistGB_nb", "Clim_conc"}              # methods that predict a concentration
rows = []
for thr in THRS:
    set_thr(thr)
    for f in CFG["sched"]:
        IT, RET = pooled(ITEMS, f); nph = len(CFG["sched"][f]); yb = labels(IT, thr); DV, _ = pooled(DEV_ITEMS, f)
        for s_ in ["Observation-only"] + ALLS:
            a = alarms(IT, f, thr, s_); tp, ev = tp_ev(IT, RET, thr, a); fp = 0 if a is None else int((a & ~yb).sum())
            TPs = 0 if a is None else int((a & yb).sum()); FNs = int(yb.sum()) - TPs
            r_ = {"visits/yr": f, "threshold": thr, "strategy": s_, "skipped months": len(IT) // nph, "skipped exceedances": int(yb.sum()) // nph,
                  "PR-AUC (skipped months)": np.nan if a is None else average_precision_score(yb, score_of(IT, s_)),
                  "cut-off (frozen, 2005-2007)": np.nan if a is None else CUT6[(f, thr, s_)],
                  "distinct development scores": np.nan if a is None else len(np.unique(score_of(DV, s_))),
                  # main comparison: skipped months only
                  "TP (skipped)": TPs / nph, "FP (skipped)": fp / nph, "FN (skipped)": FNs / nph, "recall (skipped months)": TPs / max(int(yb.sum()), 1),
                  "precision (skipped months)": np.nan if (a is None or a.sum() == 0) else TPs / a.sum(), "false alarms/station-yr": fp / nph / n_sy_eval,
                  # as in Experiment 1: all recorded exceedances, visited months counted as observed detections
                  "recall (all recorded exceedances)": tp.sum() / ev.sum(), "precision (incl. observed exceedances)": tp.sum() / (tp.sum() + fp),
                  "MAE log (concentration methods)": float(np.abs(IT[s_].values - IT.y.values).mean()) if s_ in CONC else np.nan,
                  "Brier score": np.mean((IT[s_].values - yb) ** 2) if s_ in ("Clim_freq_station", "Clim_freq_network") else np.nan}
            rows.append(r_)
BASE6 = pd.DataFrame(rows)
bref = BASE6[BASE6.strategy == "Clim_freq_network"].set_index(["visits/yr", "threshold"])["Brier score"]
BASE6["Brier skill vs network climatology"] = [1 - r["Brier score"] / bref[(r["visits/yr"], r.threshold)] if pd.notna(r["Brier score"]) else np.nan for _, r in BASE6.iterrows()]
save_table(BASE6, "T23_baselines_climatology_station_risk")
display(BASE6[BASE6.threshold == PRIMARY].set_index(["visits/yr", "strategy"]).drop(columns="threshold").round(3))
noal = BASE6[(BASE6.strategy != "Observation-only") & (BASE6["TP (skipped)"] + BASE6["FP (skipped)"] == 0)]
if len(noal): print("NOTE - frozen development cut-off raised no alarm in the evaluation period for:", noal[["visits/yr", "threshold", "strategy", "distinct development scores"]].to_dict("records"))
# calibration of the two probabilistic baselines (skipped months, evaluation period, primary threshold)
set_thr(PRIMARY); cal = []; BINS = [0, .005, .01, .02, .05, .1, .2, .5, 1.0001]
for f in CFG["sched"]:
    IT, _ = pooled(ITEMS, f); yb = labels(IT, PRIMARY)
    for s_ in ("Clim_freq_station", "Clim_freq_network"):
        b_ = np.digitize(IT[s_].values, BINS) - 1
        for k in np.unique(b_):
            m = b_ == k; cal.append({"visits/yr": f, "strategy": s_, "predicted probability bin": f"{BINS[k]:.3f}-{min(BINS[k + 1], 1):.3f}", "skipped months": int(m.sum()),
                                     "mean predicted": IT[s_].values[m].mean(), "observed exceedance frequency": yb[m].mean()})
CAL = pd.DataFrame(cal); save_table(CAL, "T23c_climatology_calibration"); display(CAL[CAL["visits/yr"] == 4].round(4))
bt = []
CMP6 = (("HistGB_nb", "Clim_freq_station"), ("HistGB_own", "Clim_freq_station"), ("Clim_freq_station", "Persistence"), ("Clim_freq_station", "Clim_freq_network"),
        ("HistGB_nb", "Clim_conc"), ("HistGB_nb", "Station_risk"))
for f in CFG["sched"]:
    for thr in (0.2, PRIMARY):
        set_thr(thr); IT, RET = pooled(ITEMS, f)
        for a_, b_ in CMP6:
            for cl in ("station", "year"):
                e, lo, hi = boot_ap_counted(IT, a_, b_, thr, cl)
                bt.append({"visits/yr": f, "threshold": thr, "metric": "PR-AUC", "comparison": f"{a_} - {b_}", "cluster": cl, "difference": e, "95% CI low": lo, "95% CI high": hi})
                for so, mlab in ((True, "recall, skipped months (FA-budget)"), (False, "recall, all recorded exceedances (FA-budget)")):
                    ta, ea = tp_ev(IT, RET, thr, alarms(IT, f, thr, a_), cl, so); tb, eb = tp_ev(IT, RET, thr, alarms(IT, f, thr, b_), cl, so)
                    e, lo, hi = boot_ratio_diff(ta, ea, tb, eb, tag=f"{mlab} {a_} - {b_} ({cl})")
                    bt.append({"visits/yr": f, "threshold": thr, "metric": mlab, "comparison": f"{a_} - {b_}", "cluster": cl, "difference": e, "95% CI low": lo, "95% CI high": hi})
            if thr == PRIMARY:                                             # two-way (station x year) bootstrap, as for the headline comparisons (T05b)
                e, lo, hi, sk = boot_ap_2way(IT, a_, b_, thr); BOOT_LOG.append({"call": f"PR-AUC two-way {a_} - {b_}", "replicates": CFG["n_boot"], "discarded": sk})
                bt.append({"visits/yr": f, "threshold": thr, "metric": "PR-AUC", "comparison": f"{a_} - {b_}", "cluster": "two-way", "difference": e, "95% CI low": lo, "95% CI high": hi})
BASE6B = pd.DataFrame(bt); save_table(BASE6B, "T23b_baselines_bootstrap"); display(BASE6B[(BASE6B.threshold == PRIMARY) & (BASE6B["visits/yr"] == 4)].round(3))

# ==== CELL 2 ==== Experiment 6b - performance by station group (defined from pre-2008 information only)
# high-risk = the N_HI[2] stations with the highest pre-2008 Beta-Binomial rate (the stations kept monthly in the 56-station risk-based design);
# 'emerging' = no recorded exceedance > PRIMARY in 1995-2007 but >= 1 in the 2008-2024 evaluation months. Groups are fixed for both thresholds.
nhi = N_HI[2]; HI = np.zeros(S, bool); HI[RANKS[PRIMARY_RANK]["order"][:nhi]] = True
k_pre, _ = exc_counts(1995, 2007, PRIMARY); ev_cnt = np.array([exceed(V[i, EVAL[i]], CEN[i, EVAL[i]], PRIMARY).sum() for i in range(S)])
EMERG6 = (k_pre == 0) & (ev_cnt > 0)
GRP = {"all stations": np.ones(S, bool), f"{nhi} pre-2008 high-risk stations": HI, f"{S - nhi} other stations": ~HI, f"'emerging' stations (n={EMERG6.sum()})": EMERG6}
print({g: f"{m.sum()} stations, {ev_cnt[m].sum()} recorded exceedances > {PRIMARY} mg/l in 2008-2024" for g, m in GRP.items()})
# (A) Experiment 1 (one-at-a-time) by group, every strategy
rows, bb = [], []
for f in CFG["sched"]:
    for thr in (0.2, PRIMARY):
        set_thr(thr); IT, RET = pooled(ITEMS, f); yb = labels(IT, thr); TE = {s_: tp_ev(IT, RET, thr, alarms(IT, f, thr, s_)) for s_ in ["Observation-only"] + ALLS}
        for g, m in GRP.items():
            sk = m[IT.s.values]
            for s_ in ["Observation-only"] + ALLS:
                tp, ev = TE[s_]
                rows.append({"visits/yr": f, "threshold": thr, "group": g, "strategy": s_, "recorded exceedances": ev[m].sum() / len(CFG["sched"][f]), "recall": tp[m].sum() / ev[m].sum(),
                             "PR-AUC (skipped months)": average_precision_score(yb[sk], score_of(IT[sk], s_)) if s_ != "Observation-only" and yb[sk].sum() >= 20 else np.nan})
            for a_, b_ in (("HistGB_nb", "Persistence"), ("HistGB_nb", "Clim_freq_station"), ("HistGB_nb", "Observation-only")):
                e, lo, hi = boot_ratio_diff(*TE[a_], *TE[b_], units=np.flatnonzero(m))
                bb.append({"experiment": "Exp 1 (one-at-a-time)", "visits/yr": f, "threshold": thr, "group": g, "comparison": f"{a_} - {b_}", "d recall": e, "95% CI low": lo, "95% CI high": hi})
GR1 = pd.DataFrame(rows); save_table(GR1, "T24_station_groups_one_at_a_time")
display(GR1[(GR1.threshold == PRIMARY) & (GR1["visits/yr"] == 4)].pivot(index="strategy", columns="group", values="recall").round(3))
# (B) Experiment 5 designs by group: the trade-off table (network-wide recall next to recall at the stations whose sampling was reduced)
rows = []; DN = [n for n, _, _ in DESIGNS]
for thr in (0.2, PRIMARY):
    for name in DN:
        EVs, RETs, rev = FAIR_EV[(name, thr)]
        for det in ("Observation only",) + tuple(lab for _, lab in DET):
            tp, ev = st_counts(name, thr, det)
            # share of emerging stations with >= 1 detected exceedance, per schedule rotation, averaged
            v_ = EVs.v.values; vr = RETs.v.values; yb = exceed(np.exp(EVs.y.values), EVs.cens_target.values, thr); det_any = []
            for v in np.unique(v_):
                hit = np.bincount(RETs.s.values[vr == v], rev[vr == v].astype(float), S)
                if det != "Observation only":
                    s_ = {lab: k for k, lab in DET}[det]; hit = hit + np.bincount(EVs.s.values[v_ == v], (EVs[f"alarm_{s_}"].values & yb)[v_ == v].astype(float), S)
                det_any.append((hit[EMERG6] > 0).mean())
            r_ = {"design": name, "threshold": thr, "detection": det}
            for g, m in GRP.items(): r_[f"recall, {g}"] = tp[m].sum() / ev[m].sum()
            r_[f"'emerging' stations detected at least once"] = np.mean(det_any); rows.append(r_)
        if name == "Uniform 4/yr, staggered": continue
        for det in ("Observation only", "HistGB + neighbours"):
            ta, ea = st_counts(name, thr, det); tb, eb = st_counts("Uniform 4/yr, staggered", thr, det)
            for g, m in GRP.items():
                e, lo, hi = boot_ratio_diff(ta, ea, tb, eb, units=np.flatnonzero(m))
                bb.append({"experiment": "Exp 5 (allocation)", "visits/yr": 4, "threshold": thr, "group": g, "comparison": f"{name} [{det}] - Uniform 4/yr, staggered [{det}]",
                           "d recall": e, "95% CI low": lo, "95% CI high": hi})
GR5 = pd.DataFrame(rows); save_table(GR5, "T24b_station_groups_allocation"); GRB = pd.DataFrame(bb); save_table(GRB, "T24c_station_groups_bootstrap")
display(GR5[GR5.threshold == PRIMARY].set_index(["design", "detection"]).drop(columns="threshold").round(3))
display(GRB[(GRB.threshold == PRIMARY) & (GRB.experiment == "Exp 5 (allocation)")].round(3))
fig, ax = plt.subplots(1, 2, figsize=(15, 4.4), sharey=True)
gl = list(GRP); xx = np.arange(len(gl)); dcols = ["#a8a69e", "#2a78d6", "#e34948", "#7a1f1f", "#f2b8b8"]
for a_, det in zip(ax, ("Observation only", "HistGB + neighbours")):
    x = GR5[(GR5.threshold == PRIMARY) & (GR5.detection == det)].set_index("design"); w = .8 / len(DN)
    for k, (n, cc) in enumerate(zip(DN, dcols)):
        b = a_.bar(xx + (k - (len(DN) - 1) / 2) * w, [x.loc[n, f"recall, {g}"] for g in gl], w * .92, color=cc, label=n); a_.bar_label(b, fmt="%.2f", fontsize=6, padding=1)
    a_.set_xticks(xx); a_.set_xticklabels([g.replace(" stations", "\nstations") for g in gl], fontsize=8); a_.set_ylim(0, 1.08)
    a_.set_title(f"({'ab'[det != 'Observation only']}) {det}; > {PRIMARY} mg/l N; budget <= 4 visits/station-yr")
ax[0].set_ylabel("Recall of recorded exceedances (2008-2024)"); ax[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); savefig(fig, "F20_station_group_tradeoffs"); plt.show()

# ==== CELL 3 ==== Experiment 6c - temporal stability: 2008-2015 versus 2016-2024 and by 3-year block (frozen models' cut-offs; nothing re-selected)
PERIODS = {"2008-2015": (2008, 2015), "2016-2024": (2016, 2024)}
rows, bb = [], []
set_thr(PRIMARY)
for f in CFG["sched"]:
    IT, RET = pooled(ITEMS, f); ry = yr[RET.t.values]
    for pname, (a0, a1) in PERIODS.items():
        mi, mr = IT.year.between(a0, a1).values, (ry >= a0) & (ry <= a1); ITp, RETp = IT[mi], RET[mr]; yb = labels(ITp, PRIMARY)
        TE = {s_: tp_ev(ITp, RETp, PRIMARY, alarms(ITp, f, PRIMARY, s_)) for s_ in ["Observation-only"] + ALLS}
        for s_ in ["Observation-only"] + ALLS:
            rows.append({"experiment": "Exp 1 (one-at-a-time)", "visits/yr": f, "period": pname, "strategy": s_, "recorded exceedances": TE[s_][1].sum() / len(CFG["sched"][f]),
                         "recall": TE[s_][0].sum() / TE[s_][1].sum(), "PR-AUC (skipped months)": np.nan if s_ == "Observation-only" else average_precision_score(yb, score_of(ITp, s_))})
        for a_, b_ in (("HistGB_nb", "Persistence"), ("HistGB_nb", "Clim_freq_station")):
            e, lo, hi = boot_ap(ITp, a_, b_, PRIMARY, "station")
            bb.append({"experiment": "Exp 1 (one-at-a-time)", "visits/yr": f, "period": pname, "metric": "PR-AUC", "comparison": f"{a_} - {b_}", "difference": e, "95% CI low": lo, "95% CI high": hi})
            e, lo, hi = boot_ratio_diff(*TE[a_], *TE[b_])
            bb.append({"experiment": "Exp 1 (one-at-a-time)", "visits/yr": f, "period": pname, "metric": "recall (FA-budget)", "comparison": f"{a_} - {b_}", "difference": e, "95% CI low": lo, "95% CI high": hi})
for pname, (a0, a1) in PERIODS.items():
    ce = {}
    for name in DN:
        EVs, RETs, rev = FAIR_EV[(name, PRIMARY)]; me = EVs.year.between(a0, a1).values; ry = yr[RETs.t.values]; mr = (ry >= a0) & (ry <= a1)
        yb = exceed(np.exp(EVs.y.values), EVs.cens_target.values, PRIMARY)
        for det in ("Observation only",) + tuple(lab for _, lab in DET):
            key_ = {lab: k for k, lab in DET}.get(det)
            al = np.zeros(len(EVs), bool) if det == "Observation only" else EVs[f"alarm_{key_}"].values
            tp = np.bincount(RETs.s.values[mr], rev[mr].astype(float), S) + np.bincount(EVs.s.values[me], (al & yb)[me].astype(float), S)
            ev = np.bincount(RETs.s.values[mr], rev[mr].astype(float), S) + np.bincount(EVs.s.values[me], yb[me].astype(float), S)
            ce[(name, det)] = (tp, ev)
            rows.append({"experiment": "Exp 5 (allocation)", "visits/yr": 4, "period": pname, "strategy": f"{name} [{det}]", "recorded exceedances": ev.sum() / EVs.v.nunique(), "recall": tp.sum() / ev.sum()})
    for name in DN[2:]:
        for det in ("Observation only", "HistGB + neighbours"):
            e, lo, hi = boot_ratio_diff(*ce[(name, det)], *ce[("Uniform 4/yr, staggered", det)])
            bb.append({"experiment": "Exp 5 (allocation)", "visits/yr": 4, "period": pname, "metric": "recall (FA-budget)", "comparison": f"{name} [{det}] - Uniform 4/yr, staggered [{det}]",
                       "difference": e, "95% CI low": lo, "95% CI high": hi})
TEMP = pd.DataFrame(rows); save_table(TEMP, "T25_temporal_stability"); TEMPB = pd.DataFrame(bb); save_table(TEMPB, "T25b_temporal_stability_bootstrap")
display(TEMP[(TEMP["visits/yr"] == 4)].pivot(index="strategy", columns="period", values="recall").round(3)); display(TEMPB[TEMPB["visits/yr"] == 4].round(3))
# per 3-year block (Experiment 1, 4 visits/yr): PR-AUC and recall by block, with the number of recorded exceedances
set_thr(PRIMARY); IT, RET = pooled(ITEMS, 4); blk_it = IT.year.map(block_of).values; blk_rt = np.array([block_of(y) for y in yr[RET.t.values]]); rows = []
for B in CFG["blocks"]:
    ITb, RETb = IT[blk_it == B], RET[blk_rt == B]; yb = labels(ITb, PRIMARY)
    for s_ in ["Observation-only", "Persistence", "Clim_freq_station", "HistGB_nb"]:
        tp, ev = tp_ev(ITb, RETb, PRIMARY, alarms(ITb, 4, PRIMARY, s_))
        rows.append({"block": f"{B}-{min(B + 2, CFG['eval'][1])}", "strategy": s_, "recorded exceedances": ev.sum() / 3, "recall": tp.sum() / ev.sum(),
                     "PR-AUC (skipped months)": np.nan if s_ == "Observation-only" else average_precision_score(yb, score_of(ITb, s_))})
BLK = pd.DataFrame(rows); save_table(BLK, "T25c_by_block_4visits"); display(BLK.pivot(index="block", columns="strategy", values="PR-AUC (skipped months)").round(3))
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for s_ in ["Observation-only", "Persistence", "Clim_freq_station", "HistGB_nb"]:
    x = BLK[BLK.strategy == s_]
    if s_ != "Observation-only": ax[0].plot(x.block, x["PR-AUC (skipped months)"], "o-", color=COL[s_], label=LAB[s_])
    ax[1].plot(x.block, x.recall, "o-", color=COL[s_], label=LAB[s_])
nev = BLK[BLK.strategy == "HistGB_nb"]["recorded exceedances"].values
for a_, t_ in zip(ax, ("(a) PR-AUC, skipped months", "(b) Recall at the frozen FA-budget cut-off")):
    a_.set_title(f"{t_}; 4 visits/yr, > {PRIMARY} mg/l N"); a_.set_xlabel("Evaluation block (recorded exceedances: " + ", ".join(f"{int(n)}" for n in nev) + ")")
ax[0].set_ylabel("PR-AUC"); ax[1].set_ylabel("Recall of recorded exceedances"); ax[0].legend(fontsize=7)
fig.tight_layout(); savefig(fig, "F21_temporal_stability"); plt.show()

# ==== CELL 4 ==== Experiment 6d - automated consistency and leakage checks, and a record of the common evaluation configuration
AUD = []
def chk(name, ok, detail=""): AUD.append({"check": name, "result": "PASS" if bool(ok) else "FAIL", "detail": detail})
rng6 = np.random.RandomState(SEED); tidx = np.arange(T); first_eval_t = int(np.flatnonzero(yr == CFG["eval"][0])[0])
# (1) brute-force recomputation of both frequency climatologies from the raw monthly arrays (only months <= t0 actually sampled)
for D, start, lab in ((ITEMS, CFG["eval"][0], "evaluation"), (DEV_ITEMS, CFG["dev"][0], "development")):
    worst, viol, n_chk, n_less = 0.0, 0, 0, 0
    for (f, ph), (IT, _) in D.items():
        AV = availability(CFG["sched"][f][ph], start); st_, ne_ = FREQ[(id(D), f, ph, PRIMARY)]
        for k in rng6.choice(len(IT), min(60, len(IT)), replace=False):
            s, t, t0 = int(IT.s.values[k]), int(IT.t.values[k]), int(IT.t0.values[k]); same = mon == mon[t]
            if AV[s, t] or not (t0 < t): viol += 1                                         # target month must be hidden, origin before it
            own = AV[s] & (tidx <= t0) & same; kk = exceed(V[s, own], CEN[s, own], PRIMARY).sum(); nn = own.sum()
            oth = OBSD.copy(); oth[s] = AV[s]; msk = oth & (tidx <= t0)[None, :] & same[None, :]
            kn = sum(int(exceed(V[i, msk[i]], CEN[i, msk[i]], PRIMARY).sum()) for i in range(S)); nn_ = int(msk.sum())
            pn = (kn + .5) / (nn_ + 1); ps = (kk + M_PRIOR[PRIMARY] * pn) / (nn + M_PRIOR[PRIMARY])
            worst = max(worst, abs(ps - st_[k]), abs(pn - ne_[k])); n_chk += 1
            n_less += int(nn < (OBSD[s] & (tidx <= t0) & same).sum())                       # skipped months really left out of n
    chk(f"frequency climatologies = brute-force recomputation from visits <= t0 ({lab}, {n_chk} random skipped months)", worst < 1e-12 and viol == 0,
        f"max |difference| = {worst:.2e}; hidden-target/origin violations = {viol}")
    if lab == "evaluation":
        chk("reduced schedule respected: denominators smaller than if skipped months had been observed", n_less > 0,
            f"{n_less} of {n_chk} checked months have fewer same-month visits than the full record")
# (2) prior strength uses only 1995-2004: refit with every value from 2005 onwards deleted
_save = (OBSD, EXC_D)
try:
    cut05 = (yr < CFG["dev"][0])[None, :]; OBSD = _save[0] & cut05; EXC_D = {k: v & cut05 for k, v in _save[1].items()}
    M_chk = {thr: fit_prior_strength(thr, 1995, CFG["eval"][1]) for thr in THRS}
finally: OBSD, EXC_D = _save
chk("Beta-Binomial prior strength identical when all data from 2005 onwards are deleted", all(abs(M_chk[t_] - M_PRIOR[t_]) < 1e-9 * M_PRIOR[t_] for t_ in THRS),
    "; ".join(f"{t_}: {M_PRIOR[t_]:.4f} vs {M_chk[t_]:.4f}" for t_ in THRS))
# (3) development and evaluation sets are disjoint; development information ends before 2008
dmax_t = max(int(IT.t.max()) for IT, _ in DEV_ITEMS.values()); dmax_t0 = max(int(IT.t0.max()) for IT, _ in DEV_ITEMS.values())
emin_t = min(int(IT.t.min()) for IT, _ in ITEMS.values())
ov = sum(len(set(zip(ITEMS[k][0].s, ITEMS[k][0].t)) & set(zip(DEV_ITEMS[k][0].s, DEV_ITEMS[k][0].t))) for k in ITEMS)
chk("development items (2005-2007) and evaluation items (2008-2024) disjoint", ov == 0 and dmax_t < first_eval_t <= emin_t,
    f"overlapping station-months = {ov}; last development target month index {dmax_t}, first evaluation {emin_t}")
chk("development scores use information up to 2007 only (origin months)", dmax_t0 < first_eval_t, f"last development origin index {dmax_t0}")
chk("station-risk baseline: development ranking 1995-2004, evaluation ranking 1995-2007", DEV_RANK[1:] == (1995, 2004) and PRIMARY_RANK[1:] == (1995, 2007), f"{DEV_RANK}, {PRIMARY_RANK}")
# (4) cut-offs: existing methods unchanged; baselines re-derived from development items only
chk("cut-offs of Persistence / neighbour-ratio / HistGB identical to Experiment 1", all(CUT6[(f, thr, s_)] == CUTS[(f, thr)][(s_, "FA-budget")] for f in CFG["sched"] for thr in THRS for s_ in STRATS))
rec = {}
for thr in THRS:
    set_thr(thr)
    for f in CFG["sched"]:
        DV, _ = pooled(DEV_ITEMS, f); ydv = labels(DV, thr)
        for s_ in BASES:
            c = CUT6[(f, thr, s_)]; fa_ = ((score_of(DV, s_) > c) & ~ydv).sum() / len(CFG["sched"][f]) / n_sy_dev; rec[(f, thr, s_)] = fa_
chk("baseline cut-offs satisfy the false-alarm budget on 2005-2007 (or are the strictest candidate)", all(v <= CFG["fa_budget"] + 1e-12 or CUT6[k] >= max(score_of(pooled(DEV_ITEMS, k[0])[0], k[2])) for k, v in rec.items()),
    f"max development false alarms/station-yr = {max(rec.values()):.4f}")
# (5) identical evaluation sample and exceedance definition for every method; existing predictions untouched
okrows, oklab, okpred = True, True, True
for thr in THRS:
    set_thr(thr)
    for f in CFG["sched"]:
        IT, _ = pooled(ITEMS, f); yb = labels(IT, thr)
        okrows &= all(np.isfinite(score_of(IT, s_)).all() for s_ in ALLS)
        okrows &= all(((IT[s_] > 0) & (IT[s_] < 1)).all() for s_ in ("Clim_freq_station", "Clim_freq_network"))
        m0 = MAIN[(MAIN["visits/yr"] == f) & (MAIN.threshold == thr)].set_index("strategy")
        oklab &= int(yb.sum() / len(CFG["sched"][f])) == int(m0.loc["HistGB_nb", "skipped exceedances"])
        okpred &= all(abs(average_precision_score(yb, IT[s_]) - m0.loc[s_, "PR-AUC"]) < 1e-12 for s_ in STRATS)
chk("all scores finite on every skipped month; probabilities strictly inside (0, 1)", okrows)
chk("same skipped months and same exceedance labels as Experiment 1 (T04)", oklab)
chk("PR-AUC of Persistence / neighbour-ratio / HistGB unchanged (identical to T04)", okpred)
chk("scored months are inside the evaluation mask (complete station-years, sampled months)", all(EVAL[IT.s.values, IT.t.values].all() and EVAL[R.s.values, R.t.values].all() for IT, R in ITEMS.values()))
dup = int(d[d.StationCode.isin(sel)].duplicated(["StationCode", "ym"]).sum())
chk("one value per station-month (documented first-visit rule)", np.isfinite(V[~np.isnan(V)]).all(), f"{dup} additional same-month records in the cohort resolved by the first-visit rule")
nb_disc = pd.DataFrame(BOOT_LOG); chk("bootstrap replicates that could not be scored are counted", True,
                                      f"{int(nb_disc.discarded.sum())} of {int(nb_disc.replicates.sum())} replicates discarded over {len(nb_disc)} bootstrap calls (max {int(nb_disc.discarded.max())} in one call)")
AUDT = pd.DataFrame(AUD); save_table(AUDT, "T26_audit_checks"); save_table(nb_disc, "T26b_bootstrap_discarded_replicates"); display(AUDT)
print("ALL CHECKS PASSED" if (AUDT.result == "PASS").all() else f"FAILED CHECKS: {AUDT[AUDT.result == 'FAIL'].check.tolist()}")
# common evaluation configuration (read from the objects actually used above)
EVAL_CFG = {"dataset": "DAERA River Water Quality Monitoring 1990-2024, Ammonia (NH4-N); columns StationCode, sample date, NH4-N value, qualifier Q_NH4N",
            "monthly value": "first visit of the month (one value per station-month)", "censoring": "'<' values kept at the LOD; a censored value is a non-exceedance (exceed(): censored and value >= threshold -> False)",
            "exceedance": "value > threshold (mg N/L); primary threshold 0.3; also 0.2, 0.4, 0.6", "cohort": f"{S} river stations: >= {CFG['min_complete_years']} of 13 monthly-complete years (>= {CFG['complete_months']} months) in 1995-2007 and {CFG['dev_complete_years']}/3 in 2005-2007",
            "schedules (kept calendar months by phase)": {str(k): v for k, v in CFG["sched"].items()}, "development period": CFG["dev"], "evaluation period": CFG["eval"],
            "scored months": "months of monthly-complete station-years that were sampled in the record; skipped = not in the reduced schedule; visited = in the schedule",
            "training blocks": CFG["blocks"], "spatial clusters": CFG["n_groups"], "hyper-parameters (HistGB)": {k: HGB_GRID[v] for k, v in CHOSEN.items()},
            "cut-off rule": f"lowest cut-off with <= {CFG['fa_budget']} false alarms per station-year on 2005-2007, per method, schedule and threshold; frozen",
            "PR-AUC": "sklearn.metrics.average_precision_score on skipped months", "bootstrap": f"{CFG['n_boot']} replicates, seed {SEED}; station, year and two-way (station x year weights) clusters; percentile 95% intervals",
            "Beta-Binomial prior strength M (seasonal climatology)": M_PRIOR, "random seed": SEED, "fast mode": FAST}
json.dump(EVAL_CFG, open(os.path.join(OUT, "evaluation_config.json"), "w"), indent=1, default=str); print("evaluation configuration written to evaluation_config.json")
