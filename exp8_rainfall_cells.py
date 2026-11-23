# ==== CELL 1 ==== Experiment 8a - rainfall covariates (HadUK-Grid; pre-declared in EXPERIMENT8_PLAN.md): data and features
# r = ln(1 + monthly rainfall). Features for target month t and origin t0: rain_t, raindays10_t (days >= 10 mm in t),
# rain_anom_t = r(t) - mean r of the same calendar month over all months <= t0, rain_change = r(t) - r(t0).
# Assumption: rainfall of month t is available at the end of month t (end-of-month inference).
RAIN_FILE = os.path.join(ROOT, "data", "haduk_rainfall_stations_monthly.csv")
HAVE_RAIN = os.path.exists(RAIN_FILE)
RAINC = ["rain_t", "raindays10_t", "rain_anom_t", "rain_change"]
if not HAVE_RAIN:
    print(f"{RAIN_FILE} not found - Experiment 8 is skipped (place the file in data/ to run it)")
else:
    rf = pd.read_csv(RAIN_FILE); rf["p"] = pd.PeriodIndex(rf.month, freq="M")
    sidx8 = {c: i for i, c in enumerate(sel)}; midx8 = {m: i for i, m in enumerate(months)}
    si, ti = rf.StationCode.map(sidx8), rf.p.map(midx8); ok_ = si.notna() & ti.notna()
    assert ok_.sum() == S * T, f"rainfall file covers {ok_.sum()} of {S * T} station-months"
    RAINL = np.full((S, T), np.nan); RD10 = np.full((S, T), np.nan)
    RAINL[si[ok_].astype(int).values, ti[ok_].astype(int).values] = np.log1p(rf.rainfall_mm[ok_].values)
    RD10[si[ok_].astype(int).values, ti[ok_].astype(int).values] = rf.raindays_ge10mm[ok_].values
    assert np.isfinite(RAINL).all() and np.isfinite(RD10).all(), "rainfall missing for some station-months"
    print(f"rainfall loaded: {rf.StationCode.nunique()} stations x {rf.month.nunique()} months; max station-cell distance {rf.cell_distance_m.max():.0f} m")
    def rain_cols_from(RL, RD, F):
        s, t, t0 = F.s.values.astype(int), F.t.values.astype(int), F.t0.values.astype(int); m = mon[t] - 1
        CS = np.stack([np.cumsum(np.where(mon[None, :] == k, np.nan_to_num(RL), 0.0), 1) for k in range(1, 13)])
        CN = np.stack([np.cumsum((mon[None, :] == k) & np.isfinite(RL), 1) for k in range(1, 13)]).astype(float)
        n = CN[m, s, t0]; clim = np.where(n > 0, CS[m, s, t0] / np.maximum(n, 1), np.nan)
        return pd.DataFrame({"rain_t": RL[s, t], "raindays10_t": RD[s, t], "rain_anom_t": RL[s, t] - clim, "rain_change": RL[s, t] - RL[s, t0]}, index=F.index)
    for F_ in [PAIRS] + [IT for IT, _ in ITEMS.values()] + [IT for IT, _ in DEV_ITEMS.values()]:
        F_[RAINC] = rain_cols_from(RAINL, RD10, F_)
    # leakage check: delete all rainfall after month t (features) and after t0 (climatology part) for 300 random skipped months
    IT0 = ITEMS[(4, 0)][0]; pk = np.random.RandomState(SEED).choice(len(IT0), 300, replace=False); md_, mc_ = 0.0, 0.0
    for k in pk:
        r = IT0.iloc[[k]]; t, t0 = int(r.t.iloc[0]), int(r.t0.iloc[0])
        RLa = RAINL.copy(); RLa[:, t + 1:] = np.nan; RDa = RD10.copy(); RDa[:, t + 1:] = np.nan
        md_ = max(md_, float(np.nanmax(np.abs(rain_cols_from(RLa, RDa, r).values - r[RAINC].values))))
        RLb = RAINL.copy(); RLb[:, t0 + 1:] = np.nan; RLb[:, t] = RAINL[:, t]                     # keep r(t), delete everything else after t0
        cb = rain_cols_from(RLb, RD10, r)["rain_anom_t"].values; mc_ = max(mc_, float(np.nanmax(np.abs(cb - r["rain_anom_t"].values))))
    print(f"rainfall leakage check (300 skipped months): max |difference| after deleting rainfall after t = {md_:.2e}; "
          f"climatology after deleting rainfall after t0 = {mc_:.2e} ->", "PASS" if max(md_, mc_) < 1e-12 else "FAIL")
    assert max(md_, mc_) < 1e-12
    display(PAIRS[RAINC].describe().round(3))

# ==== CELL 2 ==== Experiment 8b - models with rainfall (same hyper-parameters, seen design, block retraining, dev cut-offs) and evaluation
if HAVE_RAIN:
    M8 = {"HistGB_rain_own": OWN + RAINC, "HistGB_rain_nb": OWN + NBC + RAINC}
    CH8 = {"HistGB_rain_own": CHOSEN["HistGB_own"], "HistGB_rain_nb": CHOSEN["HistGB_nb"]}       # no re-tuning
    _chosen_saved = dict(CHOSEN); CHOSEN.update(CH8)
    try:
        t_ = time.time(); P8 = run_models(ITEMS, "seen", pairs=PAIRS, cols_map=M8)
    finally:
        CHOSEN.clear(); CHOSEN.update(_chosen_saved)
    for k, (IT, _) in ITEMS.items():
        for n in M8: IT[n] = P8[k][n]
    DEV8 = {n: fit(cols, HGB_GRID[CH8[n]], PAIRS[PAIRS.year < CFG["dev"][0]]) for n, cols in M8.items()}
    for k, (IT, _) in DEV_ITEMS.items():
        for n, cols in M8.items(): IT[n] = predict(DEV8[n], cols, IT)
    assert all(ITEMS[k][0][list(M8)].notna().all().all() for k in ITEMS)
    print(f"rainfall models fitted in {time.time() - t_:.0f}s")
    for f in CFG["sched"]:
        for thr in THRS:
            DV, _ = pooled(DEV_ITEMS, f); yb = labels(DV, thr); nph = len(CFG["sched"][f])
            for n in M8:
                sc = np.exp(DV[n].values); fa = np.array([((sc > c) & ~yb).sum() / nph / n_sy_dev for c in GRID]); ok = np.flatnonzero(fa <= CFG["fa_budget"])
                CUT6[(f, thr, n)] = float(GRID[ok[0]]) if len(ok) else float(GRID[-1])
    M8ALL = ["Observation-only", "Persistence", "HistGB_own", "HistGB_rain_own", "HistGB_nb", "HistGB_rain_nb", "Clim_freq_station"]
    rows = []
    for thr in THRS:
        set_thr(thr)
        for f in CFG["sched"]:
            IT, RET = pooled(ITEMS, f); nph = len(CFG["sched"][f]); yb = labels(IT, thr)
            for s_ in M8ALL:
                a = alarms(IT, f, thr, s_); tp, ev = tp_ev(IT, RET, thr, a); fp = 0 if a is None else int((a & ~yb).sum()); TPs = 0 if a is None else int((a & yb).sum())
                rows.append({"visits/yr": f, "threshold": thr, "strategy": s_, "PR-AUC (skipped months)": np.nan if a is None else average_precision_score(yb, score_of(IT, s_)),
                             "network-level recall": tp.sum() / ev.sum(), "skipped-month recall": TPs / yb.sum(), "predictive precision": TPs / a.sum() if (a is not None and a.sum()) else np.nan,
                             "false alarms/station-yr": fp / nph / n_sy_eval, "MAE log": float(np.abs(IT[s_].values - IT.y.values).mean()) if s_ in ("Persistence", "HistGB_own", "HistGB_rain_own", "HistGB_nb", "HistGB_rain_nb") else np.nan})
    RAIN8 = pd.DataFrame(rows); save_table(RAIN8, "T29_rainfall_models")
    display(RAIN8[(RAIN8.threshold == PRIMARY)].set_index(["visits/yr", "strategy"]).drop(columns="threshold").round(3))
    bt = []
    for f in CFG["sched"]:
        for thr in (0.2, PRIMARY):
            set_thr(thr); IT, RET = pooled(ITEMS, f)
            for a_, b_ in (("HistGB_rain_nb", "HistGB_nb"), ("HistGB_rain_own", "HistGB_own"), ("HistGB_rain_nb", "Clim_freq_station")):
                for cl in ("station", "year"):
                    e, lo, hi = boot_ap_counted(IT, a_, b_, thr, cl)
                    bt.append({"visits/yr": f, "threshold": thr, "metric": "PR-AUC", "comparison": f"{a_} - {b_}", "cluster": cl, "difference": e, "95% CI low": lo, "95% CI high": hi})
                    for so, mlab in ((False, "network-level recall"), (True, "skipped-month recall")):
                        ta, ea = tp_ev(IT, RET, thr, alarms(IT, f, thr, a_), cl, so); tb, eb_ = tp_ev(IT, RET, thr, alarms(IT, f, thr, b_), cl, so)
                        e, lo, hi = boot_ratio_diff(ta, ea, tb, eb_, tag=f"8 {mlab} {a_}-{b_} {f} {thr} {cl}")
                        bt.append({"visits/yr": f, "threshold": thr, "metric": mlab, "comparison": f"{a_} - {b_}", "cluster": cl, "difference": e, "95% CI low": lo, "95% CI high": hi})
                if thr == PRIMARY:
                    e, lo, hi, sk = boot_ap_2way(IT, a_, b_, thr)
                    bt.append({"visits/yr": f, "threshold": thr, "metric": "PR-AUC", "comparison": f"{a_} - {b_}", "cluster": "two-way", "difference": e, "95% CI low": lo, "95% CI high": hi, "replicates discarded": sk})
    RAIN8B = pd.DataFrame(bt); save_table(RAIN8B, "T29b_rainfall_models_bootstrap")
    display(RAIN8B[(RAIN8B.threshold == PRIMARY) & (RAIN8B["visits/yr"] == 4)].round(4))
    # by station group (4 visits/yr, primary threshold)
    set_thr(PRIMARY); IT, RET = pooled(ITEMS, 4); yb = labels(IT, PRIMARY); gr = []
    TE = {s_: tp_ev(IT, RET, PRIMARY, alarms(IT, 4, PRIMARY, s_)) for s_ in ("HistGB_nb", "HistGB_rain_nb")}
    for g, m in GRP.items():
        e, lo, hi = boot_ratio_diff(*TE["HistGB_rain_nb"], *TE["HistGB_nb"], units=np.flatnonzero(m), tag=f"8 group {g}")
        sk = m[IT.s.values]
        gr.append({"group": g, "recall HistGB_nb": TE["HistGB_nb"][0][m].sum() / TE["HistGB_nb"][1][m].sum(), "recall HistGB_rain_nb": TE["HistGB_rain_nb"][0][m].sum() / TE["HistGB_rain_nb"][1][m].sum(),
                   "d recall": e, "95% CI low": lo, "95% CI high": hi,
                   "PR-AUC HistGB_nb": average_precision_score(yb[sk], IT.HistGB_nb.values[sk]) if yb[sk].sum() >= 20 else np.nan,
                   "PR-AUC HistGB_rain_nb": average_precision_score(yb[sk], IT.HistGB_rain_nb.values[sk]) if yb[sk].sum() >= 20 else np.nan})
    RAIN8G = pd.DataFrame(gr); save_table(RAIN8G, "T29c_rainfall_models_station_groups"); display(RAIN8G.round(3))
    x = RAIN8[(RAIN8.threshold == PRIMARY) & RAIN8.strategy.isin(["Persistence", "Clim_freq_station", "HistGB_own", "HistGB_rain_own", "HistGB_nb", "HistGB_rain_nb"])]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4)); cols8 = {"Persistence": "#52514e", "Clim_freq_station": "#2e8b57", "HistGB_own": "#2a78d6", "HistGB_rain_own": "#9fc3ef", "HistGB_nb": "#e34948", "HistGB_rain_nb": "#f4a3a3"}
    for a_, met in zip(ax, ("PR-AUC (skipped months)", "network-level recall")):
        fr = list(CFG["sched"]); xx = np.arange(len(fr)); w = .8 / len(cols8)
        for i, (s_, cc) in enumerate(cols8.items()):
            b = a_.bar(xx + (i - (len(cols8) - 1) / 2) * w, [x[(x.strategy == s_) & (x["visits/yr"] == f)][met].iloc[0] for f in fr], w * .92, color=cc, label=s_)
            a_.bar_label(b, fmt="%.3f", fontsize=5.5, rotation=90, padding=1)
        a_.set_xticks(xx); a_.set_xticklabels([f"{f} visits/yr" for f in fr]); a_.set_ylabel(met); a_.set_ylim(0, x[met].max() * 1.3)
    ax[0].legend(fontsize=7, ncol=2); fig.suptitle(f"Models with and without rainfall covariates (one-at-a-time design, > {PRIMARY} mg/l N)", fontsize=9)
    fig.tight_layout(); savefig(fig, "F24_rainfall_models"); plt.show()

# ==== CELL 3 ==== Experiment 8c - addendum (EXPERIMENT8_ADDENDUM.md; specified after a fast development run had produced preliminary numbers): timing sensitivity, raindays ablation, grid-cell bootstrap,
# wet/normal/dry strata and threshold-selection table. Written after the fast-mode development run; reported as sensitivity/exploratory.
if HAVE_RAIN:
    def rain_lag_cols(F):
        s, t, t0 = F.s.values.astype(int), F.t.values.astype(int), F.t0.values.astype(int); tl = t - 1; m = mon[tl] - 1
        CS = np.stack([np.cumsum(np.where(mon[None, :] == k, RAINL, 0.0), 1) for k in range(1, 13)])
        CN = np.stack([np.cumsum(np.broadcast_to(mon[None, :] == k, RAINL.shape), 1) for k in range(1, 13)]).astype(float)
        n = CN[m, s, t0]; clim = np.where(n > 0, CS[m, s, t0] / np.maximum(n, 1), np.nan)
        return pd.DataFrame({"rainlag_t": RAINL[s, tl], "raindays10lag_t": RD10[s, tl], "rainlag_anom_t": RAINL[s, tl] - clim, "rainlag_change": RAINL[s, tl] - RAINL[s, t0]}, index=F.index)
    RAINLAGC = ["rainlag_t", "raindays10lag_t", "rainlag_anom_t", "rainlag_change"]
    for F_ in [PAIRS] + [IT for IT, _ in ITEMS.values()] + [IT for IT, _ in DEV_ITEMS.values()]:
        F_[RAINLAGC] = rain_lag_cols(F_)
    assert all((IT.t.values - 1 <= IT.t.values).all() for IT, _ in ITEMS.values())
    M8c = {"HistGB_rainlag_nb": OWN + NBC + RAINLAGC, "HistGB_rain_noRD_nb": OWN + NBC + ["rain_t", "rain_anom_t", "rain_change"]}
    CH8c = {n: CHOSEN["HistGB_nb"] for n in M8c}; _chosen_saved = dict(CHOSEN); CHOSEN.update(CH8c)
    try:
        t_ = time.time(); P8c = run_models(ITEMS, "seen", pairs=PAIRS, cols_map=M8c)
    finally:
        CHOSEN.clear(); CHOSEN.update(_chosen_saved)
    for k, (IT, _) in ITEMS.items():
        for n in M8c: IT[n] = P8c[k][n]
    DEV8c = {n: fit(cols, HGB_GRID[CH8c[n]], PAIRS[PAIRS.year < CFG["dev"][0]]) for n, cols in M8c.items()}
    for k, (IT, _) in DEV_ITEMS.items():
        for n, cols in M8c.items(): IT[n] = predict(DEV8c[n], cols, IT)
    for f in CFG["sched"]:
        for thr in THRS:
            DV, _ = pooled(DEV_ITEMS, f); yb = labels(DV, thr); nph = len(CFG["sched"][f])
            for n in M8c:
                sc = np.exp(DV[n].values); fa = np.array([((sc > c) & ~yb).sum() / nph / n_sy_dev for c in GRID]); ok = np.flatnonzero(fa <= CFG["fa_budget"])
                CUT6[(f, thr, n)] = float(GRID[ok[0]]) if len(ok) else float(GRID[-1])
    print(f"addendum models (timing sensitivity, raindays ablation) fitted in {time.time() - t_:.0f}s")
    # (a, b) ablation table: M1 own, M2 own+nb, M3 own+rain, M4 own+nb+rain, M4 with t-1 rainfall, M4 without raindays
    ABL = [("M1 own history", "HistGB_own"), ("M2 + neighbours", "HistGB_nb"), ("M3 own + rainfall", "HistGB_rain_own"), ("M4 + neighbours + rainfall", "HistGB_rain_nb"),
           ("M4 with rainfall of month t-1 (timing sensitivity)", "HistGB_rainlag_nb"), ("M4 without raindays10mm (ablation)", "HistGB_rain_noRD_nb")]
    rows, bt = [], []
    for thr in (0.2, PRIMARY):
        set_thr(thr)
        for f in CFG["sched"]:
            IT, RET = pooled(ITEMS, f); nph = len(CFG["sched"][f]); yb = labels(IT, thr)
            for lab, s_ in ABL:
                a = alarms(IT, f, thr, s_); tp, ev = tp_ev(IT, RET, thr, a); TPs = int((a & yb).sum()); FPs = int((a & ~yb).sum()); FNs = int(yb.sum()) - TPs
                prec = tp.sum() / (tp.sum() + FPs); rec = tp.sum() / ev.sum()
                rows.append({"visits/yr": f, "threshold": thr, "model": lab, "strategy": s_, "network-level recall": rec, "skipped-month recall": TPs / yb.sum(),
                             "predictive precision": TPs / a.sum() if a.sum() else np.nan, "operational precision": prec, "F1 (network-level)": 2 * prec * rec / (prec + rec),
                             "PR-AUC": average_precision_score(yb, IT[s_]), "ROC-AUC": roc_auc_score(yb, IT[s_]), "TP (skipped, per phase)": TPs / nph, "FP (per phase)": FPs / nph,
                             "FN (skipped, per phase)": FNs / nph, "false alarms/station-yr": FPs / nph / n_sy_eval, "missed exceedances/station-yr": (ev.sum() - tp.sum()) / nph / n_sy_eval})
            for s_ in ("HistGB_rainlag_nb", "HistGB_rain_noRD_nb"):
                for cl in ("station", "year"):
                    e, lo, hi = boot_ap_counted(IT, s_, "HistGB_nb", thr, cl)
                    bt.append({"visits/yr": f, "threshold": thr, "metric": "PR-AUC", "comparison": f"{s_} - HistGB_nb", "cluster": cl, "difference": e, "95% CI low": lo, "95% CI high": hi})
                    ta, ea = tp_ev(IT, RET, thr, alarms(IT, f, thr, s_), cl); tb, eb_ = tp_ev(IT, RET, thr, alarms(IT, f, thr, "HistGB_nb"), cl)
                    e, lo, hi = boot_ratio_diff(ta, ea, tb, eb_, tag=f"8c {s_} {f} {thr} {cl}")
                    bt.append({"visits/yr": f, "threshold": thr, "metric": "network-level recall", "comparison": f"{s_} - HistGB_nb", "cluster": cl, "difference": e, "95% CI low": lo, "95% CI high": hi})
    ABLT = pd.DataFrame(rows); save_table(ABLT, "T29d_ablation_rainfall"); save_table(pd.DataFrame(bt), "T29e_ablation_rainfall_bootstrap")
    display(ABLT[(ABLT.threshold == PRIMARY) & (ABLT["visits/yr"] == 4)].set_index("model").drop(columns=["visits/yr", "threshold", "strategy"]).round(3))
    # (c) grid-cell cluster bootstrap of the primary comparison
    cellkey = rf.drop_duplicates("StationCode").set_index("StationCode")[["cell_x", "cell_y"]].apply(tuple, axis=1)
    cells = sorted(set(cellkey.values)); cid = np.array([cells.index(cellkey[c]) for c in sel]); K = len(cells)
    set_thr(PRIMARY); IT, RET = pooled(ITEMS, 4); yb = labels(IT, PRIMARY); kc = cid[IT.s.values]
    mem = [np.flatnonzero(kc == k) for k in range(K)]; rr = np.random.RandomState(SEED); out = []; skc = 0
    for _ in range(CFG["n_boot"]):
        idx = np.concatenate([mem[k] for k in rr.randint(0, K, K)])
        if yb[idx].sum() == 0: skc += 1; continue
        out.append(average_precision_score(yb[idx], IT.HistGB_rain_nb.values[idx]) - average_precision_score(yb[idx], IT.HistGB_nb.values[idx]))
    ta, ea = tp_ev(IT, RET, PRIMARY, alarms(IT, 4, PRIMARY, "HistGB_rain_nb")); tb, eb_ = tp_ev(IT, RET, PRIMARY, alarms(IT, 4, PRIMARY, "HistGB_nb"))
    agg = lambda x: np.bincount(cid, x, K)
    e2, lo2, hi2 = boot_ratio_diff(agg(ta), agg(ea), agg(tb), agg(eb_), tag="8c grid-cell recall")
    CELLB = pd.DataFrame([{"comparison": "HistGB_rain_nb - HistGB_nb", "metric": "PR-AUC", "cluster": f"grid cell ({K} cells)",
                           "difference": average_precision_score(yb, IT.HistGB_rain_nb) - average_precision_score(yb, IT.HistGB_nb), "95% CI low": np.percentile(out, 2.5), "95% CI high": np.percentile(out, 97.5), "replicates discarded": skc},
                          {"comparison": "HistGB_rain_nb - HistGB_nb", "metric": "network-level recall", "cluster": f"grid cell ({K} cells)", "difference": e2, "95% CI low": lo2, "95% CI high": hi2, "replicates discarded": np.nan}])
    save_table(CELLB, "T29f_rainfall_gridcell_bootstrap"); display(CELLB.round(4))
    # (d) wet / normal / dry months: terciles of the station's 1995-2007 rainfall for the same calendar month (no evaluation data)
    q = np.full((S, 12, 2), np.nan); hist = (yr >= 1995) & (yr <= 2007)
    for m_ in range(1, 13):
        cols_ = hist & (mon == m_); q[:, m_ - 1, 0] = np.percentile(RAINL[:, cols_], 100 / 3, axis=1); q[:, m_ - 1, 1] = np.percentile(RAINL[:, cols_], 200 / 3, axis=1)
    def wet_class(F):
        s, t = F.s.values.astype(int), F.t.values.astype(int); r = RAINL[s, t]; lo_, hi_ = q[s, mon[t] - 1, 0], q[s, mon[t] - 1, 1]
        return np.where(r <= lo_, "dry (lower tercile)", np.where(r > hi_, "wet (upper tercile)", "normal (middle tercile)"))
    rows = []
    for f in CFG["sched"]:
        IT, RET = pooled(ITEMS, f); yb = labels(IT, PRIMARY); wc = wet_class(IT); wr = wet_class(RET.assign(t0=RET.t))
        for cls in ("dry (lower tercile)", "normal (middle tercile)", "wet (upper tercile)"):
            mi, mr = wc == cls, wr == cls; ITc, RETc = IT[mi], RET[mr]; ybc = yb[mi]
            TE = {s_: tp_ev(ITc, RETc, PRIMARY, alarms(ITc, f, PRIMARY, s_)) for s_ in ("Persistence", "Clim_freq_station", "HistGB_own", "HistGB_nb", "HistGB_rain_nb")}
            e, lo, hi = boot_ratio_diff(*TE["HistGB_nb"], *TE["HistGB_own"], tag=f"8c wet/dry {cls} {f}")
            r_ = {"visits/yr": f, "rainfall class": cls, "skipped months": int(mi.sum()) // len(CFG["sched"][f]), "skipped exceedances": int(ybc.sum()) // len(CFG["sched"][f]),
                  "exceedance rate (skipped months)": ybc.mean(), "neighbour benefit, recall (HistGB_nb - HistGB_own)": e, "95% CI low": lo, "95% CI high": hi}
            for s_ in TE:
                r_[f"PR-AUC {s_}"] = average_precision_score(ybc, score_of(ITc, s_)) if ybc.sum() >= 20 else np.nan; r_[f"recall {s_}"] = TE[s_][0].sum() / TE[s_][1].sum()
            rows.append(r_)
    WET = pd.DataFrame(rows); save_table(WET, "T29g_wet_normal_dry_months"); display(WET[WET["visits/yr"] == 4].set_index("rainfall class").drop(columns="visits/yr").T.round(3))
    # (e) threshold-selection table: development curve, selected cut-off and frozen evaluation result
    TS, curve = [], []
    for f in CFG["sched"]:
        DV, DRET = pooled(DEV_ITEMS, f); IT, RET = pooled(ITEMS, f); ydv, yev = labels(DV, PRIMARY), labels(IT, PRIMARY); nph = len(CFG["sched"][f])
        for s_ in ("Persistence", "Clim_freq_station", "HistGB_own", "HistGB_nb", "HistGB_rain_nb"):
            cands = cands7(s_, score_of(DV, s_)); sdv, sev = score_of(DV, s_), score_of(IT, s_)
            for c in cands:
                curve.append({"visits/yr": f, "strategy": s_, "candidate cut-off": float(c), "development skipped-month recall": ((sdv > c) & ydv).sum() / ydv.sum(),
                              "development false alarms/station-yr": ((sdv > c) & ~ydv).sum() / nph / n_sy_dev})
            c = CUT6[(f, PRIMARY, s_)]; a_dv, a_ev = sdv > c, sev > c
            tpd, evd = tp_ev(DV, DRET, PRIMARY, a_dv); tpe, eve = tp_ev(IT, RET, PRIMARY, a_ev)
            TS.append({"visits/yr": f, "strategy": s_, "selected cut-off (frozen)": c, "development skipped-month recall": (a_dv & ydv).sum() / ydv.sum(),
                       "development network-level recall": tpd.sum() / evd.sum(), "development false alarms/station-yr": (a_dv & ~ydv).sum() / nph / n_sy_dev,
                       "evaluation skipped-month recall": (a_ev & yev).sum() / yev.sum(), "evaluation network-level recall": tpe.sum() / eve.sum(),
                       "evaluation false alarms/station-yr": (a_ev & ~yev).sum() / nph / n_sy_eval, "constraint met in evaluation (<= 0.1)": (a_ev & ~yev).sum() / nph / n_sy_eval <= CFG["fa_budget"]})
    TST = pd.DataFrame(TS); save_table(TST, "T29h_threshold_selection"); save_table(pd.DataFrame(curve), "T29i_threshold_selection_curves")
    display(TST[TST["visits/yr"] == 4].set_index("strategy").drop(columns="visits/yr").round(3))
