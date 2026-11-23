# ==== CELL 1 ==== Experiment 7a - rolling recalibration of alarm cut-offs (one-at-a-time design; pre-declared in EXPERIMENT7_PLAN.md)
# For block B and spatial cluster g the FA-budget cut-off is re-selected on the skipped-month predictions of the OTHER clusters in B-3..B-1
# (under the one-at-a-time design those stations keep monthly sampling, so their outcomes are known); B = 2008 keeps the frozen 2005-2007 cut-off.
# Model predictions are unchanged; only the cut-off changes.
M7 = ["Persistence", "Neighbour_ratio", "HistGB_own", "HistGB_nb", "Clim_freq_station"]
def cands7(s_, sc): return GRID if s_ not in PROB else np.unique(np.quantile(sc, np.linspace(0, 1, 401)))
def roll_cuts(f, thr, s_):
    set_thr(thr); IT, _ = pooled(ITEMS, f); nph = len(CFG["sched"][f]); yb = labels(IT, thr); sc = score_of(IT, s_); cuts = {}
    for B in CFG["blocks"]:
        for g in range(CFG["n_groups"]):
            if B == CFG["eval"][0]: cuts[(B, g)] = CUT6[(f, thr, s_)]; continue
            m = IT.year.between(B - 3, B - 1).values & (IT.group.values != g)
            assert IT.year.values[m].max() < B                                              # calibration strictly before the block
            nsy = int(okSY[groups != g][:, B - 3 - 1990:B - 1990].sum())
            c_ = cands7(s_, sc[m]); fa = np.array([((sc[m] > c) & ~yb[m]).sum() / nph / nsy for c in c_]); ok = np.flatnonzero(fa <= CFG["fa_budget"])
            cuts[(B, g)] = float(c_[ok[0]]) if len(ok) else float(c_[-1])
    return cuts
def roll_alarm(IT, s_, cuts):
    key = list(zip(IT.year.map(block_of).values, IT.group.values)); return score_of(IT, s_) > np.array([cuts[k] for k in key])
NSY_ST = okSY[:, CFG["eval"][0] - 1990:CFG["eval"][1] - 1989].sum(1).astype(float)          # evaluation station-years per station
NSY_YR = okSY[:, CFG["eval"][0] - 1990:CFG["eval"][1] - 1989].sum(0).astype(float)          # ... per year
def fp_units(IT, a, yb, cluster):
    ke = IT.s.values if cluster == "station" else IT.year.values - CFG["eval"][0]; K = S if cluster == "station" else NYR
    return np.bincount(ke, (a & ~yb).astype(float), K)
def boot_fa_diff(fa_a, fa_b, nsy, nph, B=None, seed=SEED):
    B = CFG["n_boot"] if B is None else B; K = len(nsy); dd = np.random.RandomState(seed).randint(0, K, (B, K))
    d_ = (fa_a[dd].sum(1) - fa_b[dd].sum(1)) / nph / nsy[dd].sum(1)
    return (fa_a.sum() - fa_b.sum()) / nph / nsy.sum(), np.percentile(d_, 2.5), np.percentile(d_, 97.5)
rows, bb, CUT7 = [], [], {}
for thr in (0.2, PRIMARY):
    for f in CFG["sched"]:
        set_thr(thr); IT, RET = pooled(ITEMS, f); nph = len(CFG["sched"][f]); yb = labels(IT, thr)
        for s_ in M7:
            cuts = roll_cuts(f, thr, s_); CUT7[(f, thr, s_)] = cuts; set_thr(thr)
            for lab, a in (("frozen 2005-2007", alarms(IT, f, thr, s_)), ("rolling 3-year recalibration", roll_alarm(IT, s_, cuts))):
                tp, ev = tp_ev(IT, RET, thr, a); fp = int((a & ~yb).sum()); TPs = int((a & yb).sum())
                rows.append({"visits/yr": f, "threshold": thr, "strategy": s_, "cut-off rule": lab, "network-level recall": tp.sum() / ev.sum(),
                             "skipped-month recall": TPs / yb.sum(), "predictive precision": TPs / a.sum() if a.sum() else np.nan,
                             "false alarms/station-yr": fp / nph / n_sy_eval, "median cut-off": float(np.median(list(cuts.values()))) if lab.startswith("rolling") else CUT6[(f, thr, s_)]})
            aF, aR = alarms(IT, f, thr, s_), roll_alarm(IT, s_, cuts)
            for cl in ("station", "year"):
                (ta, ea), (tb, eb_) = tp_ev(IT, RET, thr, aR, cl), tp_ev(IT, RET, thr, aF, cl)
                e, lo, hi = boot_ratio_diff(ta, ea, tb, eb_, tag=f"7a recall {s_} {f} {thr} {cl}")
                bb.append({"visits/yr": f, "threshold": thr, "strategy": s_, "metric": "network-level recall, rolling - frozen", "cluster": cl, "difference": e, "95% CI low": lo, "95% CI high": hi})
                e, lo, hi = boot_fa_diff(fp_units(IT, aR, yb, cl), fp_units(IT, aF, yb, cl), NSY_ST if cl == "station" else NSY_YR, nph)
                bb.append({"visits/yr": f, "threshold": thr, "strategy": s_, "metric": "false alarms/station-yr, rolling - frozen", "cluster": cl, "difference": e, "95% CI low": lo, "95% CI high": hi})
RECAL = pd.DataFrame(rows); save_table(RECAL, "T27_rolling_recalibration"); RECALB = pd.DataFrame(bb); save_table(RECALB, "T27b_rolling_recalibration_bootstrap")
display(RECAL[(RECAL.threshold == PRIMARY) & (RECAL["visits/yr"] == 4)].set_index(["strategy", "cut-off rule"]).drop(columns=["visits/yr", "threshold"]).round(3))
display(RECALB[(RECALB.threshold == PRIMARY) & (RECALB["visits/yr"] == 4) & (RECALB.strategy == "HistGB_nb")].round(4))
# realised false alarms per block (HistGB + nb, 4 visits/yr, primary threshold), frozen vs rolling
set_thr(PRIMARY); IT, RET = pooled(ITEMS, 4); yb = labels(IT, PRIMARY); blk = IT.year.map(block_of).values; rb = []
for lab, a in (("frozen 2005-2007", alarms(IT, 4, PRIMARY, "HistGB_nb")), ("rolling 3-year recalibration", roll_alarm(IT, "HistGB_nb", CUT7[(4, PRIMARY, "HistGB_nb")]))):
    for i_, B in enumerate(CFG["blocks"]):
        end_ = CFG["blocks"][i_ + 1] - 1 if i_ + 1 < len(CFG["blocks"]) else CFG["eval"][1]          # block runs to the next block start
        ys = list(range(B, end_ + 1)); nsy = okSY[:, [y - 1990 for y in ys]].sum()
        m = blk == B; rb.append({"cut-off rule": lab, "block": f"{B}-{ys[-1]}", "false alarms/station-yr": (a[m] & ~yb[m]).sum() / len(CFG["sched"][4]) / nsy,
                                 "skipped-month recall": (a[m] & yb[m]).sum() / max(yb[m].sum(), 1)})
RB = pd.DataFrame(rb); save_table(RB, "T27c_rolling_recalibration_by_block"); display(RB.round(3))
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for lab, cc in (("frozen 2005-2007", "#52514e"), ("rolling 3-year recalibration", "#e34948")):
    x = RB[RB["cut-off rule"] == lab]; ax[0].plot(x.block, x["false alarms/station-yr"], "o-", color=cc, label=lab); ax[1].plot(x.block, x["skipped-month recall"], "o-", color=cc, label=lab)
ax[0].axhline(CFG["fa_budget"], color="k", ls=":", lw=.8); ax[0].set_ylim(0, None); ax[1].set_ylim(0, None)
ax[0].set_title(f"(a) Realised false alarms per station-year (target {CFG['fa_budget']})"); ax[1].set_title("(b) Skipped-month recall")
for a_ in ax: a_.set_xlabel("Evaluation block"); a_.tick_params(axis="x", labelsize=7)
ax[0].legend(fontsize=7); fig.suptitle(f"HistGB + neighbours, one-at-a-time design, 4 visits/yr, > {PRIMARY} mg/l N", fontsize=9)
fig.tight_layout(); savefig(fig, "F22_rolling_recalibration"); plt.show()

# ==== CELL 2 ==== Experiment 7b - alarm-triggered follow-up sampling (pre-declared in EXPERIMENT7_PLAN.md)
# A trigger for station s in month t (known at the end of month t) adds one visit in month t+1 unless t+1 is already a scheduled visit.
# The extra visit detects a recorded exceedance in t+1 not already detected; it does not feed back into later features (conservative).
# Extra visits to months without a usable record value count as cost without detection. Random control: the same number of scorable
# extra visits placed at random skipped months of the same rotation (200 draws).
N_RAND = 20 if FAST else 200
def followup(name, thr):
    EVs, RETs, rev = FAIR_EV[(name, thr)]; yb = exceed(np.exp(EVs.y.values), EVs.cens_target.values, thr); al = EVs["alarm_HistGB_nb"].values
    rr = np.random.RandomState(SEED); out = {}; ev_st = np.zeros(S); base_st = np.zeros(S); nv = EVs.v.nunique()
    gain = {k: np.zeros(S) for k in ("HistGB + nb alarm", "observed exceedance", "either trigger")}; cost = {k: 0 for k in gain}; scor = {k: 0 for k in gain}
    rnd = {k: [] for k in gain}; rnd_st = {k: np.zeros(S) for k in gain}
    for v in np.unique(EVs.v.values):
        me, mr = EVs.v.values == v, RETs.v.values == v
        ks = EVs.s.values[me] * T + EVs.t.values[me]; kr = RETs.s.values[mr] * T + RETs.t.values[mr]
        y_sk, a_sk, ex_r = yb[me], al[me], rev[mr].astype(bool)
        idx_sk = {k: i for i, k in enumerate(ks)}; vis = set(kr.tolist())
        detected = set(ks[a_sk & y_sk].tolist()) | set(kr[ex_r].tolist())
        ev_st += np.bincount(EVs.s.values[me], y_sk.astype(float), S) + np.bincount(RETs.s.values[mr], ex_r.astype(float), S)
        base_st += np.bincount(EVs.s.values[me], (a_sk & y_sk).astype(float), S) + np.bincount(RETs.s.values[mr], ex_r.astype(float), S)
        trig = {"HistGB + nb alarm": set(ks[a_sk].tolist()), "observed exceedance": set(kr[ex_r].tolist())}; trig["either trigger"] = trig["HistGB + nb alarm"] | trig["observed exceedance"]
        for k, tk in trig.items():
            nxt = {x + 1 for x in tk if (x % T) + 1 < T} - vis                               # next month, not already scheduled
            cost[k] += len(nxt); sc_ = [x for x in nxt if x in idx_sk]; scor[k] += len(sc_)
            new = [x for x in sc_ if y_sk[idx_sk[x]] and x not in detected]
            gain[k] += np.bincount(np.array([x // T for x in new], int), minlength=S).astype(float) if new else 0
            n_ = len(sc_)
            for _ in range(N_RAND):                                                          # budget-matched random control (scorable visits)
                pick = rr.choice(len(ks), n_, replace=False) if n_ else np.array([], int)
                hit = pick[y_sk[pick] & ~np.isin(ks[pick], list(detected))]
                rnd[k].append(len(hit)); rnd_st[k] += np.bincount(EVs.s.values[me][hit], minlength=S) / N_RAND
    rows = []
    for k in gain:
        rd = np.array(rnd[k]).reshape(nv, N_RAND).sum(0)                                 # per draw, summed over rotations
        e, lo, hi = boot_ratio_diff(base_st + gain[k], ev_st, base_st, ev_st, tag=f"7b {name} {thr} {k}")
        rows.append({"design": name, "threshold": thr, "trigger": k, "extra visits/station-yr": cost[k] / nv / n_sy_eval, "scorable share of extra visits": scor[k] / max(cost[k], 1),
                     "new detections per 100 scorable extra visits": 100 * gain[k].sum() / max(scor[k], 1),
                     "random control: per 100 visits (mean)": 100 * rd.mean() / max(scor[k], 1), "random control: 2.5%": 100 * np.percentile(rd, 2.5) / max(scor[k], 1),
                     "random control: 97.5%": 100 * np.percentile(rd, 97.5) / max(scor[k], 1),
                     "network-level recall without follow-up": base_st.sum() / ev_st.sum(), "recall gain": e, "recall gain 95% CI low": lo, "recall gain 95% CI high": hi,
                     "random control: recall gain (mean)": rd.mean() / ev_st.sum(),
                     **{f"recall gain, {g}": gain[k][m].sum() / ev_st[m].sum() for g, m in GRP.items()},
                     **{f"random control recall gain, {g}": rnd_st[k][m].sum() / ev_st[m].sum() for g, m in GRP.items()}})
    return rows
FU = pd.DataFrame([r for name in ("Uniform 4/yr, staggered", f"Risk-based: {N_HI[2]} monthly, others 2/yr") for thr in (0.2, PRIMARY) for r in followup(name, thr)])
save_table(FU, "T28_alarm_triggered_followup"); display(FU[FU.threshold == PRIMARY].set_index(["design", "trigger"]).drop(columns="threshold").T.round(3))
x = FU[FU.threshold == PRIMARY]; fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for a_, name in zip(ax, x.design.unique()):
    z = x[x.design == name]; xx = np.arange(len(z))
    b1 = a_.bar(xx - .2, z["new detections per 100 scorable extra visits"], .38, color="#e34948", label="triggered extra visits")
    b2 = a_.bar(xx + .2, z["random control: per 100 visits (mean)"], .38, color="#a8a69e", label="random extra visits (same number)")
    a_.errorbar(xx + .2, z["random control: per 100 visits (mean)"], yerr=[z["random control: per 100 visits (mean)"] - z["random control: 2.5%"], z["random control: 97.5%"] - z["random control: per 100 visits (mean)"]], fmt="none", ecolor="k", lw=.8)
    a_.bar_label(b1, fmt="%.1f", fontsize=7); a_.bar_label(b2, fmt="%.1f", fontsize=7)
    a_.set_xticks(xx); a_.set_xticklabels(z.trigger, fontsize=8); a_.set_title(name, fontsize=9); a_.set_ylim(0, None)
ax[0].set_ylabel("New recorded exceedances per 100 extra visits"); ax[0].legend(fontsize=7)
fig.suptitle(f"Follow-up visit in the month after a trigger, > {PRIMARY} mg/l N", fontsize=9); fig.tight_layout(); savefig(fig, "F23_alarm_triggered_followup"); plt.show()
