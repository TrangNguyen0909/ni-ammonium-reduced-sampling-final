# ==== CELL 1 ==== Experiment 5a - detection at the station-year level (one-at-a-time design, cut-offs from Experiment 1)
# A station-year has an event if >= 1 recorded exceedance occurs in its evaluation months; it is DETECTED if >= 1 of those
# exceedances is observed (visited month) or alarmed (skipped month, pre-selected FA-budget cut-off).
def station_year_table(f, thr, rule="FA-budget"):
    rows = []
    for p in range(len(CFG["sched"][f])):
        IT, RET = ITEMS[(f, p)]
        e_ret = exceed(V[RET.s.values, RET.t.values], CEN[RET.s.values, RET.t.values], thr)
        base = pd.DataFrame({"s": RET.s.values, "year": yr[RET.t.values], "event": e_ret})
        yb = labels(IT, thr); sk = pd.DataFrame({"s": IT.s.values, "year": IT.year.values, "event": yb})
        for s_ in ["Observation-only"] + STRATS:
            a_sk = np.zeros(len(IT), bool) if s_ == "Observation-only" else np.exp(IT[s_].values) > CUTS[(f, thr)][(s_, rule)]
            m = pd.concat([base.assign(alarm=e_ret), sk.assign(alarm=a_sk)], ignore_index=True); m["tp"] = m.alarm & m.event
            g = m.groupby(["s", "year"]).agg(event=("event", "any"), tp=("tp", "any"), alarm=("alarm", "any"))
            rows.append(g.assign(strategy=s_, phase=p).reset_index())
    return pd.concat(rows, ignore_index=True)
def sy_summary(Tt):
    g = Tt.groupby("strategy"); nph = Tt.phase.nunique(); nyr = CFG["eval"][1] - CFG["eval"][0] + 1
    out = pd.DataFrame({"station-years with event": g.event.sum() / nph,
                        "detected (station-year recall)": g.apply(lambda x: x.tp.sum() / x.event.sum(), include_groups=False),
                        "flagged station-years without event / yr": g.apply(lambda x: (x.alarm & ~x.event).sum() / nph, include_groups=False) / nyr})
    return out.loc[["Observation-only"] + STRATS]
def boot_sy(Tt, a, b, B=None, seed=SEED):
    B = CFG["n_boot"] if B is None else B
    agg = Tt.groupby(["strategy", "s"]).agg(tp=("tp", "sum"), ev=("event", "sum")).unstack(0).fillna(0)
    tpa, tpb, ev = agg[("tp", a)].values, agg[("tp", b)].values, agg[("ev", a)].values
    idx = np.random.RandomState(seed).randint(0, len(ev), (B, len(ev)))
    d_ = tpa[idx].sum(1) / ev[idx].sum(1) - tpb[idx].sum(1) / ev[idx].sum(1)
    return tpa.sum() / ev.sum() - tpb.sum() / ev.sum(), np.percentile(d_, 2.5), np.percentile(d_, 97.5)
SY_ROWS, SY_BOOT = [], []
for f in CFG["sched"]:
    for thr in (0.2, PRIMARY):
        Tt = station_year_table(f, thr); SY_ROWS.append(sy_summary(Tt).reset_index().assign(**{"visits/yr": f, "threshold": thr}))
        for a, b in (("HistGB_nb", "Observation-only"), ("HistGB_nb", "Persistence"), ("Neighbour_ratio", "Persistence")):
            e, lo, hi = boot_sy(Tt, a, b)
            SY_BOOT.append({"visits/yr": f, "threshold": thr, "comparison": f"{a} - {b}", "d station-year recall": e, "95% CI low": lo, "95% CI high": hi})
SYT = pd.concat(SY_ROWS, ignore_index=True); SYB = pd.DataFrame(SY_BOOT)
save_table(SYT, "T17_station_year_detection"); save_table(SYB, "T18_station_year_detection_bootstrap")
display(SYT[SYT.threshold == PRIMARY].set_index(["visits/yr", "strategy"]).round(3)); display(SYB[SYB.threshold == PRIMARY].round(3))

# ==== CELL 2 ==== Experiment 5b - risk ranking from pre-2008 data and its stability (raw vs Beta-Binomial shrinkage)
from scipy.special import betaln
from scipy.optimize import minimize
from scipy.stats import spearmanr
def exc_counts(y0_, y1_, thr):
    hist = (yr >= y0_) & (yr <= y1_); k = np.zeros(S); n = np.zeros(S)
    for i in range(S):
        m = hist & ~np.isnan(V[i]); n[i] = m.sum(); k[i] = exceed(V[i, m], CEN[i, m], thr).sum()
    return k, n
def eb_beta_binomial(k, n):
    # empirical Bayes: maximise the Beta-Binomial marginal likelihood over stations with >= 1 sample
    m = n > 0
    def nll(p):
        a, b = np.exp(p); return -np.sum(betaln(k[m] + a, n[m] - k[m] + b) - betaln(a, b))
    r0 = max(k[m].sum() / n[m].sum(), 1e-3)
    res = minimize(nll, np.log([r0 * 20, (1 - r0) * 20]), method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-9})
    return tuple(np.exp(res.x))
def risk_score(y0_, y1_, method, thr=PRIMARY):
    k, n = exc_counts(y0_, y1_, thr)
    if method == "raw": return np.where(n > 0, k / np.maximum(n, 1), 0.0), k, n, (np.nan, np.nan)
    a, b = eb_beta_binomial(k, n); return (k + a) / (n + a + b), k, n, (a, b)
REST = (2,) if FAST else (3, 2, 1); N_HI = {r_: int(np.floor((4 - r_) * S / (12 - r_))) for r_ in REST}      # mean budget <= 4 visits/station-yr
RANKS = {}
for (a_, b_) in ((1995, 2000), (2001, 2007), (1995, 2007), (1995, 2004)):
    for meth in ("raw", "eb"):
        sc, k, n, ab = risk_score(a_, b_, meth); RANKS[(meth, a_, b_)] = dict(order=np.lexsort((-n, -sc)), score=sc, k=k, n=n, ab=ab)
PRIMARY_RANK, DEV_RANK = ("eb", 1995, 2007), ("eb", 1995, 2004)
ev_st = np.array([exceed(V[i, EVAL[i]], CEN[i, EVAL[i]], PRIMARY).sum() for i in range(S)])
nh = N_HI[2]; top = lambda key, n_=nh: set(RANKS[key]["order"][:n_].tolist())
rank_label = lambda key: f"{'Beta-Binomial' if key[0] == 'eb' else 'raw rate'} {key[1]}-{key[2]}"
rows = []
for key, R_ in RANKS.items():
    if key[1:] == (1995, 2004): continue
    tp_ = list(top(key)); sh = ev_st[tp_].sum() / ev_st.sum()
    rows.append({"ranking": rank_label(key), "alpha, beta": "-" if key[0] == "raw" else f"{R_['ab'][0]:.2f}, {R_['ab'][1]:.1f}",
                 f"Jaccard of top {nh} with primary": len(top(key) & top(PRIMARY_RANK)) / len(top(key) | top(PRIMARY_RANK)),
                 "Spearman with primary": spearmanr(R_["score"], RANKS[PRIMARY_RANK]["score"]).correlation,
                 f"min. samples among top {nh}": int(R_["n"][tp_].min()), f"share of 2008-2024 events at top {nh}": sh,
                 **{f"observation-only recall, {N_HI[r_]} monthly + {r_}/yr": (ev_st[list(top(key, N_HI[r_]))].sum() / ev_st.sum()) * (1 - r_ / 12) + r_ / 12 for r_ in REST}})
RSTAB = pd.DataFrame(rows); save_table(RSTAB, "T19_risk_ranking_stability"); display(RSTAB.round(3))
keys_ = [k_ for k_ in RANKS if k_[1:] != (1995, 2004)]
J = np.array([[len(top(a) & top(b)) / len(top(a) | top(b)) for b in keys_] for a in keys_])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
im = ax[0].imshow(J, vmin=0, vmax=1, cmap="Blues"); ax[0].grid(False)
ax[0].set_xticks(range(len(keys_))); ax[0].set_xticklabels([rank_label(k_) for k_ in keys_], fontsize=7, rotation=35, ha="right")
ax[0].set_yticks(range(len(keys_))); ax[0].set_yticklabels([rank_label(k_) for k_ in keys_], fontsize=7)
for i in range(len(keys_)):
    for j in range(len(keys_)): ax[0].text(j, i, f"{J[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if J[i, j] > .6 else "k")
plt.colorbar(im, ax=ax[0], shrink=.8, label=f"Jaccard overlap of top-{nh} sets"); ax[0].set_title(f"(a) Overlap of the {nh} highest-risk stations")
Rr, Re = RANKS[("raw", 1995, 2007)], RANKS[PRIMARY_RANK]
ax[1].scatter(Rr["n"], Rr["score"] * 100, s=14, color="#a8a69e", label="raw rate")
ax[1].scatter(Re["n"], Re["score"] * 100, s=14, color="#e34948", label="Beta-Binomial posterior mean")
for i in range(S): ax[1].plot([Rr["n"][i], Re["n"][i]], [Rr["score"][i] * 100, Re["score"][i] * 100], color="#d0d0d0", lw=.4, zorder=0)
ax[1].set_xlabel("Monthly samples 1995-2007"); ax[1].set_ylabel(f"Exceedance rate > {PRIMARY} mg/l (%)"); ax[1].legend(fontsize=7)
ax[1].set_title("(b) Shrinkage of rates based on few samples")
fig.tight_layout(); savefig(fig, "F16_risk_ranking_stability"); plt.show()

# ==== CELL 3 ==== Experiment 5c - fair comparison: uniform versus risk-based allocation at the same budget and alarm standard
def alloc_av(order, rest, start, rot, _cache={}):
    k_ = 12 // rest; phases = [[q + 1 + j * k_ for j in range(rest)] for q in range(k_)]
    if k_ not in _cache: _cache[k_] = colour(k_)                                   # staggered phases among neighbours
    ph = (_cache[k_] + rot) % k_; hi = np.zeros(S, bool); hi[order[:N_HI[rest]]] = True
    sch = np.zeros((S, T), bool)
    for i in range(S): sch[i] = True if hi[i] else np.isin(mon, phases[ph[i]])
    return ~np.isnan(V) & (sch | (yr < start)[None, :])
def run_alloc(rest):
    # same training rules as Experiment 4: schedule simulated on 1995-2007 + visit pairs actually taken under the allocation in 2008..B-1,
    # one model per 3-year block and spatial cluster (seen design); development model and cut-offs unchanged (1995-2004 ranking, 2005-2007)
    parts = []
    for rot in range(12 // rest):
        TR, _ = items_for(None, None, "network", TRAIN_FROM, train_mask(TRAIN_FROM, CFG["dev"][1]), AV=alloc_av(RANKS[DEV_RANK]["order"], rest, TRAIN_FROM, rot))
        AVe = alloc_av(RANKS[PRIMARY_RANK]["order"], rest, CFG["eval"][0], rot)
        EV, RET = items_for(None, None, "network", CFG["eval"][0], EVAL, AV=AVe)
        POST = observed_pairs(AVe, CFG["eval"][0], CFG["eval"][1] - 1)
        EV, dev = train_and_predict(TR, EV, POST, f"risk-based rest {rest}/yr rot {rot}")
        parts.append((EV, RET, dev, rot))
    return parts
t_ = time.time(); ALLOC = {r_: run_alloc(r_) for r_ in REST}; print(f"risk-based allocation runs: {time.time() - t_:.0f}s")
DET = (("Persistence", "Persistence"), ("Neighbour_ratio", "Neighbour-ratio rule"), ("HistGB_nb", "HistGB + neighbours"))
def summarise(parts, thr, design, budget):
    o, EVs, RETs, rev = evaluate_scheme(parts, None, thr)
    yb = exceed(np.exp(EVs.y.values), EVs.cens_target.values, thr); R = rev.sum()
    out = [{"design": design, "mean visits/station-yr": budget, "threshold": thr, "detection": "Observation only", "recall": o["Observation-only recall"],
            "precision": 1.0, "false alarms/station-yr": 0.0, "cut-off (mg/l)": np.nan, "skipped months with same-month neighbour": o["skipped months with a neighbour sampled in the same month"]}]
    for s_, lab in DET:
        a = EVs[f"alarm_{s_}"].values; tp = (a & yb).sum() + R
        out.append({"design": design, "mean visits/station-yr": budget, "threshold": thr, "detection": lab, "recall": o[f"{s_} recall"], "precision": tp / (a.sum() + R),
                    "false alarms/station-yr": o[f"{s_} false alarms/station-yr"], "cut-off (mg/l)": o[f"{s_} cut-off"],
                    "skipped months with same-month neighbour": o["skipped months with a neighbour sampled in the same month"]})
    return out, (EVs, RETs, rev)
DESIGNS = [("Uniform 4/yr, synchronised", NET[(4, "network: synchronised")], 4.0), ("Uniform 4/yr, staggered", NET[(4, "network: staggered (graph colouring)")], 4.0)] + \
          [(f"Risk-based: {N_HI[r_]} monthly, others {r_}/yr", ALLOC[r_], (12 * N_HI[r_] + r_ * (S - N_HI[r_])) / S) for r_ in REST]
FAIR_ROWS, FAIR_EV = [], {}
for thr in (0.2, PRIMARY):
    for name, parts, bud in DESIGNS:
        rows_, ev_ = summarise(parts, thr, name, bud); FAIR_ROWS += rows_; FAIR_EV[(name, thr)] = ev_
FAIR = pd.DataFrame(FAIR_ROWS); save_table(FAIR, "T20_fair_allocation_comparison")
display(FAIR[FAIR.threshold == PRIMARY].set_index(["design", "detection"]).drop(columns="threshold").round(3))
def st_counts(name, thr, det, cluster="station"):
    EVs, RETs, rev = FAIR_EV[(name, thr)]; yb = exceed(np.exp(EVs.y.values), EVs.cens_target.values, thr); ke, kr, K = unit_keys(EVs, RETs, cluster)
    tp = np.bincount(kr, rev.astype(float), K); ev = tp + np.bincount(ke, yb.astype(float), K)
    if det != "Observation only":
        s_ = {lab: k for k, lab in DET}[det]; tp = tp + np.bincount(ke, (EVs[f"alarm_{s_}"].values & yb).astype(float), K)
    return tp, ev
fb = []; U = "Uniform 4/yr, staggered"
for cl in ("station", "year"):
    K = S if cl == "station" else NYR; dd = np.random.RandomState(SEED).randint(0, K, (CFG["n_boot"], K))
    for thr in (0.2, PRIMARY):
        for r_ in REST:
            Rn = f"Risk-based: {N_HI[r_]} monthly, others {r_}/yr"
            for (na, da), (nb_, db) in (((Rn, "HistGB + neighbours"), (U, "HistGB + neighbours")), ((Rn, "Observation only"), (U, "HistGB + neighbours")),
                                        ((Rn, "Persistence"), (U, "Persistence")), ((Rn, "HistGB + neighbours"), (Rn, "Observation only"))):
                ta, ea = st_counts(na, thr, da, cl); tb, eb = st_counts(nb_, thr, db, cl)
                bs_ = ta[dd].sum(1) / ea[dd].sum(1) - tb[dd].sum(1) / eb[dd].sum(1)
                fb.append({"threshold": thr, "cluster": cl, "comparison": f"{na} [{da}]  minus  {nb_} [{db}]", "d recall": ta.sum() / ea.sum() - tb.sum() / eb.sum(),
                           "95% CI low": np.percentile(bs_, 2.5), "95% CI high": np.percentile(bs_, 97.5)})
FAIRB = pd.DataFrame(fb); save_table(FAIRB, "T20b_fair_allocation_bootstrap"); display(FAIRB[FAIRB.threshold == PRIMARY].round(3))
ONCE5 = pd.DataFrame([{"design": n, "threshold": thr, "HistGB_nb recall (block-retrained)": evaluate_scheme(p_, None, thr)[0]["HistGB_nb recall"],
                       "HistGB_nb recall (trained once)": evaluate_scheme(p_, None, thr)[0]["HistGB_nb (trained once) recall"]} for n, p_, _ in DESIGNS for thr in (0.2, PRIMARY)])
save_table(ONCE5, "T20c_allocation_retraining_vs_trained_once"); display(ONCE5.round(3))
dets = ["Observation only", "Persistence", "Neighbour-ratio rule", "HistGB + neighbours"]; dcol = [COL["Observation-only"], COL["Persistence"], COL["Neighbour_ratio"], COL["HistGB_nb"]]
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6), sharey=True)
for a_, thr in zip(ax, (0.2, PRIMARY)):
    x = FAIR[FAIR.threshold == thr]; names = [n for n, _, _ in DESIGNS]; xx = np.arange(len(names)); w = .2
    for i, (dt, cc) in enumerate(zip(dets, dcol)):
        y_ = [x[(x.design == n) & (x.detection == dt)].recall.iloc[0] for n in names]
        b = a_.bar(xx + (i - 1.5) * w, y_, w * .92, color=cc, label=dt); a_.bar_label(b, fmt="%.2f", fontsize=6, padding=1)
    a_.set_xticks(xx); a_.set_xticklabels([n.replace(": ", ":\n").replace(", ", ",\n") for n in names], fontsize=7.5)
    a_.set_title(f"> {thr} mg/l N; budget <= 4 visits/station-yr; cut-offs fixed on 2005-2007"); a_.set_ylim(0, 1)
ax[0].set_ylabel("Recall of recorded exceedances (2008-2024)"); ax[0].legend(fontsize=7, ncol=2, loc="upper left")
fig.tight_layout(); savefig(fig, "F17_fair_allocation_comparison"); plt.show()
