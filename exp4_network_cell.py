# Experiment 4 - network-wide reduction: synchronised vs staggered sampling schedules (block-retrained, seen design).
# Relies on notebook globals: S, V, L, CEN, mon, yr, T, NB_ALL, EVAL, DEVM, okSY, CFG, HGB_GRID, CHOSEN, OWN, NBC,
# cumstats, exceed, n_sy_eval, n_sy_dev, STRATS, COL, LAB, SEED, savefig, save_table
import itertools
TRAIN_FROM = 1995
NET_FREQS = [f for f in CFG["sched"] if f in ((4,) if FAST else (6, 4))]          # network-wide experiment: 6 and 4 visits/yr
def neighbour_graph():
    adj = [set() for _ in range(S)]
    for i in range(S):
        for j, km in NB_ALL[i]: adj[i].add(j); adj[j].add(i)
    return adj
ADJ = neighbour_graph()
def colour(n_phases):
    # Greedy graph colouring: neighbours receive different sampling phases where possible (ties -> least-used phase)
    order = sorted(range(S), key=lambda i: -len(ADJ[i])); ph = -np.ones(S, int); load = np.zeros(n_phases)
    for i in order:
        conf = [sum(ph[j] == p for j in ADJ[i]) for p in range(n_phases)]
        p = min(range(n_phases), key=lambda q: (conf[q], load[q])); ph[i] = p; load[p] += 1
    return ph
def conflicts(ph): return sum(ph[i] == ph[j] for i in range(S) for j in ADJ[i]) // 2
def schemes(f):
    k = len(CFG["sched"][f]); stag = colour(k)
    out = {"one-at-a-time (neighbours monthly)": [("single", np.full(S, p)) for p in range(k)],
           "network: synchronised": [("network", np.full(S, p)) for p in range(k)],
           "network: staggered (graph colouring)": [("network", (stag + r) % k) for r in range(k)],
           # random phases, balanced: each random assignment is used with all k rotations so every station visits every phase equally often
           "network: random phases": [("network", (np.random.RandomState(SEED + r).randint(0, k, S) + q) % k) for r in range(2) for q in range(k)]}
    return out
def availability_matrix(f, phases, start):
    sched = np.zeros((S, T), bool)
    for i in range(S): sched[i] = np.isin(mon, CFG["sched"][f][phases[i]])
    return ~np.isnan(V) & (sched | (yr < start)[None, :])
def last_index(A):
    li = np.full(A.shape, -1)
    for i in range(A.shape[0]):
        idx = np.flatnonzero(A[i]); pos = np.searchsorted(idx, np.arange(A.shape[1]), side="right") - 1
        li[i] = np.where(pos >= 0, idx[np.maximum(pos, 0)], -1)
    return li
def _features(AV, NBAV, s, t, t0):
    # features for target months t with origin t0 (last available own visit); own history from AV, neighbours from NBAV
    own = cumstats(np.where(AV, L, np.nan)); nbs = cumstats(np.where(NBAV, L, np.nan)); nli = last_index(NBAV)
    cs, cn, csm, cnm = own; dcs, dcn, dcsm, dcnm = nbs
    last = L[s, t0]; base = cs[s, t0] / cn[s, t0]; t12 = np.maximum(t0 - 12, 0); n12 = cn[s, t0] - cn[s, t12]
    def clim_(ti):
        mm = mon[ti] - 1; return np.where(cnm[mm, s, t0] >= 2, csm[mm, s, t0] / np.maximum(cnm[mm, s, t0], 1), base)
    ct, co = clim_(t), clim_(t0)
    F = pd.DataFrame({"last": last, "last_censored": CEN[s, t0].astype(float), "h": (t - t0).astype(float), "baseline": base,
                      "mean12": np.where(n12 > 0, (cs[s, t0] - cs[s, t12]) / np.maximum(n12, 1), last), "anomaly": last - base,
                      "clim_target": ct, "clim_origin": co, "seasonal_change": ct - co, "seas_anom": last - co,
                      "tsin": np.sin(2 * np.pi * mon[t] / 12), "tcos": np.cos(2 * np.pi * mon[t] / 12)})
    nbv = {k: np.full(len(s), np.nan) for k in NBC}
    for r in range(len(s)):
        ch, an, va = [], [], []
        for j, km in NB_ALL[s[r]]:
            if not NBAV[j, t[r]]: continue                                   # neighbour must have been sampled in month t
            ref = nli[j, t0[r]]
            mm = mon[t[r]] - 1; den = dcnm[mm, j, t0[r]]
            cj = dcsm[mm, j, t0[r]] / den if den >= 2 else dcs[j, t0[r]] / max(dcn[j, t0[r]], 1)
            c_ = L[j, t[r]] - L[j, ref] if (ref >= 0 and t0[r] - ref <= 3) else np.nan     # change vs neighbour's own last visit <= t0
            ch.append(c_); an.append(L[j, t[r]] - cj); va.append(L[j, t[r]])
            if len(an) == 1: nbv["nb1_change"][r], nbv["nb1_anom"][r], nbv["nb1_value"][r], nbv["nb1_km"][r] = c_, an[0], va[0], km
        if an:
            nbv["nb_mean_change"][r] = np.nanmean(ch) if np.isfinite(ch).any() else np.nan
            nbv["nb_mean_anom"][r], nbv["nb_max_value"][r] = np.mean(an), np.max(va)
        nbv["nb_n"][r] = len(an)
    for k_, v in nbv.items(): F[k_] = v
    F["s"], F["t"], F["year"], F["y"], F["cens_target"], F["group"] = s, t, yr[t], L[s, t], CEN[s, t], groups[s]
    F["Persistence"] = F["last"]; F["Neighbour_ratio"] = F["last"] + np.nan_to_num(F.nb1_change)
    return F

def items_for(f, phases, mode, start, mask, AV=None):
    # skipped (hidden) months inside `mask`: target value is used only for scoring (or, before 2008, as a training label)
    AV = availability_matrix(f, phases, start) if AV is None else AV
    NBAV = AV if mode == "network" else ~np.isnan(V)                       # what neighbours actually sampled
    s_, t_, t0_, ret = [], [], [], []
    for i in range(S):
        idx = np.flatnonzero(AV[i])
        for t in np.flatnonzero(mask[i]):
            if AV[i, t]: ret.append((i, t)); continue
            k = np.searchsorted(idx, t) - 1
            if k < 0: continue
            s_.append(i); t_.append(t); t0_.append(idx[k])
    s, t, t0 = map(np.array, (s_, t_, t0_)); assert np.all(t0 < t) and not AV[s, t].any()
    return _features(AV, NBAV, s, t, t0), pd.DataFrame(ret, columns=["s", "t"])

def observed_pairs(AV, y0, y1):
    # pairs of consecutive visits that were ACTUALLY taken under the design (target t and origin t0 both sampled, t - t0 <= max_h),
    # target year in [y0, y1]; network mode: neighbours = what neighbours sampled under the same design. No hidden month is used.
    s_, t_, t0_ = [], [], []
    for i in range(S):
        idx = np.flatnonzero(AV[i])
        for k in range(1, len(idx)):
            t, t0 = idx[k], idx[k - 1]
            if y0 <= yr[t] <= y1 and t - t0 <= CFG["max_h"]: s_.append(i); t_.append(t); t0_.append(t0)
    if not s_: return None
    s, t, t0 = map(np.array, (s_, t_, t0_)); assert AV[s, t].all() and AV[s, t0].all()
    return _features(AV, AV, s, t, t0)

def train_mask(y0, y1): return okSY[:, yr - 1990] & ((yr >= y0) & (yr <= y1))[None, :] & ~np.isnan(V)
GRIDC = np.round(np.exp(np.linspace(np.log(0.05), np.log(3.0), 200)), 4)
def fa_cutoff(y_log, cens, score_log, thr, n_units):
    yb = exceed(np.exp(y_log), cens, thr); sc = np.exp(score_log)
    fa = np.array([((sc > c) & ~yb).sum() / n_units for c in GRIDC]); ok = np.flatnonzero(fa <= CFG["fa_budget"])
    return float(GRIDC[ok[0]]) if len(ok) else float(GRIDC[-1])
cfg_nb = HGB_GRID[CHOSEN["HistGB_nb"]]; cols_nb = OWN + NBC
n_sy_train_dev = int(okSY[:, CFG["dev"][0] - 1990:CFG["dev"][1] - 1989].sum())

def fit_hgb(X):
    use = [c for c in cols_nb if X[c].notna().any()]       # neighbour columns can be all-missing (e.g. synchronised schedules)
    return HistGradientBoostingRegressor(random_state=SEED, **cfg_nb).fit(X[use].values, (X.y - X["last"]).values), use
def pred_hgb(mu, X): m, use = mu; return m.predict(X[use].values) + X["last"].values
RETRAIN_LOG = []
def train_and_predict(TR, EV, POST, label):
    # Same temporal/spatial rules as Experiments 1-3 (seen design):
    #   development model: simulated skipped months < 2005 -> cut-offs on 2005-2007 (unchanged);
    #   evaluation: one model per 3-year block B and spatial cluster g, trained on the schedule simulated over 1995-2007 PLUS the visit pairs
    #   actually taken under the design in 2008..B-1, excluding cluster g's post-2007 pairs. Hidden months never enter training.
    trA, dev = TR[TR.year < CFG["dev"][0]], TR[TR.year.between(*CFG["dev"])]
    assert len(trA) > 100 and len(dev) > 100, "too few simulated training items"
    dev = dev.assign(HistGB_nb=pred_hgb(fit_hgb(trA), dev))
    m0 = fit_hgb(TR); EV = EV.copy(); EV["HistGB_nb_once"] = pred_hgb(m0, EV)           # previous procedure (trained once), kept for comparison
    pred = np.full(len(EV), np.nan); evb = EV.year.map(block_of).values; evg = EV.group.values
    for B in CFG["blocks"]:
        add_B = POST[POST.year < B] if POST is not None else None
        for g in range(CFG["n_groups"]):
            sel = (evb == B) & (evg == g)
            if not sel.any(): continue
            add = add_B[add_B.group != g] if add_B is not None else None
            if add is None or len(add) == 0: mu = m0
            else:
                tr = pd.concat([TR, add], ignore_index=True); assert tr.t.max() < np.flatnonzero(yr == B)[0]; mu = fit_hgb(tr)
            pred[sel] = pred_hgb(mu, EV[sel])
            RETRAIN_LOG.append({"design": label, "block": B, "cluster": g, "simulated pre-2008 items": len(TR), "post-2007 observed pairs added": 0 if add is None else len(add)})
    assert np.isfinite(pred).all(); EV["HistGB_nb"] = pred
    return EV, dev

def run_scheme(f, vectors, label=""):
    parts = []
    if vectors[0][0] == "single":
        # one-at-a-time reference = Experiment 1 itself (same items, block-retrained seen-design predictions, development items)
        for p in range(len(vectors)):
            EV, RET = ITEMS[(f, p)]; DV, _ = DEV_ITEMS[(f, p)]
            parts.append((EV.assign(HistGB_nb_once=EV.HistGB_nb), RET, DV, p))
        return parts
    for v_i, (mode, ph) in enumerate(vectors):
        TR, _ = items_for(f, ph, mode, TRAIN_FROM, train_mask(TRAIN_FROM, CFG["dev"][1]))       # schedule simulated on 1995-2007
        EV, RET = items_for(f, ph, mode, CFG["eval"][0], EVAL)                                  # evaluation 2008-2024 (hidden months)
        POST = observed_pairs(availability_matrix(f, ph, CFG["eval"][0]), CFG["eval"][0], CFG["eval"][1] - 1)
        EV, dev = train_and_predict(TR, EV, POST, f"{label} {f}/yr v{v_i}")
        parts.append((EV, RET, dev, v_i))
    return parts

def evaluate_scheme(parts, f, thr):
    nv = len(parts); rows = {}
    EVs = pd.concat([p[0].assign(v=p[3]) for p in parts], ignore_index=True); RETs = pd.concat([p[1].assign(v=p[3]) for p in parts], ignore_index=True)
    DEVs = pd.concat([p[2].assign(v=p[3]) for p in parts], ignore_index=True)
    yb = exceed(np.exp(EVs.y.values), EVs.cens_target.values, thr)
    rev = exceed(V[RETs.s.values, RETs.t.values], CEN[RETs.s.values, RETs.t.values], thr); tot = yb.sum() + rev.sum()
    out = {"skipped months with a neighbour sampled in the same month": (EVs.nb_n > 0).mean(),
           "Observation-only recall": rev.sum() / tot}
    for s_ in ("Persistence", "Neighbour_ratio", "HistGB_nb"):
        c = fa_cutoff(DEVs.y.values, DEVs.cens_target.values, DEVs[s_].values, thr, nv * n_sy_train_dev)
        a = np.exp(EVs[s_].values) > c
        out[f"{s_} PR-AUC"] = average_precision_score(yb, EVs[s_]); out[f"{s_} recall"] = ((a & yb).sum() + rev.sum()) / tot
        out[f"{s_} false alarms/station-yr"] = (a & ~yb).sum() / nv / n_sy_eval; out[f"{s_} cut-off"] = c
        EVs[f"alarm_{s_}"] = a
    a = np.exp(EVs["HistGB_nb_once"].values) > out["HistGB_nb cut-off"]                  # previous procedure, same development cut-off
    out["HistGB_nb (trained once) PR-AUC"] = average_precision_score(yb, EVs["HistGB_nb_once"]); out["HistGB_nb (trained once) recall"] = ((a & yb).sum() + rev.sum()) / tot
    return out, EVs, RETs, rev

NYR = CFG["eval"][1] - CFG["eval"][0] + 1
def unit_keys(EVs, RETs, cluster):
    # bootstrap units: whole stations or whole calendar years
    if cluster == "station": return EVs.s.values, RETs.s.values, S
    return EVs.year.values - CFG["eval"][0], yr[RETs.t.values] - CFG["eval"][0], NYR
def station_recall(EVs, RETs, rev, thr, s_, cluster="station"):
    ke, kr, K = unit_keys(EVs, RETs, cluster)
    yb = exceed(np.exp(EVs.y.values), EVs.cens_target.values, thr); a = EVs[f"alarm_{s_}"].values
    tp = np.bincount(ke, (a & yb).astype(float), K) + np.bincount(kr, rev.astype(float), K)
    ev = np.bincount(ke, yb.astype(float), K) + np.bincount(kr, rev.astype(float), K)
    return tp, ev

t_ = time.time(); NET = {}; NET_ROWS = []
for f in NET_FREQS:
    sch = schemes(f)
    for name, vecs in sch.items():
        parts = run_scheme(f, vecs, name); NET[(f, name)] = parts
        for thr in (0.2, 0.3, 0.6):
            o, _, _, _ = evaluate_scheme(parts, f, thr)
            NET_ROWS.append({"visits/yr": f, "scheme": name, "threshold": thr,
                             "neighbour-phase conflicts": int(np.mean([conflicts(ph) for _, ph in vecs])) if name.startswith("network") else np.nan, **o})
NETT = pd.DataFrame(NET_ROWS); save_table(NETT, "T15_network_schedules")
RLOG = pd.DataFrame(RETRAIN_LOG)
print("post-2007 observed visit pairs added to training (median over clusters, last block):",
      RLOG[RLOG.block == CFG["blocks"][-1]].groupby("design")["post-2007 observed pairs added"].median().astype(int).to_dict())
ONCE = NETT[["visits/yr", "scheme", "threshold", "HistGB_nb PR-AUC", "HistGB_nb (trained once) PR-AUC", "HistGB_nb recall", "HistGB_nb (trained once) recall"]]
save_table(ONCE, "T15c_network_retraining_vs_trained_once"); display(ONCE[ONCE.threshold == PRIMARY].round(3))
print(f"network-schedule experiment: {time.time() - t_:.0f}s; neighbour edges: {sum(len(a) for a in ADJ) // 2}")
show_cols = ["visits/yr", "scheme", "neighbour-phase conflicts", "skipped months with a neighbour sampled in the same month", "Observation-only recall",
             "Persistence recall", "Neighbour_ratio recall", "HistGB_nb recall", "HistGB_nb false alarms/station-yr", "Persistence PR-AUC", "Neighbour_ratio PR-AUC", "HistGB_nb PR-AUC"]
display(NETT[NETT.threshold == PRIMARY][show_cols].round(3))

# paired station-cluster bootstrap: staggered minus synchronised (recall of recorded exceedances, HistGB + neighbours and ratio rule)
bb = []
for f in NET_FREQS:
    for thr in (0.2, 0.3):
        res_ = {}
        for name in ("network: staggered (graph colouring)", "network: synchronised", "one-at-a-time (neighbours monthly)"):
            o, EVs, RETs, rev = evaluate_scheme(NET[(f, name)], f, thr)
            res_[name] = {(s_, cl): station_recall(EVs, RETs, rev, thr, s_, cl) for s_ in ("Persistence", "Neighbour_ratio", "HistGB_nb") for cl in ("station", "year")}
        for cl in ("station", "year"):
            K = S if cl == "station" else NYR; dd = np.random.RandomState(SEED).randint(0, K, (CFG["n_boot"], K))
            for a_name, b_name in (("network: staggered (graph colouring)", "network: synchronised"), ("one-at-a-time (neighbours monthly)", "network: staggered (graph colouring)")):
                for s_ in ("Neighbour_ratio", "HistGB_nb"):
                    ta, ea = res_[a_name][(s_, cl)]; tb, eb = res_[b_name][(s_, cl)]
                    boot = ta[dd].sum(1) / ea[dd].sum(1) - tb[dd].sum(1) / eb[dd].sum(1)
                    bb.append({"visits/yr": f, "threshold": thr, "strategy": s_, "comparison": f"{a_name}  minus  {b_name}", "cluster": cl,
                               "d recall": ta.sum() / ea.sum() - tb.sum() / eb.sum(), "95% CI low": np.percentile(boot, 2.5), "95% CI high": np.percentile(boot, 97.5)})
# station-cluster bootstrap of PR-AUC differences on skipped months (HistGB + neighbours)
def boot_ap_schemes(EA, EB, thr, B=None, seed=SEED):
    B = CFG["n_boot"] if B is None else B; r = np.random.RandomState(seed)
    ya, yb_ = exceed(np.exp(EA.y.values), EA.cens_target.values, thr), exceed(np.exp(EB.y.values), EB.cens_target.values, thr)
    ma = [np.flatnonzero(EA.s.values == i) for i in range(S)]; mb = [np.flatnonzero(EB.s.values == i) for i in range(S)]; out = []
    for _ in range(B):
        d_ = r.randint(0, S, S); ia = np.concatenate([ma[i] for i in d_]); ib = np.concatenate([mb[i] for i in d_])
        if ya[ia].sum() == 0 or yb_[ib].sum() == 0: continue
        out.append(average_precision_score(ya[ia], EA.HistGB_nb.values[ia]) - average_precision_score(yb_[ib], EB.HistGB_nb.values[ib]))
    return average_precision_score(ya, EA.HistGB_nb) - average_precision_score(yb_, EB.HistGB_nb), np.percentile(out, 2.5), np.percentile(out, 97.5)
for f in NET_FREQS:
    for thr in (0.2, 0.3):
        E = {n: pd.concat([p[0] for p in NET[(f, n)]], ignore_index=True) for n in ("network: staggered (graph colouring)", "network: synchronised", "network: random phases", "one-at-a-time (neighbours monthly)")}
        for a_name, b_name in (("network: staggered (graph colouring)", "network: synchronised"), ("network: random phases", "network: synchronised"),
                               ("network: staggered (graph colouring)", "network: random phases"), ("one-at-a-time (neighbours monthly)", "network: staggered (graph colouring)")):
            e, lo, hi = boot_ap_schemes(E[a_name], E[b_name], thr)
            bb.append({"visits/yr": f, "threshold": thr, "strategy": "HistGB_nb (PR-AUC)", "comparison": f"{a_name}  minus  {b_name}", "cluster": "station", "d recall": np.nan,
                       "d PR-AUC": e, "95% CI low": lo, "95% CI high": hi})
NETB = pd.DataFrame(bb); save_table(NETB, "T16_network_schedules_bootstrap"); display(NETB.round(4))

fig, ax = plt.subplots(2, len(NET_FREQS), figsize=(6.5 * len(NET_FREQS), 8.2), squeeze=False)
order_ = ["network: synchronised", "network: random phases", "network: staggered (graph colouring)", "one-at-a-time (neighbours monthly)"]
short = {"network: synchronised": "synchronised", "network: random phases": "random phases", "network: staggered (graph colouring)": "staggered\n(graph colouring)", "one-at-a-time (neighbours monthly)": "one-at-a-time\n(neighbours monthly)"}
COVC = "skipped months with a neighbour sampled in the same month"
for j, f in enumerate(NET_FREQS):
    x = NETT[(NETT["visits/yr"] == f) & (NETT.threshold == PRIMARY)].set_index("scheme").loc[order_]; xx = np.arange(len(x))
    labs = [f"{short[s]}\nnb coverage {c:.0%}" for s, c in zip(order_, x[COVC])]
    # (top) threshold-free ranking: PR-AUC
    a_ = ax[0, j]; w = .26
    for i, (c, lab, cc) in enumerate((("Persistence PR-AUC", "Persistence", COL["Persistence"]), ("Neighbour_ratio PR-AUC", "Neighbour ratio", COL["Neighbour_ratio"]),
                                      ("HistGB_nb PR-AUC", "HistGB + neighbours", COL["HistGB_nb"]))):
        b = a_.bar(xx + (i - 1) * w, x[c], w * .92, color=cc, label=lab)
        a_.bar_label(b, fmt="%.3f", fontsize=6.5, padding=1.5)
    a_.set_xticks(xx); a_.set_xticklabels(labs, fontsize=7.5); a_.set_ylim(0, x[["Neighbour_ratio PR-AUC", "HistGB_nb PR-AUC"]].values.max() * 1.18)
    a_.set_ylabel("PR-AUC (skipped months)"); a_.set_title(f"({'ab'[j]}) PR-AUC, {f} visits/yr at every station, > {PRIMARY} mg/l N")
    # (bottom) operational recall at the pre-selected FA-budget cut-off
    a_ = ax[1, j]; w = .2
    for i, (c, lab, cc) in enumerate((("Observation-only recall", "Observation only", COL["Observation-only"]), ("Persistence recall", "Persistence", COL["Persistence"]),
                                      ("Neighbour_ratio recall", "Neighbour ratio", COL["Neighbour_ratio"]), ("HistGB_nb recall", "HistGB + neighbours", COL["HistGB_nb"]))):
        b = a_.bar(xx + (i - 1.5) * w, x[c], w * .92, color=cc, label=lab); a_.bar_label(b, fmt="%.3f", fontsize=6, padding=1.5, rotation=90)
    a_.set_xticks(xx); a_.set_xticklabels(labs, fontsize=7.5)
    a_.set_ylim(0, x[[c for c in x.columns if c.endswith("recall")]].values.max() * 1.38)         # bars start at zero (no truncated axis)
    a_.set_ylabel("Recall at FA-budget cut-off"); a_.set_title(f"({'cd'[j]}) Recall, {f} visits/yr at every station, > {PRIMARY} mg/l N")
ax[0, 0].legend(fontsize=7, loc="upper left"); ax[1, 0].legend(fontsize=7, ncol=2, loc="upper left")
fig.tight_layout(); savefig(fig, "F14_network_schedules"); plt.show()
