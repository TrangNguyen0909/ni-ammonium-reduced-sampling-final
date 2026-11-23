# ==== CELL R1 ==== 14.6 (revision) Realised sampling effort and detection at event, station-year and station level
# Observation-only detection is exact (no model, no false alarms) and comparable across designs. Rankings, station groups and
# schedules are those of Sections 14.2-14.4; nothing is re-selected on evaluation data. Results are averaged over all schedule rotations.
Y0, Y1 = CFG["eval"]; ORDER = RANKS[PRIMARY_RANK]["order"]
SYM = okSY[:, yr - 1990] & ((yr >= Y0) & (yr <= Y1))[None, :]            # every month of the evaluated (complete) station-years
EXC_ALL = exceed(np.nan_to_num(V, nan=0.0), CEN, PRIMARY) & ~np.isnan(V)  # recorded exceedances, all months
EXC = EXC_ALL & EVAL                                                      # recorded exceedances in the evaluation months
EV_COLS = np.flatnonzero((yr >= Y0) & (yr <= Y1)); NY = Y1 - Y0 + 1
def sched_static(n_hi, rest, rot, order=ORDER):
    # scheduled months in 2008-2024: n_hi highest-risk stations monthly, the others `rest` visits/yr on staggered phases
    k_ = 12 // rest; phases = [[q + 1 + j * k_ for j in range(rest)] for q in range(k_)]
    ph = (colour(k_) + rot) % k_; hi = np.zeros(S, bool); hi[order[:n_hi]] = True
    sch = np.zeros((S, T), bool)
    for i in range(S): sch[i] = True if hi[i] else np.isin(mon, phases[ph[i]])
    return sch & ((yr >= Y0) & (yr <= Y1))[None, :], hi
def sched_reranked(n_hi, rest, rot):
    # as Section 14.4: the ranking is updated every three years from the samples actually taken under the reduced schedule
    k_ = 12 // rest; phases = [[q + 1 + j * k_ for j in range(rest)] for q in range(k_)]
    pre = ~np.isnan(V) & (yr < Y0)[None, :]; sch = np.zeros((S, T), bool)
    for B in range(Y0, Y1 + 1, 3):
        seen = (pre | (sch & ~np.isnan(V))) & ((yr >= 1995) & (yr < B))[None, :] if B > Y0 else ~np.isnan(V) & ((yr >= 1995) & (yr <= Y0 - 1))[None, :]
        order = np.argsort(-eb_score((EXC_ALL & seen).sum(1).astype(float), seen.sum(1).astype(float)), kind="stable")
        hi = np.zeros(S, bool); hi[order[:n_hi]] = True
        lo_rank = {i: r for r, i in enumerate([i for i in order if not hi[i]])}
        blk = (yr >= B) & (yr <= min(B + 2, Y1))
        for i in range(S): sch[i, blk] = True if hi[i] else np.isin(mon[blk], phases[(lo_rank[i] + rot) % k_])
    return sch, None
def per_station(masks):
    # per-station counts averaged over rotations: events, station-years with an event, station with any event; and how many are observed
    ev_t = EXC.sum(1).astype(float); sy_e = EXC[:, EV_COLS].reshape(S, NY, 12).any(2); sy_t = sy_e.sum(1).astype(float); st_t = EXC.any(1).astype(float)
    ev_d, sy_d, st_d = np.zeros(S), np.zeros(S), np.zeros(S)
    for sch in masks:
        D = EXC & sch; ev_d += D.sum(1); sy_d += D[:, EV_COLS].reshape(S, NY, 12).any(2).sum(1); st_d += D.any(1)
    n = len(masks); return dict(ev=(ev_t, ev_d / n), sy=(sy_t, sy_d / n), st=(st_t, st_d / n))
LEVELS = {"ev": "recorded exceedances", "sy": "station-years with an exceedance", "st": "stations with an exceedance"}
DES6 = [("Uniform 4/yr, staggered", 0, 4, "static")] + [(f"Risk-based: {N_HI[r_]} monthly, others {r_}/yr", N_HI[r_], r_, "static") for r_ in REST]
if 2 in REST: DES6.append((f"Risk-based: {N_HI[2]} monthly, others 2/yr, re-ranked every 3 yr", N_HI[2], 2, "reranked"))
HI56 = np.zeros(S, bool); HI56[ORDER[:N_HI[2]]] = True
EMERG_R = (EXC_ALL & ~np.isnan(V) & ((yr >= 1995) & (yr < Y0))[None, :]).sum(1) == 0
EMERG_R &= EXC.any(1)
GROUPS_R = {"all stations": np.ones(S, bool), f"{HI56.sum()} pre-2008 high-risk stations": HI56, f"{S - HI56.sum()} other stations": ~HI56,
            f"'emerging' stations (n={EMERG_R.sum()})": EMERG_R}
PS, EFF = {}, []
for name, n_hi, rest, kind in DES6:
    masks, his = [], []
    for rot in range(12 // rest):
        sch, hi = sched_static(n_hi, rest, rot) if kind == "static" else sched_reranked(n_hi, rest, rot)
        masks.append(sch); his.append(hi)
    PS[name] = per_station(masks)
    if kind == "static":
        hi = his[0]
        EFF.append({"design": name, "nominal visits per station-year (equal station-years)": (12 * n_hi + rest * (S - n_hi)) / S,
                    "planned visits per evaluated station-year": np.mean([(m & SYM).sum() for m in masks]) / n_sy_eval,
                    "realised visits per evaluated station-year": np.mean([(m & EVAL).sum() for m in masks]) / n_sy_eval,
                    "share of evaluated station-years at monthly stations": okSY[hi][:, Y0 - 1990:Y1 - 1989].sum() / n_sy_eval if n_hi else np.nan})
EFFORT = pd.DataFrame(EFF); save_table(EFFORT, "T32_sampling_effort"); display(EFFORT.round(3))
rows = []
for name in PS:
    for lev, lab in LEVELS.items():
        tot, det = PS[name][lev]
        r_ = {"design": name, "level": lab}
        for g, m in GROUPS_R.items(): r_[g] = det[m].sum() / tot[m].sum() if tot[m].sum() else np.nan
        r_["units (all stations)"] = tot.sum(); rows.append(r_)
LEVT = pd.DataFrame(rows); save_table(LEVT, "T33_detection_levels"); display(LEVT.round(3))
# paired station bootstrap of design - uniform differences (all stations)
rb = []; idx = np.random.RandomState(SEED).randint(0, S, (CFG["n_boot"], S)); base = PS[DES6[0][0]]
for name in list(PS)[1:]:
    for lev, lab in LEVELS.items():
        (t0_, d0_), (t1_, d1_) = base[lev], PS[name][lev]
        est = d1_.sum() / t1_.sum() - d0_.sum() / t0_.sum()
        bs = d1_[idx].sum(1) / t1_[idx].sum(1) - d0_[idx].sum(1) / t0_[idx].sum(1)
        rb.append({"comparison": f"{name}  minus  {DES6[0][0]}", "level": lab, "difference": est,
                   "95% CI low": np.percentile(bs, 2.5), "95% CI high": np.percentile(bs, 97.5)})
LEVB = pd.DataFrame(rb); save_table(LEVB, "T33b_detection_levels_bootstrap"); display(LEVB.round(3))
# overlap of the top-56 sets between all ranking periods (Jaccard), exported for Table S5
keys = list(RANKS); jac = [{"ranking A": rank_label(a), "ranking B": rank_label(b),
                            f"Jaccard of top {N_HI[2]}": len(top(a) & top(b)) / len(top(a) | top(b)), "shared stations": len(top(a) & top(b))}
                           for ai, a in enumerate(keys) for b in keys[ai + 1:]]
JAC = pd.DataFrame(jac); save_table(JAC, "T19b_ranking_overlap"); display(JAC.round(3))
# figure: detection of recorded exceedances, station-years and stations under each design (observation only)
fig, ax = plt.subplots(figsize=(10, 3.8)); xx = np.arange(len(LEVELS)); w = .16
cols_ = (["#2a78d6", "#f08a8a", "#e34948", "#7a1f1f", "#eda100"] if len(PS) == 5 else ["#2a78d6", "#e34948", "#eda100"])
for k, name in enumerate(PS):
    v = [LEVT[(LEVT.design == name) & (LEVT.level == lab)]["all stations"].iloc[0] for lab in LEVELS.values()]
    b = ax.bar(xx + (k - (len(PS) - 1) / 2) * w, v, w * .92, color=cols_[k], label=name); ax.bar_label(b, fmt="%.2f", fontsize=6.5, padding=1)
ax.set_xticks(xx); ax.set_xticklabels([f"Share of {v} detected" for v in LEVELS.values()], fontsize=8); ax.set_ylim(0, 1)
ax.set_ylabel("Observation-only detection, 2008-2024"); ax.legend(fontsize=7, ncol=2, loc="upper left")
ax.set_title(f"Detection at three levels, > {PRIMARY} mg N L$^{{-1}}$ (budget ≤ 4 visits per station-year)")
fig.tight_layout(); savefig(fig, "F25_detection_levels"); plt.show()

# ==== CELL P1 ==== Publication figures (manuscript versions). All values are read from the tables written above.
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
UNIT = "mg N L$^{-1}$"
_RC = dict(plt.rcParams)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})
def rd(name): return pd.read_csv(os.path.join(OUT, name + ".csv"))
# ---- Fig. 1 - study workflow
fig, ax = plt.subplots(figsize=(7.5, 5.6)); ax.set_xlim(0, 10.5); ax.set_ylim(0, 7.7); ax.axis("off"); ax.grid(False)
GREY, BLUE, PEACH, PINK = "#ededeb", "#dce8f7", "#fbe3d8", "#f6dadd"
BX = [("1 Open data", "174,667 samples\n1,184 sites\n1990–2024", GREY, 0, 0), ("2 Validation", "structural checks\ncensoring (LOD)\nriver vs lake sites", GREY, 1, 0),
      ("3 Station cohort", f"{S} river stations\nselected on\n1995–2007 only", GREY, 2, 0), ("4 Reduction designs", "8, 6, 4 visits/yr\none-at-a-time,\nnetwork, risk-based", BLUE, 2, 1),
      ("5 Features", "own history ≤ t$_0$\nneighbours in t\nleakage test", BLUE, 1, 1), ("6 Strategies", "observation, persist.\nratio rule\nHistGB own / + nb", PEACH, 0, 1),
      ("7 Validation", "cut-offs: 2005–07\n3-yr retraining\nevaluation 2008–24", PEACH, 0, 2), ("8 Evaluation", "AP, recall at cut-off\nunseen stations\nschedules, allocation", PINK, 1, 2),
      ("9 Uncertainty", "station, year,\ntwo-way, basin\nbootstrap", PINK, 2, 2)]
W_, H_, GX, GY = 3.0, 2.15, 3.55, 2.6
def pos(c, r): return 0.1 + c * GX, 7.7 - 0.1 - H_ - r * GY
for t_, b_, fc, c, r in BX:
    x0, y0 = pos(c, r)
    ax.add_patch(FancyBboxPatch((x0, y0), W_, H_, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#52514e", lw=1))
    ax.text(x0 + W_ / 2, y0 + H_ - 0.38, t_, ha="center", va="center", fontsize=11.5, weight="bold")
    ax.text(x0 + W_ / 2, y0 + H_ / 2 - 0.25, b_, ha="center", va="center", fontsize=10, linespacing=1.35)
def arr(c1, r1, c2, r2):
    (x1, y1), (x2, y2) = pos(c1, r1), pos(c2, r2)
    if r1 == r2:
        xa, xb = (x1 + W_, x2) if c2 > c1 else (x1, x2 + W_); ya = y1 + H_ / 2; ax.annotate("", (xb, ya), (xa, ya), arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#52514e"))
    else: ax.annotate("", (x2 + W_ / 2, y2 + H_), (x1 + W_ / 2, y1), arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#52514e"))
for a_, b_ in [((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (2, 1)), ((2, 1), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (1, 2)), ((1, 2), (2, 2))]: arr(*a_, *b_)
savefig(fig, "Fig01_workflow"); plt.show()

# ---- Fig. 2 - study area (requires the map objects of Section 14.5)
fig, ax = plt.subplots(2, 1, figsize=(7.6, 9.0), sharex=True)
for a_ in ax:
    a_.add_collection(LineCollection(segs, linewidths=0.25 + 0.45 * (ordr - 1), colors="#9cc3e6", zorder=1))
    a_.set_aspect("equal"); a_.set_xlim(XY[:, 0].min() / 1000 - 8, XY[:, 0].max() / 1000 + 8); a_.set_ylim(XY[:, 1].min() / 1000 - 8, XY[:, 1].max() / 1000 + 8)
    a_.set_ylabel("Northing (km)"); a_.grid(alpha=.15)
ax[1].set_xlabel("Easting (km, Irish Grid)")
sc = ax[0].scatter(XY[:, 0] / 1000, XY[:, 1] / 1000, c=rate_, cmap="Reds", vmin=0, vmax=10, s=26, edgecolor="k", lw=.35, zorder=3)
ax[0].scatter(XY[~reliable, 0] / 1000, XY[~reliable, 1] / 1000, s=80, facecolor="none", edgecolor="#2a78d6", lw=1.2, zorder=4)
cb = fig.colorbar(sc, ax=ax[0], shrink=.55, anchor=(0, 1)); cb.ax.set_title(f"Months > 0.3\n{UNIT},\n2008–24 (%)", fontsize=10, loc="left")
ax[0].legend(handles=[Line2D([], [], color="#9cc3e6", lw=2, label="River network\n(width: order)"),
                      Line2D([], [], marker="o", ls="", mfc="none", mec="#2a78d6", ms=8, label=f"> {SNAP_MAX_M} m from\na mapped reach")],
             fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 0.45))
for i, j in pairs: ax[1].plot(XY[[i, j], 0] / 1000, XY[[i, j], 1] / 1000, color="#e34948", lw=1.2, zorder=2)
ax[1].scatter(XY[~HI0, 0] / 1000, XY[~HI0, 1] / 1000, s=14, color="#52514e", zorder=3)
ax[1].scatter(XY[HI0, 0] / 1000, XY[HI0, 1] / 1000, s=70, marker="*", color="#eda100", edgecolor="k", lw=.4, zorder=4)
ax[1].legend(handles=[Line2D([], [], color="#e34948", lw=1.5, label=f"Flow-connected\nneighbour pair\n(n = {len(pairs)})"),
                      Line2D([], [], marker="*", ls="", color="#eda100", mec="k", ms=11, label=f"{NHI} high-risk\nstations kept\nmonthly"),
                      Line2D([], [], marker="o", ls="", color="#52514e", ms=5, label="Other stations\n(2 visits/yr)")], fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1.0))
for k_, a_ in enumerate(ax): a_.set_title(f"({'ab'[k_]})", loc="left", x=-0.12)
fig.tight_layout(); savefig(fig, "Fig02_study_area"); plt.show()

# ---- Fig. 3 - precision-recall curves for skipped months (same curves as F09)
nf = len(CFG["sched"]); fig, ax = plt.subplots(nf, 1, figsize=(6.8, 3.05 * nf), sharex=True, sharey=True, squeeze=False); ax = ax[:, 0]
for k, (a_, f) in enumerate(zip(ax, CFG["sched"])):
    sel_f3 = (MAIN["visits/yr"] == f) & (MAIN.threshold == PRIMARY)
    for s_ in STRATS:
        p_, r_, _ = curves[(f, s_)]; ap = MAIN[sel_f3 & (MAIN.strategy == s_)]["PR-AUC"].iloc[0]
        a_.plot(r_, p_, color=COL[s_], lw=1.8, label=f"{LAB[s_]} (AP {ap:.3f})")
    prev = MAIN[sel_f3 & MAIN.strategy.str.startswith("prev")]["PR-AUC"].iloc[0]
    a_.axhline(prev, color="k", ls=":", lw=0.9, label=f"Prevalence ({prev:.3f})")
    a_.set_title(f"({'abcdefgh'[k]}) {f} visits per year, > {PRIMARY} {UNIT}", loc="left"); a_.set_ylabel("Precision")
    a_.set_xlim(0, 1); a_.set_ylim(0, 1); a_.legend(frameon=False, loc="upper right"); a_.tick_params(labelbottom=True)
ax[-1].set_xlabel("Recall (skipped months)"); fig.tight_layout(); savefig(fig, "Fig03_pr_curves"); plt.show()

# ---- Fig. 4 - neighbour distance (4 visits/yr, > 0.3)
RADT_, DIST_ = rd("T11_radius_restriction"), rd("T10_distance_strata")
r = RADT_[RADT_.threshold == PRIMARY].sort_values("radius (km)"); x = np.arange(len(r))
dd = DIST_[(DIST_["visits/yr"] == 4) & (DIST_.threshold == PRIMARY)].reset_index(drop=True); xd = np.arange(len(dd))
fig, ax = plt.subplots(3, 1, figsize=(6.4, 8.4))
a2 = ax[0].twinx(); a2.grid(False); b = a2.bar(x, r["skipped months with a neighbour"] * 100, .55, color="#dce9f8", zorder=0)
a2.bar_label(b, labels=[f"{v:.0%}" for v in r["skipped months with a neighbour"]], fontsize=10, color="#2a78d6", label_type="edge", padding=-12)
a2.set_ylim(0, 260); a2.set_yticks([0, 25, 50, 75, 100]); a2.set_ylabel("Months with a sampled\nneighbour (%, bars)", color="#2a78d6", fontsize=10)
ax[0].set_zorder(a2.get_zorder() + 1); ax[0].patch.set_visible(False)
ax[0].plot(x, r["PR-AUC HistGB + nb"], "o-", color=COL["HistGB_nb"], lw=1.6, label="HistGB + neighbours (retrained)")
for xi, v in zip(x, r["PR-AUC HistGB + nb"]): ax[0].annotate(f"{v:.3f}", (xi, v), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10, color=COL["HistGB_nb"])
ax[0].axhline(r["PR-AUC own history"].iloc[0], color=COL["HistGB_own"], ls="--", lw=1.1, label="own history only")
ax[0].axhline(r["PR-AUC persistence"].iloc[0], color="k", ls=":", lw=1.1, label="persistence")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{v:g} km" for v in r["radius (km)"]]); ax[0].set_xlabel("Maximum neighbour search radius", fontsize=10)
ax[0].set_ylabel("AP (skipped months)"); ax[0].set_ylim(0, .25); ax[0].legend(fontsize=10, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.28))
w = .2
for i, s_ in enumerate(STRATS): ax[1].bar(xd + (i - 1.5) * w, dd[f"PR-AUC {s_}"], w * .92, color=COL[s_], label={"HistGB_own": "HistGB own", "HistGB_nb": "HistGB + nb", "Neighbour_ratio": "Ratio rule"}.get(s_, LAB[s_]))
lab_d = [c.replace("no neighbour sampled", "No neighbour\nsampled") for c in dd.distance]
ax[1].set_xticks(xd); ax[1].set_xticklabels(lab_d); ax[1].set_xlabel("Distance to nearest sampled neighbour", fontsize=10); ax[1].set_ylabel("AP")
ax[1].legend(fontsize=10, ncol=2, loc="upper left")
e = dd["dPR-AUC (nb - persistence)"]; lo, hi = dd["95% CI low"], dd["95% CI high"]
ax[2].errorbar(xd, e, yerr=[e - lo, hi - e], fmt="none", ecolor="#52514e", capsize=3, lw=1)
ax[2].scatter(xd, e, s=40, zorder=3, color=[COL["HistGB_nb"] if l_ > 0 else "white" for l_ in lo], edgecolor=COL["HistGB_nb"], lw=1.3)
ax[2].axhline(0, color="k", lw=.7); ax[2].set_xticks(xd); ax[2].set_xticklabels(lab_d)
ax[2].set_xlabel("Distance to nearest sampled neighbour", fontsize=10); ax[2].set_ylabel("AP gain over\npersistence")
for k_, a_ in enumerate(ax): a_.set_title(f"({'abc'[k_]})", loc="left", x=-0.1, fontsize=11)
fig.tight_layout(); savefig(fig, "Fig04_neighbour_distance"); plt.show()

# ---- Fig. 5 - network-wide schedules (AP and recall, 6 and 4 visits/yr)
NT = rd("T15_network_schedules"); NT = NT[NT.threshold == PRIMARY]
SCH = [("network: synchronised", "Sync."), ("network: random phases", "Rand."), ("network: staggered (graph colouring)", "Stag."), ("one-at-a-time (neighbours monthly)", "One at\na time")]
fig, ax = plt.subplots(2, 1, figsize=(8.2, 8.6)); w = .21
for row, (metric, keys_) in enumerate((("PR-AUC", ["Persistence", "Neighbour_ratio", "HistGB_nb"]), ("recall", ["Observation-only", "Persistence", "Neighbour_ratio", "HistGB_nb"]))):
    a_ = ax[row]; xt, xl = [], []
    for g, f in enumerate([f_ for f_ in (6, 4) if f_ in set(NT["visits/yr"])]):
        for j, (sc_, sl) in enumerate(SCH):
            rr = NT[(NT["visits/yr"] == f) & (NT.scheme == sc_)].iloc[0]; x0 = g * 4.6 + j
            for k, s_ in enumerate(keys_):
                col = f"{s_} {metric}"; xb = x0 + (k - (len(keys_) - 1) / 2) * w
                bb = a_.bar(xb, rr[col], w * .92, color=COL[s_])
                if s_ == "HistGB_nb": a_.annotate(f"{rr[col]:.3f}" if metric == "PR-AUC" else f"{rr[col]:.2f}", (xb, rr[col]), xytext=(0, 3), textcoords="offset points", ha="center", rotation=90, fontsize=10)
            xt.append(x0); xl.append(f"{sl}\n{rr['skipped months with a neighbour sampled in the same month']:.0%}")
        a_.text(g * 4.6 + 1.5, (0.265 if metric == "PR-AUC" else 0.8), f"{f} visits/yr", ha="center", fontsize=10)
    a_.axvline(3.8, color="grey", ls="--", lw=.8); a_.set_xticks(xt); a_.set_xticklabels(xl)
    a_.set_ylabel("AP (skipped months)" if metric == "PR-AUC" else "Recall at FA-budget cut-off"); a_.set_ylim(0, .27 if metric == "PR-AUC" else .82)
    a_.set_title(f"({'ab'[row]})", loc="left", x=-0.08)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[s_]) for s_ in ["Observation-only", "Persistence", "Neighbour_ratio", "HistGB_nb"]],
           labels=["Observation only", "Persistence", "Neighbour-ratio rule", "HistGB + neighbours"], ncol=2, loc="lower center", fontsize=11)
fig.tight_layout(rect=(0, .07, 1, 1)); savefig(fig, "Fig05_network_schedules"); plt.show()

# ---- Fig. 6 - uniform versus risk-based allocation (recall, 0.2 and 0.3)
FA_ = rd("T20_fair_allocation_comparison"); DLAB = ["Observation only", "Persistence", "Neighbour-ratio rule", "HistGB + neighbours"]
DCOL = dict(zip(DLAB, [COL["Observation-only"], COL["Persistence"], COL["Neighbour_ratio"], COL["HistGB_nb"]]))
dsg = list(dict.fromkeys(FA_.design)); dshort = ["Uniform\nsynchr.", "Uniform\nstagg."] + [d.split(": ")[1].replace(", others", "\nothers") for d in dsg[2:]]
fig, ax = plt.subplots(2, 1, figsize=(7.6, 7.4), sharex=True); w = .2
for row, thr in enumerate((0.2, PRIMARY)):
    for j, d_ in enumerate(dsg):
        for k, dl in enumerate(DLAB):
            v = FA_[(FA_.design == d_) & (FA_.threshold == thr) & (FA_.detection == dl)]["recall"].iloc[0]; xb = j + (k - 1.5) * w
            ax[row].bar(xb, v, w * .92, color=DCOL[dl]); ax[row].annotate(f"{v:.2f}", (xb, v), xytext=(0, 2), textcoords="offset points", ha="center", rotation=90, fontsize=10)
    ax[row].axvline(1.5, color="grey", ls="--", lw=.8); ax[row].set_ylim(0, 1); ax[row].set_ylabel("Recall of recorded\nexceedances, 2008–2024")
    ax[row].set_title(f"({'ab'[row]})", loc="left", x=-0.1); ax[row].text(1.6, .93, f"> {thr} {UNIT}", fontsize=10)
ax[1].set_xticks(range(len(dsg))); ax[1].set_xticklabels(dshort)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=DCOL[d]) for d in DLAB], labels=DLAB, ncol=2, loc="lower center", fontsize=10)
fig.tight_layout(rect=(0, .07, 1, 1)); savefig(fig, "Fig06_allocation"); plt.show()

# ---- Fig. 7 - recall by station group (observation only; HistGB + neighbours)
SG = rd("T24b_station_groups_allocation"); SG = SG[SG.threshold == PRIMARY]
gcols = [c for c in SG.columns if c.startswith("recall, ")]
glab = ["All\nstations", f"{N_HI[2]} high-risk\n(pre-2008)", f"{S - N_HI[2]} other\nstations", f"{EMERG_R.sum()} emerging\nstations"]
DC7 = dict(zip(dsg, ["#a8a69e", "#2a78d6", "#f08a8a", "#e34948", "#7a1f1f"] if len(dsg) == 5 else ["#a8a69e", "#2a78d6", "#e34948"]))
fig, ax = plt.subplots(2, 1, figsize=(7.6, 7.6), sharex=True); w = .17
for row, det in enumerate(("Observation only", "HistGB + neighbours")):
    for k, d_ in enumerate(dsg):
        v = SG[(SG.design == d_) & (SG.detection == det)][gcols].iloc[0].values; xb = np.arange(len(gcols)) + (k - (len(dsg) - 1) / 2) * w
        ax[row].bar(xb, v, w * .92, color=DC7[d_], label=d_)
        for xi, vi in zip(xb, v): ax[row].annotate(f"{vi:.2f}", (xi, vi), xytext=(0, 2), textcoords="offset points", ha="center", rotation=90, fontsize=10)
    ax[row].set_ylim(0, 1.25); ax[row].set_yticks([0, .25, .5, .75, 1]); ax[row].set_ylabel("Recall, 2008–2024")
    ax[row].set_title(f"({'ab'[row]}) {det} (> {PRIMARY} {UNIT})", loc="left", fontsize=11)
ax[1].set_xticks(range(len(gcols))); ax[1].set_xticklabels(glab)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=DC7[d]) for d in dsg], labels=dsg, ncol=2, loc="lower center", fontsize=10)
fig.tight_layout(rect=(0, .1, 1, 1)); savefig(fig, "Fig07_station_groups"); plt.show()

# ---- Fig. 8 - alarm-triggered follow-up
FU = rd("T28_alarm_triggered_followup"); FU = FU[FU.threshold == PRIMARY]
fig, ax = plt.subplots(2, 1, figsize=(6.6, 7.0), sharex=True, sharey=True)
for row, d_ in enumerate(dict.fromkeys(FU.design)):
    q = FU[FU.design == d_].reset_index(drop=True); x = np.arange(len(q))
    b1 = ax[row].bar(x - .2, q["new detections per 100 scorable extra visits"], .38, color="#e34948", label="Triggered extra visits")
    b2 = ax[row].bar(x + .2, q["random control: per 100 visits (mean)"], .38, color="#a8a69e", label="Random extra visits (same number; 2.5–97.5% range)")
    ax[row].errorbar(x + .2, q["random control: per 100 visits (mean)"], yerr=[q["random control: per 100 visits (mean)"] - q["random control: 2.5%"],
                     q["random control: 97.5%"] - q["random control: per 100 visits (mean)"]], fmt="none", ecolor="k", capsize=3, lw=1)
    ax[row].bar_label(b1, fmt="%.1f", fontsize=10, padding=1); ax[row].bar_label(b2, fmt="%.1f", fontsize=10, padding=14)
    ax[row].set_ylabel("New exceedances per\n100 extra visits"); ax[row].set_title(f"({'ab'[row]}) {d_}", loc="left", fontsize=11)
ax[1].set_xticks(x); ax[1].set_xticklabels(["Model alarm", "Observed exceedance", "Either trigger"])
fig.legend(handles=[b1, b2], loc="lower center", fontsize=10); fig.tight_layout(rect=(0, .08, 1, 1)); savefig(fig, "Fig08_followup"); plt.show()

# ---- Fig. S1 - spatial clusters and per-cluster AP (seen / unseen)
fig, ax = plt.subplots(2, 1, figsize=(6.6, 8.2), gridspec_kw={"height_ratios": [1.3, 1]})
ax[0].scatter(xy[:, 0] / 1000, xy[:, 1] / 1000, c=groups, cmap="tab10", s=18, edgecolor="k", lw=.3); ax[0].set_aspect("equal")
for g in range(CFG["n_groups"]):
    c = xy[groups == g].mean(0) / 1000; ax[0].text(c[0], c[1], f"C{g}", fontsize=11, weight="bold")
ax[0].set_xlabel("Easting (km, Irish Grid)"); ax[0].set_ylabel("Northing (km)"); ax[0].set_title("(a)", loc="left")
xg = np.arange(len(PG)); w = .27
for i, (c, lab, cc) in enumerate((("Persistence", "Persistence", "#52514e"), ("seen", "HistGB + nb (seen)", "#e34948"), ("unseen", "HistGB + nb (unseen)", "#f4a3a3"))):
    ax[1].bar(xg + (i - 1) * w, PG[c], w * .92, color=cc, label=lab)
ax[1].set_xticks(xg); ax[1].set_xticklabels([f"C{g}\n{n}" for g, n in zip(PG.cluster, PG.events)], fontsize=10)
ax[1].set_xlabel("Cluster (second line: recorded exceedances)"); ax[1].set_ylabel("AP (skipped months)"); ax[1].legend(fontsize=10); ax[1].set_title("(b)", loc="left")
fig.tight_layout(); savefig(fig, "FigS1_spatial_generalisation"); plt.show()

# ---- Fig. S2 - detection versus coverage of emerging pollution (observation only)
ob = SG[SG.detection == "Observation only"]; pts = []
for d_ in dsg[1:]:
    q = ob[ob.design == d_].iloc[0]; lab_ = "Uniform 4/yr" if d_.startswith("Uniform") else d_.split(": ")[1].replace(", others ", ",\nothers ")
    pts.append((q["'emerging' stations detected at least once"] * 100, q["recall, all stations"] * 100, lab_, "#2a78d6"))
rr_ = COST.loc["risk-based, re-ranked every 3 yr"]
pts.append((rr_.iloc[3] * 100, rr_.iloc[0] * 100, f"{NHI} monthly, 2/yr,\nre-ranked", "#e34948"))
fig, ax = plt.subplots(figsize=(6.2, 4.3))
for x_, y_, l_, c_ in pts:
    off = (-62, 8) if "Uniform" in l_ else (8, 4) if "re-ranked" in l_ else (-92, -26) if l_.startswith(f"{N_HI[2]} monthly") else (8, -4)
    ax.scatter(x_, y_, s=70, color=c_, edgecolor="k", lw=.6, zorder=3); ax.annotate(l_, (x_, y_), xytext=off, textcoords="offset points", fontsize=11)
ax.set_xlim(5, 50); ax.set_xlabel(f"Emerging stations detected\nat least once (%, n = {EMERG_R.sum()})"); ax.set_ylabel("Network-level recall (%),\nobservation only")
fig.tight_layout(); savefig(fig, "FigS2_detection_vs_emerging"); plt.show()
plt.rcParams.update(_RC)
print("publication figures:", ", ".join(sorted(f for f in os.listdir(FIG) if f.startswith("Fig") and f.endswith(".png"))))
