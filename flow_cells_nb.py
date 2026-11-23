# ==== CELL A ==== 12.2 Flow connectivity from HydroRIVERS (standalone: reads only exported files)
# Requires: pip install pyproj shapely ; file data/hydrorivers_ni.json (HydroRIVERS v1.0 Europe, clipped to -8.3,53.9,-5.3,55.4)
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score as AP
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
ROOT = os.getcwd() if os.path.exists("data/raw") else os.path.dirname(os.getcwd())
OUT = os.path.join(ROOT, "results", "complete_study"); DAT = os.path.join(OUT, "data"); FIG = os.path.join(OUT, "figures")
SNAP_MAX_M = 300                                     # stations further than this from a HydroRIVERS reach are 'unreliable'
# --- monthly reference record of the cohort (exported in Section 5.2)
_mref = pd.read_csv(os.path.join(DAT, "monthly_reference_record.csv.gz")); _meta = pd.read_csv(os.path.join(DAT, "station_metadata.csv"), index_col=0)
codes = sorted(_mref.StationCode.unique()); si = {c: i for i, c in enumerate(codes)}; NS = len(codes)
months = pd.period_range("1990-01", "2024-12", freq="M"); mi = {str(p): i for i, p in enumerate(months)}; TT = len(months)
VV = np.full((NS, TT), np.nan); CC = np.zeros((NS, TT), bool); EVM = np.zeros((NS, TT), bool)
for r in _mref.itertuples(): i, t = si[r.StationCode], mi[r.month]; VV[i, t] = r.NH4_N; CC[i, t] = r.censored; EVM[i, t] = r.evaluation_month
LL = np.log(VV); MON = np.array([p.month for p in months]); YR = np.array([p.year for p in months])
XY = _meta.loc[codes, ["E", "N"]].values.astype(float)
# --- river network: reaches drawn from upstream to downstream; NEXT_DOWN = downstream reach id
import glob
_cands = [p for p in [os.path.join(ROOT, "data", "hydrorivers_ni.json")] + glob.glob(os.path.join(ROOT, "**", "*ydro*iver*.json"), recursive=True) if os.path.exists(p)]
assert _cands, "HydroRIVERS file not found: place hydrorivers_ni.json in data/"
hr = json.load(open(_cands[0], encoding="utf-8"))
to_ig = Transformer.from_crs("EPSG:4326", "EPSG:29902", always_xy=True)
rid, lines, PR = [], [], {}
for f in hr["features"]:
    c = np.array(f["geometry"]["coordinates"]); x, y = to_ig.transform(c[:, 0], c[:, 1])
    rid.append(f["properties"]["HYRIV_ID"]); lines.append(LineString(np.c_[x, y])); PR[f["properties"]["HYRIV_ID"]] = f["properties"]
tree = STRtree(lines); snap = []
for i in range(NS):
    p = Point(XY[i]); k = tree.nearest(p); snap.append((rid[k], lines[k].distance(p), lines[k].project(p) / 1000, lines[k].length / 1000))
SNAP = pd.DataFrame(snap, columns=["reach", "snap_m", "pos_km", "reach_km"], index=codes)
def dpath(r0):
    out, seen = [r0], {r0}
    while (nd := PR.get(out[-1], {}).get("NEXT_DOWN", 0)) and nd in PR and nd not in seen: out.append(nd); seen.add(nd)
    return out
PATH = [dpath(r) for r in SNAP.reach]
def flow_rel(a, b):
    # relation of neighbour b to target a and flow distance (km)
    if SNAP.snap_m.iloc[a] > SNAP_MAX_M or SNAP.snap_m.iloc[b] > SNAP_MAX_M: return "unreliable snap", np.nan
    ra, rb = SNAP.reach.iloc[a], SNAP.reach.iloc[b]
    if ra == rb: dk = SNAP.pos_km.iloc[a] - SNAP.pos_km.iloc[b]; return ("upstream" if dk > 0 else "downstream"), abs(dk)
    pa, pb = PATH[a], PATH[b]
    if ra in pb:
        k = pb.index(ra); return "upstream", SNAP.reach_km.iloc[b] - SNAP.pos_km.iloc[b] + sum(PR[x]["LENGTH_KM"] for x in pb[1:k]) + SNAP.pos_km.iloc[a]
    if rb in pa:
        k = pa.index(rb); return "downstream", SNAP.reach_km.iloc[a] - SNAP.pos_km.iloc[a] + sum(PR[x]["LENGTH_KM"] for x in pa[1:k]) + SNAP.pos_km.iloc[b]
    return ("same system, other branch" if pa[-1] == pb[-1] else "other river system"), np.nan
print(f"stations snapped within {SNAP_MAX_M} m of a HydroRIVERS reach: {(SNAP.snap_m <= SNAP_MAX_M).sum()}/{NS} (median snap distance {SNAP.snap_m.median():.0f} m)")
# --- skipped months of the 4 visits/yr one-at-a-time design, nearest sampled neighbour (<= 3 within 15 km) classified by flow relation
dK, nK = cKDTree(XY).query(XY, k=min(12, NS)); REL = {}; rows = []
for keep in ([1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]):
    av = ~np.isnan(VV) & (np.isin(MON, keep)[None, :] | (YR < 2008)[None, :])
    for i in range(NS):
        idx = np.flatnonzero(av[i])
        for t in np.flatnonzero(EVM[i] & ~av[i]):
            k = np.searchsorted(idx, t) - 1
            if k < 0: continue
            t0 = idx[k]; cand = []
            for j, dd in zip(nK[i, 1:], dK[i, 1:] / 1000):
                if np.isnan(LL[j, t]) or np.isnan(LL[j, t0]): continue
                if (i, j) not in REL: REL[(i, j)] = flow_rel(i, j)
                cand.append((dd, j) + REL[(i, j)])
            near = [c for c in cand if c[0] <= 15][:3]; con = [c for c in cand if c[2] in ("upstream", "downstream") and c[3] <= 25]
            row = {"s": i, "y": bool((VV[i, t] > 0.3) & ~CC[i, t]), "last": LL[i, t0], "target_change": LL[i, t] - LL[i, t0]}
            if near: dd, j, rel_, fd = near[0]; row.update(cls=rel_, km=dd, nb_change=LL[j, t] - LL[j, t0], rr_near=LL[i, t0] + LL[j, t] - LL[j, t0])
            else: row.update(cls="no neighbour sampled", km=np.nan, nb_change=np.nan, rr_near=LL[i, t0])
            c_ = sorted(con, key=lambda c: (c[2] != "upstream", c[3]))[0] if con else None
            row["rr_flow"] = LL[i, t0] + LL[c_[1], t] - LL[c_[1], t0] if c_ else row["rr_near"]
            rows.append(row)
FL = pd.DataFrame(rows); nph_ = 3; rows = []
for g in ["upstream", "downstream", "same system, other branch", "other river system", "unreliable snap", "no neighbour sampled"]:
    x = FL[FL.cls == g]
    if len(x) == 0: continue
    rows.append({"nearest sampled neighbour": g, "share of skipped months": len(x) / len(FL), "events (per phase)": int(x.y.sum() / nph_), "median km": x.km.median(),
                 "r (neighbour vs target change)": np.corrcoef(x.nb_change.fillna(0), x.target_change)[0, 1] if g != "no neighbour sampled" else np.nan,
                 "PR-AUC persistence": AP(x.y, x["last"]), "PR-AUC neighbour-ratio rule": AP(x.y, x.rr_near)})
FLOW = pd.DataFrame(rows); FLOW.to_csv(os.path.join(OUT, "T12d_flow_connectivity.csv"), index=False); display(FLOW.round(3))
rng = np.random.RandomState(42); us = np.unique(FL.s); mem = {s: np.flatnonzero(FL.s.values == s) for s in us}; bs = []
for _ in range(1000):
    ix = np.concatenate([mem[s] for s in rng.choice(us, len(us))]); yy = FL.y.values[ix]
    if yy.sum(): bs.append(AP(yy, FL.rr_flow.values[ix]) - AP(yy, FL.rr_near.values[ix]))
print(f"neighbour-ratio rule, PR-AUC: nearest neighbour {AP(FL.y, FL.rr_near):.4f} | flow-connected neighbour first (upstream > downstream, <= 25 km along the river) "
      f"{AP(FL.y, FL.rr_flow):.4f} | difference {AP(FL.y, FL.rr_flow) - AP(FL.y, FL.rr_near):+.4f} (95% CI {np.percentile(bs, 2.5):+.4f} to {np.percentile(bs, 97.5):+.4f})")
print(f"share of the {int(FL.y.sum() / nph_)} exceedances (per phase) at stations without a reliable reach: {FL[FL.cls == 'unreliable snap'].y.sum() / FL.y.sum():.0%} (small streams are missing from HydroRIVERS)")
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8)); F_ = FLOW[FLOW["nearest sampled neighbour"] != "no neighbour sampled"]
cols_ = ["#e34948" if g in ("upstream", "downstream") else "#a8a69e" for g in F_["nearest sampled neighbour"]]
ax[0].barh(F_["nearest sampled neighbour"], F_["r (neighbour vs target change)"], color=cols_); ax[0].invert_yaxis(); ax[0].set_xlabel("Correlation of month-to-month change")
ax[0].set_title("(a) Co-variation with the nearest sampled neighbour")
ax[1].barh(F_["nearest sampled neighbour"], F_["PR-AUC neighbour-ratio rule"], color=cols_); ax[1].invert_yaxis(); ax[1].set_xlabel("PR-AUC, neighbour-ratio rule (4 visits/yr, > 0.3 mg/l)")
ax[1].set_title("(b) Value of the neighbour by flow relation"); fig.tight_layout()
fig.savefig(os.path.join(FIG, "F12d_flow_connectivity.png"), dpi=300, bbox_inches="tight"); plt.show()

# ==== CELL B ==== 14.4 Cost of risk-based allocation for low-risk and 'emerging' stations, and adaptive re-ranking (standalone)
# Observation-only detection (no model, no false alarms): exact and comparable for every design. Budget <= 4 visits per station-year.
from scipy.special import betaln
from scipy.optimize import minimize
THR_ = 0.3; NHI = int(np.floor(2 * NS / 10)); REST_ = 2; KPH = 12 // REST_          # 56 stations monthly, others 2 visits/yr (6 phases)
EXCM = (VV > THR_) & ~CC & ~np.isnan(VV)                                                # recorded exceedance
def eb_score(k, n):
    msk = n > 0
    def nll(p): a, b = np.exp(p); return -np.sum(betaln(k[msk] + a, n[msk] - k[msk] + b) - betaln(a, b))
    r0 = max(k[msk].sum() / n[msk].sum(), 1e-3); a, b = np.exp(minimize(nll, np.log([r0 * 20, (1 - r0) * 20]), method="Nelder-Mead").x)
    return (k + a) / (n + a + b)
def static_rank():
    h = (YR >= 1995) & (YR <= 2007); obs = ~np.isnan(VV) & h[None, :]
    sc = eb_score((EXCM & obs).sum(1).astype(float), obs.sum(1).astype(float)); return np.argsort(-sc, kind="stable")
PH = [[q + 1 + j * KPH for j in range(REST_)] for q in range(KPH)]
def simulate(design, rot):
    # returns the visited-month mask (S x T) for 2008-2024 under the design; before 2008 everything observed
    vis = ~np.isnan(VV) & (YR < 2008)[None, :]
    if design == "uniform 4/yr":
        sch = np.isin(MON, [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]][rot % 3])[None, :].repeat(NS, 0)
        return vis | (~np.isnan(VV) & sch & (YR >= 2008)[None, :]), None
    order = static_rank(); hi_hist = []
    for B in range(2008, 2025, 3):
        if design == "risk-based, re-ranked every 3 yr" and B > 2008:
            h = (YR >= 1995) & (YR < B); obs = vis & h[None, :]                         # only what was actually sampled so far
            order = np.argsort(-eb_score((EXCM & obs).sum(1).astype(float), obs.sum(1).astype(float)), kind="stable")
        hi = np.zeros(NS, bool); hi[order[:NHI]] = True; hi_hist.append(hi.copy())
        lo_rank = {i: r for r, i in enumerate([i for i in order if not hi[i]])}
        blk = (YR >= B) & (YR <= min(B + 2, 2024))
        for i in range(NS):
            sch = np.ones(TT, bool) if hi[i] else np.isin(MON, PH[(lo_rank[i] + rot) % KPH])
            vis[i] |= ~np.isnan(VV[i]) & sch & blk
    return vis, hi_hist
# station groups defined from pre-2008 information only
h0 = (YR >= 1995) & (YR <= 2007); pre_exc = (EXCM & h0[None, :]).sum(1); ev_mask = EVM & ~np.isnan(VV)
HI0 = np.zeros(NS, bool); HI0[static_rank()[:NHI]] = True
EMERG = (pre_exc == 0) & (EXCM & ev_mask).any(1)                                        # no exceedance before 2008, at least one after
rows, delay = [], []
for design in ("uniform 4/yr", "risk-based, static ranking (1995-2007)", "risk-based, re-ranked every 3 yr"):
    nrot = 3 if design == "uniform 4/yr" else KPH; acc = []
    for rot in range(nrot):
        vis, _ = simulate(design, rot); det = EXCM & ev_mask & vis; tot = EXCM & ev_mask
        first_true = np.where(tot.any(1), np.argmax(tot, 1), -1); first_det = np.where(det.any(1), np.argmax(det, 1), -1)
        dl = [(first_det[i] - first_true[i]) / 12 if first_det[i] >= 0 else np.nan for i in np.flatnonzero(EMERG)]
        acc.append({"recall, all stations": det.sum() / tot.sum(), f"recall, {NHI} pre-2008 high-risk stations": det[HI0].sum() / tot[HI0].sum(),
                    f"recall, {NS - NHI} other stations": det[~HI0].sum() / tot[~HI0].sum(),
                    f"'emerging' stations (n={int(EMERG.sum())}) detected at least once": np.mean([det[i].any() for i in np.flatnonzero(EMERG)])})
    r_ = pd.DataFrame(acc).mean().to_dict(); r_["design"] = design; rows.append(r_)
COST = pd.DataFrame(rows).set_index("design")
COST.to_csv(os.path.join(OUT, "T22_allocation_tradeoffs.csv")); display(COST.round(3))
print(f"pre-2008 high-risk set: {NHI} stations holding {(EXCM & ev_mask)[HI0].sum() / (EXCM & ev_mask).sum():.0%} of the 2008-2024 exceedances; "
      f"'emerging' stations: {int(EMERG.sum())} with {(EXCM & ev_mask)[EMERG].sum()} exceedances ({(EXCM & ev_mask)[EMERG].sum() / (EXCM & ev_mask).sum():.0%})")
fig, ax = plt.subplots(figsize=(10, 3.8)); cols = list(COST.columns[:4]); xx = np.arange(len(cols)); w = .26
for k, (dname, cc) in enumerate(zip(COST.index, ["#a8a69e", "#2a78d6", "#e34948"])):
    b = ax.bar(xx + (k - 1) * w, COST.loc[dname, cols], w * .92, color=cc, label=dname); ax.bar_label(b, fmt="%.2f", fontsize=7, padding=1)
ax.set_xticks(xx); ax.set_xticklabels([c.replace(", ", ",\n").replace(" (", "\n(") for c in cols], fontsize=8); ax.set_ylim(0, 1.05)
ax.set_ylabel("Share detected (observation only)"); ax.legend(fontsize=7, loc="upper right"); ax.set_title(f"Who gains and who loses under risk-based allocation (> {THR_} mg/l, budget <= 4 visits/station-yr)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "F18_allocation_tradeoffs.png"), dpi=300, bbox_inches="tight"); plt.show()

# ==== CELL C ==== Map: river network, stations, flow-connected neighbour pairs and high-risk stations (run after CELL A and CELL B)
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
rate_ = np.array([(EXCM[i] & ev_mask[i]).sum() / max(ev_mask[i].sum(), 1) for i in range(NS)]) * 100        # % of 2008-2024 months > 0.3 mg/l
pairs = []
for i in range(NS):
    for k in range(1, 4):
        j = nK[i, k]
        if dK[i, k] / 1000 > 15: continue
        r_, fd = flow_rel(i, j)
        if r_ in ("upstream", "downstream") and i < j: pairs.append((i, j))
ordr = np.array([PR[r]["ORD_STRA"] for r in rid])
segs = [np.asarray(L.coords) / 1000 for L in lines]
reliable = SNAP.snap_m.values <= SNAP_MAX_M
fig, ax = plt.subplots(1, 2, figsize=(16, 7.6))  # combined figure; panels are also saved separately below
for a_ in ax:
    a_.add_collection(LineCollection(segs, linewidths=0.25 + 0.45 * (ordr - 1), colors="#9cc3e6", zorder=1))
    a_.set_aspect("equal"); a_.set_xlim(XY[:, 0].min() / 1000 - 8, XY[:, 0].max() / 1000 + 8); a_.set_ylim(XY[:, 1].min() / 1000 - 8, XY[:, 1].max() / 1000 + 8)
    a_.set_xlabel("Easting (km, Irish Grid)"); a_.set_ylabel("Northing (km)"); a_.grid(alpha=.15)
# (a) stations coloured by exceedance rate_
sc = ax[0].scatter(XY[:, 0] / 1000, XY[:, 1] / 1000, c=rate_, cmap="Reds", vmin=0, vmax=10, s=26, edgecolor="k", lw=.35, zorder=3)
plt.colorbar(sc, ax=ax[0], shrink=.75, label="Months > 0.3 mg/l NH4-N, 2008-2024 (%)")
ax[0].scatter(XY[~reliable, 0] / 1000, XY[~reliable, 1] / 1000, s=70, facecolor="none", edgecolor="#2a78d6", lw=1, zorder=4)
ax[0].legend(handles=[Line2D([], [], color="#9cc3e6", lw=2, label="HydroRIVERS network (width = Strahler order)"),
                      Line2D([], [], marker="o", ls="", mfc="none", mec="#2a78d6", label=f"station > {SNAP_MAX_M} m from a reach (small stream)")], fontsize=7, loc="lower left")
ax[0].set_title(f"(a) {NS} river stations and the river network")
# (b) flow-connected neighbour pairs and pre-2008 high-risk stations
for i, j in pairs: ax[1].plot(XY[[i, j], 0] / 1000, XY[[i, j], 1] / 1000, color="#e34948", lw=1.2, zorder=2)
ax[1].scatter(XY[~HI0, 0] / 1000, XY[~HI0, 1] / 1000, s=14, color="#52514e", zorder=3)
ax[1].scatter(XY[HI0, 0] / 1000, XY[HI0, 1] / 1000, s=60, marker="*", color="#eda100", edgecolor="k", lw=.4, zorder=4)
ax[1].legend(handles=[Line2D([], [], color="#e34948", lw=1.5, label=f"flow-connected neighbour pair (n = {len(pairs)})"),
                      Line2D([], [], marker="*", ls="", color="#eda100", mec="k", ms=10, label=f"{NHI} high-risk stations kept monthly (ranking 1995-2007)"),
                      Line2D([], [], marker="o", ls="", color="#52514e", ms=4, label="other stations (2 visits/yr)")], fontsize=7, loc="lower left")
ax[1].set_title("(b) Flow-connected neighbours and risk-based allocation")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "F19_map_river_network.png"), dpi=300, bbox_inches="tight")
for k_, nm in ((0, "F19a_study_area_map"), (1, "F19b_flow_pairs_allocation_map")):
    ext = ax[k_].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    if k_ == 0: ext = ext.union([ext, fig.axes[2].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())])
    fig.savefig(os.path.join(FIG, nm + ".png"), dpi=300, bbox_inches=ext.expanded(1.02, 1.02))
plt.show()
print(f"flow-connected neighbour pairs (<= 15 km, 3 nearest): {len(pairs)}; stations in at least one pair: {len(set(np.ravel(pairs)))}")
