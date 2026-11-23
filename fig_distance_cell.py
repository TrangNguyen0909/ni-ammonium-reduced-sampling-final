# Figure F12b - effect of neighbour distance (radius restriction + nearest-neighbour distance strata)
# Uses RADT (T11) and DIST (T10) from Experiment 3; if the notebook was restarted, they are read from the saved CSV files.
try: RADT, DIST
except NameError:
    RADT = pd.read_csv(os.path.join(OUT, "T11_radius_restriction.csv")); DIST = pd.read_csv(os.path.join(OUT, "T10_distance_strata.csv"))
F_, THR_ = 4, PRIMARY                                                   # scenario shown: 4 visits/yr, > 0.3 mg/l N

fig, ax = plt.subplots(1, 3, figsize=(16, 4.3), gridspec_kw={"width_ratios": [1.1, 1.3, 1]})

# (a) radius restriction: coverage (bars) and PR-AUC (line)
r = RADT[RADT.threshold == THR_].sort_values("radius (km)"); x = np.arange(len(r))
a2 = ax[0].twinx(); a2.grid(False)
b = a2.bar(x, r["skipped months with a neighbour"] * 100, 0.55, color="#dce9f8", zorder=0)
a2.bar_label(b, labels=[f"{v:.0%}" for v in r["skipped months with a neighbour"]], fontsize=7, color="#2a78d6", padding=1)
a2.set_ylim(0, 115); a2.set_ylabel("Skipped months with a sampled neighbour (%)", color="#2a78d6"); a2.tick_params(axis="y", colors="#2a78d6")
ax[0].set_zorder(a2.get_zorder() + 1); ax[0].patch.set_visible(False)
ax[0].plot(x, r["PR-AUC HistGB + nb"], "o-", color=COL["HistGB_nb"], lw=2, label="HistGB + neighbours (retrained)")
for xi, v in zip(x, r["PR-AUC HistGB + nb"]): ax[0].annotate(f"{v:.3f}", (xi, v), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7, color=COL["HistGB_nb"])
ax[0].axhline(r["PR-AUC own history"].iloc[0], color=COL["HistGB_own"], ls="--", lw=1.2, label="own history only")
ax[0].axhline(r["PR-AUC persistence"].iloc[0], color=COL["Persistence"], ls=":", lw=1.2, label="persistence")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{v:g} km" for v in r["radius (km)"]]); ax[0].set_xlabel("Maximum neighbour search radius")
ax[0].set_ylabel("PR-AUC (skipped months)"); ax[0].set_ylim(0, r["PR-AUC HistGB + nb"].max() * 1.35)
ax[0].set_title("(a) Radius restriction: coverage drives recovery"); ax[0].legend(fontsize=7, loc="upper left")

# (b) nearest sampled neighbour distance: PR-AUC by strategy
dd = DIST[(DIST["visits/yr"] == F_) & (DIST.threshold == THR_) & (DIST.events >= 10)].reset_index(drop=True); x = np.arange(len(dd)); w = .2   # strata with < 10 events omitted
for i, s_ in enumerate(STRATS):
    bb = ax[1].bar(x + (i - 1.5) * w, dd[f"PR-AUC {s_}"], w * .92, color=COL[s_], label=LAB[s_])
    if s_ == "HistGB_nb": ax[1].bar_label(bb, fmt="%.3f", fontsize=6.5, padding=1)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{c}\n(n={n} events)" for c, n in zip(dd.distance, dd.events)], fontsize=7.5)
ax[1].set_xlabel("Distance to nearest sampled neighbour"); ax[1].set_ylabel("PR-AUC (skipped months)")
ax[1].set_title("(b) PR-AUC by nearest-neighbour distance"); ax[1].legend(fontsize=7, loc="upper right")

# (c) gain over persistence with 95% station-bootstrap CI
e = dd["dPR-AUC (nb - persistence)"]; lo, hi = dd["95% CI low"], dd["95% CI high"]
sig = lo > 0
ax[2].errorbar(x, e, yerr=[e - lo, hi - e], fmt="none", ecolor="#52514e", capsize=4, lw=1.2)
ax[2].scatter(x, e, s=60, zorder=3, color=[COL["HistGB_nb"] if s else "white" for s in sig], edgecolor=COL["HistGB_nb"], lw=1.5)
for xi, v in zip(x, e): ax[2].annotate(f"+{v:.3f}", (xi, v), xytext=(10, 0), textcoords="offset points", va="center", fontsize=7)
ax[2].axhline(0, color="k", lw=.7); ax[2].set_xticks(x); ax[2].set_xticklabels(dd.distance, fontsize=7.5)
ax[2].set_xlabel("Distance to nearest sampled neighbour"); ax[2].set_ylabel("dPR-AUC, HistGB + nb - persistence")
ax[2].set_title("(c) Gain over persistence (open = CI includes 0)")

fig.suptitle(f"Effect of neighbour distance, {F_} visits/yr, exceedance > {THR_} mg/l NH4-N", fontsize=11, weight="bold", y=1.02)
fig.tight_layout(); savefig(fig, "F12b_neighbour_distance"); plt.show()

display(r[["radius (km)", "skipped months with a neighbour", "PR-AUC HistGB + nb", "PR-AUC own history", "PR-AUC persistence"]].round(3))
display(dd[["distance", "events", "PR-AUC Persistence", "PR-AUC Neighbour_ratio", "PR-AUC HistGB_nb", "dPR-AUC (nb - persistence)", "95% CI low", "95% CI high"]].round(3))
