"""Regenerates every number, table and figure in PAPER.md from the raw job records in data/.
  python analysis.py      -> results/summary.json, results/tables.md, figures/fig1_error_vs_depth.png,
                             figures/fig2_false_convergence.png
"""
import json, os, math
import numpy as np

D, R, F = "data", "results", "figures"
os.makedirs(R, exist_ok=True); os.makedirs(F, exist_ok=True)
jl = lambda p: [json.loads(l) for l in open(os.path.join(D, p))]
js = lambda p: json.load(open(os.path.join(D, p)))

one = jl("1d/sweep_results.jsonl")
hha = jl("heavyhex/avg_sweep_results.jsonl")
hh5 = jl("heavyhex/t5_site_results.jsonl")
rep, scl, eqv = js("checks/repeat_results.json"), js("checks/scale_results.json"), js("checks/equivalence_results.json")
ok = lambda r: r.get("status", "ok") == "ok"
S, T = {}, []


def grid_table(rows, title, thr, steps, thetas, key="steps"):
    T.append(f"\n### {title} - error x 100 at threshold {thr:g} (R = refused)\n")
    T.append("| steps | " + " | ".join(f"θ={t}" for t in thetas) + " |")
    T.append("|---|" + "---|" * len(thetas))
    for s in steps:
        cells = []
        for th in thetas:
            r = next((x for x in rows if x[key] == s and x["theta"] == th and x["threshold"] == thr), None)
            cells.append("–" if r is None else "R" if not ok(r) else f"{100 * r['error']:.3f}")
        T.append(f"| {s} | " + " | ".join(cells) + " |")


# ---------------- averaged observable (1D chain and heavy-hex)
for name, rows in (("1d_avg", one), ("heavyhex_avg", hha)):
    steps = sorted({r["steps"] for r in rows}); thetas = sorted({r["theta"] for r in rows})
    a = [r for r in rows if ok(r)]
    best = {}
    for s in steps:
        for th in thetas:
            acc = [x for x in a if x["steps"] == s and x["theta"] == th]
            if acc: best[(s, th)] = min(acc, key=lambda x: x["threshold"])
    S[name] = dict(
        jobs=len(rows), answered=len(a), refused=len(rows) - len(a),
        refused_list=[(r["steps"], r["theta"], r["threshold"]) for r in rows if not ok(r)],
        exact_hits=sum(abs(r["error"]) < 1e-12 for r in a), under=sum(r["error"] < -1e-12 for r in a),
        over=sum(r["error"] > 1e-12 for r in a),
        silent_zeros=[(r["steps"], r["theta"], r["threshold"], r["exact"]) for r in a if r["value"] == 0.0 and r["exact"] != 0],
        worst_by_threshold={f"{thr:g}": max((abs(r["error"]) for r in a if r["threshold"] == thr), default=None) for thr in (1e-3, 1e-4, 1e-5)},
        best_allowed_outside_0p01=[(s, th, round(abs(b["error"]), 5), b["threshold"]) for (s, th), b in best.items() if abs(b["error"]) >= 0.01],
        points=len(best), cost=sum(r["cost"] for r in rows))
    for thr in (1e-3, 1e-4, 1e-5):
        grid_table(rows, name, thr, steps, thetas)

# false convergence on the averaged observables (change 1e-4 -> 1e-5 smaller than remaining error, error > 0.01)
def false_conv(rows, keyf):
    idx = {keyf(r) + (r["threshold"],): r for r in rows}
    out = []
    for k, r in idx.items():
        if k[-1] != 1e-5 or not ok(r): continue
        r4 = idx.get(k[:-1] + (1e-4,))
        if r4 is not None and ok(r4):
            out.append(dict(key=k[:-1], change=abs(r["value"] - r4["value"]), error=abs(r["error"]), exact=r["exact"]))
    return out
fc1 = false_conv(one, lambda r: (r["steps"], r["theta"]))
fch = false_conv(hha, lambda r: (r["steps"], r["theta"]))
fc5 = false_conv(hh5, lambda r: (r["theta"], r["site"]))
for name, fc in (("1d_avg", fc1), ("heavyhex_avg", fch), ("heavyhex_t5_sites", fc5)):
    bad = [x for x in fc if x["error"] > x["change"] and x["error"] > 0.01]                       # pre-set absolute rule
    bad_rel = [x for x in fc if x["error"] > x["change"] and x["error"] > 0.1 * abs(x["exact"])]   # relative criterion (>10%)
    S.setdefault(name, {})["false_convergence"] = dict(
        pairs=len(fc), cases_abs_0p01=len(bad), cases_rel_10pct=len(bad_rel),
        detail_abs=[dict(x, rel=round(x["error"] / abs(x["exact"]), 3)) for x in sorted(bad, key=lambda x: -x["error"])],
        detail_rel=[dict(x, rel=round(x["error"] / abs(x["exact"]), 3)) for x in sorted(bad_rel, key=lambda x: -x["error"] / abs(x["exact"]))])

# ---------------- heavy-hex, 5 steps, per site
a5 = [r for r in hh5 if ok(r)]
per = {}
for th in sorted({r["theta"] for r in hh5}):
    for thr in (1e-4, 1e-5):
        g = [r for r in hh5 if r["theta"] == th and r["threshold"] == thr]; o = [r for r in g if ok(r)]
        e = [abs(r["error"]) for r in o]
        per[f"{th}|{thr:g}"] = dict(answered=len(o), refused=len(g) - len(o), max_err=max(e) if e else None,
                                    mean_err=float(np.mean(e)) if e else None,
                                    rel_gt_10pct=sum(abs(r["error"]) > 0.1 * abs(r["exact"]) for r in o),
                                    under=sum(r["error"] < -1e-12 for r in o), over=sum(r["error"] > 1e-12 for r in o))
S["heavyhex_t5_sites"].update(jobs=len(hh5), sites=sorted({r["site"] for r in hh5}), refused=len(hh5) - len(a5),
                              max_err=max(abs(r["error"]) for r in a5), by_theta_threshold=per,
                              ordering=js("heavyhex/t5_ordering.json"), cost=sum(r["cost"] for r in hh5))
T.append("\n### Heavy-hex, 5 steps, 30 sites - per angle and threshold\n")
T.append("| θ | threshold | answered | refused | max abs error | mean abs error | >10% relative | low / high |")
T.append("|---|---|---|---|---|---|---|---|")
for k, v in per.items():
    th, thr = k.split("|")
    mx = "–" if v["max_err"] is None else f"{v['max_err']:.4f}"
    mn = "–" if v["mean_err"] is None else f"{v['mean_err']:.5f}"
    T.append(f"| {th} | {thr} | {v['answered']} | {v['refused']} | {mx} | {mn} | {v['rel_gt_10pct']} | {v['under']} / {v['over']} |")

# ---------------- checks
S["repeat"] = dict(jobs=len(rep), identical=sum(r["identical"] for r in rep),
                   kinds={k: sum(r["kind"] == k for r in rep) for k in ("refused", "silent_zero", "control")})
S["scaling"] = dict(jobs=len(scl), unscaled_refused=sum(r["unscaled_status"] == "refused" for r in scl),
                    scaled_refused=sum(r["scaled_status"] == "refused" for r in scl))
S["equivalence"] = dict(points=len(eqv), both_ran=sum(e["unscaled_1e3_over_n"] is not None for e in eqv),
                        identical=sum(e["same"] for e in eqv),
                        max_diff=max(abs(e["scaled_1e5"] - e["unscaled_1e3_over_n"]) for e in eqv if e["unscaled_1e3_over_n"] is not None),
                        unscaled_refused=[(e["steps"], e["theta"]) for e in eqv if e["unscaled_1e3_over_n"] is None])
allrows = one + hha + hh5 + rep + scl
S["totals"] = dict(bluequbit_jobs=len(one) + len(hha) + len(hh5) + len(rep) + len(scl) + len(eqv) + 2,
                   cost=sum(r.get("cost", 0) for r in allrows))
json.dump(S, open(os.path.join(R, "summary.json"), "w"), indent=1, default=str)
open(os.path.join(R, "tables.md"), "w").write("\n".join(T) + "\n")

# ---------------- figures
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = {1e-3: "#2a78d6", 1e-4: "#eb6834", 1e-5: "#1baf7a"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF})

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (title, rows) in zip(axes, (("1D chain, 100 qubits (average ⟨Z⟩)", one), ("Heavy-hex, ibm_marrakesh layout, 156 qubits (average ⟨Z⟩)", hha))):
    steps = sorted({r["steps"] for r in rows})
    for thr in (1e-3, 1e-4, 1e-5):
        y = [max([abs(r["error"]) for r in rows if ok(r) and r["steps"] == s and r["threshold"] == thr] or [np.nan]) for s in steps]
        y = [max(v, 1e-16) for v in y]
        ax.plot(steps, y, color=C[thr], lw=2, marker="o", ms=8, mec=SURF, mew=2, label=f"threshold {thr:g}")
    ax.axhline(0.01, color=INK2, lw=1, ls="--"); ax.text(steps[0] + 0.05, 0.016, "0.01 tolerance", color=INK2, fontsize=8, ha="left", va="bottom")
    ax.set_yscale("log"); ax.set_ylim(1e-16, 1.5); ax.set_xticks(steps); ax.set_xlabel("Trotter steps")
    ax.set_title(title, fontsize=10, color=INK, loc="left"); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("worst |error| across 7 angles")
axes[0].legend(frameon=False, loc="lower right", fontsize=9)
fig.suptitle("Pauli-path error vs exact ground truth: worst case per depth and threshold", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig1_error_vs_depth.png"), dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5.4))
FLOOR = 1e-7
shown = {}
for lab, fc, col in (("1D chain, average ⟨Z⟩ (coefficients 1/100)", fc1, "#2a78d6"),
                     ("Heavy-hex, average ⟨Z⟩ (coefficients 1/156)", fch, "#1baf7a"),
                     ("Heavy-hex, 5 steps, single sites (coefficient 1)", fc5, "#eb6834")):
    pts = [p for p in fc if p["error"] >= FLOOR]
    shown[lab] = (len(pts), len(fc))
    ax.scatter([max(p["change"], FLOOR) for p in pts], [p["error"] for p in pts], s=48, color=col, edgecolor=SURF,
               linewidth=1.5, label=f"{lab}  [{len(pts)} of {len(fc)} shown]", zorder=3)
lim = (FLOOR, 1)
ax.plot(lim, lim, color=INK2, lw=1, zorder=2)
ax.axhspan(0.01, 1, color="#e34948", alpha=0.06, lw=0, zorder=1)
ax.text(1.3 * FLOOR, 0.6, "error > 0.01", color=INK2, fontsize=8, va="top")
ax.text(1.3 * FLOOR, 2e-3, "above the diagonal: tightening the threshold\nmoved the answer LESS than its remaining error", color=INK2, fontsize=8, va="top")
for p in fc1:
    if p["error"] > p["change"] and p["error"] > 0.1 * abs(p["exact"]):
        ax.annotate(f"1D, {p['key'][0]} steps, θ={p['key'][1]}: {100 * p['error'] / abs(p['exact']):.0f}% low",
                    (p["change"], p["error"]), xytext=(-12, 10 if p["error"] > 0.01 else -14), textcoords="offset points",
                    ha="right", fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("change in the answer when the threshold is tightened 1e-4 → 1e-5")
ax.set_ylabel("remaining |error| at 1e-5 (vs exact)")
ax.set_title("Does tightening the threshold reveal the error?", fontsize=10, color=INK, loc="left")
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=1)
fig.text(0.01, 0.005, f"Points with error < {FLOOR:g} (exact to rounding) are omitted; changes below {FLOOR:g} are drawn at the left edge.",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(F, "fig2_false_convergence.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
print(json.dumps({k: (v if k in ("totals", "repeat", "scaling", "equivalence") else {kk: vv for kk, vv in v.items() if kk in ("jobs", "refused", "worst_by_threshold", "false_convergence", "silent_zeros", "under", "over", "max_err")}) for k, v in S.items()}, indent=1, default=str)[:6000])
