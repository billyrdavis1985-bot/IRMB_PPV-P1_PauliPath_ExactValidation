"""Heavy-hex, 5 steps, SITE BY SITE: exact <Z_i> for every site whose exact contraction fits (greedy width <= 26)."""
import sys, json, time, numpy as np
import hh_exact as H

T, ANGLES, W_MAX = 5, (0.5, 0.7, 0.9, 1.1, 1.3), 26


def main():
    t0 = time.time(); checks = {}
    c4 = H.circuit(H.N, H.LAYERS, 4, 0.9)
    checks["per_site_avg_T4_theta0.9_vs_reference"] = abs(np.mean([H.site_z_tn(c4, i) for i in range(H.N)]) - 0.350814271611)
    c5 = H.circuit(H.N, H.LAYERS, T, 0.9)
    sv = [(i, H.site_z_statevector(H.N, H.LAYERS, T, 0.9, i, max_window=24)) for i in range(H.N)]
    sv = [(i, v) for i, v in sv if v is not None]
    checks[f"tn_vs_statevector_T5_theta0.9_{len(sv)}sites"] = max(abs(H.site_z_tn(c5, i) - v) for i, v in sv) if sv else None
    ok = all(v is not None and v < 1e-9 for v in checks.values())
    print("self-checks:", {k: (f"{v:.1e}" if v is not None else None) for k, v in checks.items()}, "->",
          "PASS" if ok else "FAIL", f"({time.time() - t0:.0f}s)", flush=True)
    if not ok: sys.exit("self-checks failed")
    out = dict(steps=T, w_max=W_MAX, checks=checks, sites={}, widths={})
    for th in ANGLES:
        s = time.time(); c = H.circuit(H.N, H.LAYERS, T, th)
        W = {i: float(c.local_expectation_rehearse(H.Z, i, optimize="greedy", simplify_sequence="ADCR")["W"]) for i in range(H.N)}
        feas = [i for i in range(H.N) if W[i] <= W_MAX]
        out["widths"][str(th)] = W
        out["sites"][str(th)] = {str(i): H.site_z_tn(c, i) for i in feas}
        json.dump(out, open("hh5_exact_sites.json", "w"), indent=1)
        print(f"theta={th}: {len(feas)} sites exact ({time.time() - s:.0f}s)", flush=True)
    print(f"saved hh5_exact_sites.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
