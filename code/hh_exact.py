"""Exact average <Z> for the kicked-Ising circuit on the ibm_marrakesh heavy-hex layout (156 qubits), by exact
tensor-network contraction (quimb), cross-checked against an independent statevector light-cone simulation."""
import sys, json, math, time, numpy as np
import quimb.tensor as qtn

LAYERS = json.load(open("marrakesh_layers.json"))["layers"]
N = json.load(open("marrakesh_layers.json"))["n"]
STEPS, ANGLES = (1, 2, 3, 4), (0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3)
Z = np.diag([1.0, -1.0])


def circuit(n, layers, T, th):
    c = qtn.Circuit(n)
    for _ in range(T):
        for q in range(n): c.apply_gate("RX", th, q)
        for L in layers:
            for a, b in L: c.apply_gate("RZZ", -math.pi / 2, a, b)
    return c


def site_z_tn(c, i):
    return float(np.real(c.local_expectation(Z, i, optimize="greedy", simplify_sequence="ADCR")))


def site_z_statevector(n, layers, T, th, i, max_window=24):
    C = {i}
    for _ in range(T):
        for L in reversed(layers):
            for a, b in L:
                if a in C or b in C: C |= {a, b}
    S = sorted(C); idx = {q: k for k, q in enumerate(S)}; m = len(S)
    if m > max_window: return None
    psi = np.zeros(2 ** m, complex); psi[0] = 1
    psi = psi.reshape([2] * m)
    cx, sx = math.cos(th / 2), math.sin(th / 2)
    rx = np.array([[cx, -1j * sx], [-1j * sx, cx]])
    zz = np.exp(-1j * (-math.pi / 2) / 2 * np.array([[1, -1], [-1, 1]]))
    for _ in range(T):
        for q in S: psi = np.moveaxis(np.tensordot(rx, psi, axes=([1], [idx[q]])), 0, idx[q])
        for L in layers:
            for a, b in L:
                if a in C and b in C:
                    sl = [slice(None)] * m
                    for va in (0, 1):
                        for vb in (0, 1):
                            sl[idx[a]], sl[idx[b]] = va, vb
                            psi[tuple(sl)] *= zz[va, vb]
    p = np.abs(psi) ** 2
    return float(p.take(0, axis=idx[i]).sum() - p.take(1, axis=idx[i]).sum())


def avg_z(T, th):
    c = circuit(N, LAYERS, T, th)
    return float(np.mean([site_z_tn(c, i) for i in range(N)]))


def self_checks():
    out = {}
    for th in (0.3, 1.1):
        c = circuit(N, LAYERS, 2, th)
        out[f"tn_vs_statevector_T2_theta{th}"] = max(abs(site_z_tn(c, i) - site_z_statevector(N, LAYERS, 2, th, i)) for i in range(N))
    for th in (0.7, 1.3):
        c = circuit(N, LAYERS, 3, th)
        diffs = [abs(site_z_tn(c, i) - v) for i in range(N)
                 for v in [site_z_statevector(N, LAYERS, 3, th, i, max_window=22)] if v is not None]
        out[f"tn_vs_statevector_T3_theta{th}_{len(diffs)}sites"] = max(diffs)
    return out


if __name__ == "__main__":
    t = time.time()
    chk = self_checks(); ok = all(v < 1e-9 for v in chk.values())
    print("self-checks:", {k: f"{v:.1e}" for k, v in chk.items()}, "->", "PASS" if ok else "FAIL", f"({time.time() - t:.0f}s)", flush=True)
    if not ok: sys.exit("self-checks failed")
    truth = []
    for T in STEPS:
        for th in ANGLES:
            s = time.time(); v = avg_z(T, th)
            truth.append(dict(n=N, steps=T, theta=th, exact=v))
            json.dump(dict(layout="ibm_marrakesh heavy-hex", checks=chk, truth=truth), open("hh_exact_truth.json", "w"), indent=1)
            print(f"T={T} theta={th}: exact <Z> = {v:.12f}  ({time.time() - s:.1f}s)", flush=True)
    print(f"saved hh_exact_truth.json ({len(truth)} points, {time.time() - t:.0f}s)")
