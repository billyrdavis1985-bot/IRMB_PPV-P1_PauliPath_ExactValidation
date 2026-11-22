"""Exact average <Z> of the n-qubit kicked-Ising chain via light cones (ground truth for project #3)."""
import sys, json, math, time, numpy as np, pennylane as qml

N, STEPS, ANGLES = 100, (1, 2, 3, 4, 5, 6), (0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3)
DEV = "lightning.qubit" if "--cpu" in sys.argv else "lightning.gpu"


def window_z(n, T, th, i, R):
    a, b = max(0, i - R), min(n - 1, i + R); m = b - a + 1
    dev = qml.device(DEV, wires=m)
    @qml.qnode(dev)
    def c():
        for _ in range(T):
            for q in range(m): qml.RX(th, wires=q)
            for start in (0, 1):
                for g in range(start, n - 1, 2):
                    if a <= g and g + 1 <= b: qml.IsingZZ(-math.pi / 2, wires=[g - a, g + 1 - a])
        return qml.expval(qml.PauliZ(i - a))
    return float(c())


def avg_z(n, T, th, R=None):
    R = R or 2 * T + 1
    vals, cache = np.zeros(n), {}
    for i in range(n):
        j = min(i, n - 1 - i)
        if i - R >= 0 and i + R <= n - 1:
            key, site = ("bulk", i % 2), i
        else:
            key, site = ("edge", j), j
        if key not in cache: cache[key] = window_z(n, T, th, site, R)
        vals[i] = cache[key]
    return float(vals.mean())


def full_avg_z(n, T, th):
    dev = qml.device(DEV, wires=n)
    @qml.qnode(dev)
    def c():
        for _ in range(T):
            for q in range(n): qml.RX(th, wires=q)
            for start in (0, 1):
                for g in range(start, n - 1, 2): qml.IsingZZ(-math.pi / 2, wires=[g, g + 1])
        return [qml.expval(qml.PauliZ(q)) for q in range(n)]
    return float(np.mean(c()))


if __name__ == "__main__":
    t = time.time()
    checks = dict(lightcone_vs_full_n20_T3=abs(avg_z(20, 3, 0.3, R=7) - full_avg_z(20, 3, 0.3)),
                  radius_R_vs_R2_n100_T3=abs(avg_z(N, 3, 0.7) - avg_z(N, 3, 0.7, R=9)),
                  matches_reference_n100_T5_theta03=abs(avg_z(N, 5, 0.3) - 0.954230981863))
    ok = all(v < 1e-9 for v in checks.values())
    print("self-checks:", {k: f"{v:.1e}" for k, v in checks.items()}, "->", "PASS" if ok else "FAIL", flush=True)
    if not ok: sys.exit("ground-truth self-checks failed")
    truth = []
    for T in STEPS:
        for th in ANGLES:
            s = time.time(); v = avg_z(N, T, th)
            truth.append(dict(n=N, steps=T, theta=th, exact=v))
            print(f"T={T} theta={th}: exact <Z> = {v:.12f}  ({time.time() - s:.1f}s)", flush=True)
    json.dump(dict(device=DEV, checks=checks, truth=truth), open("exact_truth.json", "w"), indent=1)
    print(f"saved exact_truth.json ({len(truth)} points, {time.time() - t:.0f}s)")
