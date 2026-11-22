"""Heavy-hex, 5 steps, site by site: BlueQubit pauli-path <Z_i> vs exact. 30 hardest exactly-checkable sites (fixed
rule), thresholds 1e-4 and 1e-5, free ordering probe first. Resumable; refusals recorded; stops if any job costs > $0."""
import os, sys, json, math, time

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
LAYERS = json.load(open("marrakesh_layers.json"))["layers"]
N = json.load(open("marrakesh_layers.json"))["n"]
T, THRESHOLDS, N_SITES, TOL = 5, (1e-4, 1e-5), 30, 0.01
RESULTS = "hh5_results.jsonl"


def circuit(theta):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(N)
    for _ in range(T):
        for q in range(N): qc.rx(theta, q)
        for L in LAYERS:
            for a, b in L: qc.rzz(-math.pi / 2, a, b)
    return qc


def zstring(i, order):
    s = ["I"] * N
    s[i if order == "qubit0_first" else N - 1 - i] = "Z"
    return "".join(s)


def ordering_probe(bq):
    if os.path.exists("hh5_ordering.json"):
        return json.load(open("hh5_ordering.json"))["order"]
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(N); qc.x(0)
    vals = {}
    for order in ("qubit0_first", "qiskit_order"):
        r = bq.run(qc, device="pauli-path", pauli_sum=[(zstring(0, order), 1.0)], options={"pauli_path_truncation_threshold": 1e-3})
        if float(getattr(r, "cost", 0) or 0) > 0: sys.exit("STOPPED: probe job was charged")
        v = r.expectation_value; vals[order] = float(v[0] if isinstance(v, list) else v)
    hits = [o for o, v in vals.items() if abs(v + 1) < 1e-9]
    if len(hits) != 1 or any(abs(v - 1) > 1e-9 for o, v in vals.items() if o not in hits):
        sys.exit(f"ordering probe ambiguous: {vals} - stopping")
    json.dump(dict(order=hits[0], probe=vals), open("hh5_ordering.json", "w"), indent=1)
    print(f"ordering probe: X on qubit 0 gives -1 with {hits[0]} strings ({vals})", flush=True)
    return hits[0]


def chosen_sites(ex):
    W = ex["widths"]["0.9"]
    feas = [int(i) for i in ex["sites"]["0.9"]]
    feas = [i for i in feas if all(str(i) in ex["sites"][th] for th in ex["sites"])]
    return sorted(feas, key=lambda i: (-W[str(i)], i))[:N_SITES]


def load_done():
    done = {}
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            r = json.loads(line); done[(r["theta"], r["site"], r["threshold"])] = r
    return done


def run():
    import bluequbit
    ex = json.load(open("hh5_exact_sites.json"))
    bq = bluequbit.init(open(os.path.expanduser("~/.bq_token")).read().strip())
    order = ordering_probe(bq)
    sites = chosen_sites(ex); done = load_done()
    todo = [(float(th), i, thr) for th in ex["sites"] for i in sites for thr in THRESHOLDS if (float(th), i, thr) not in done]
    print(f"sites {sites}\n{len(done)} done, {len(todo)} to run", flush=True)
    circs = {}
    for k, (th, i, thr) in enumerate(todo, 1):
        exact = ex["sites"][str(th)][str(i)]
        base = dict(steps=T, theta=th, site=i, threshold=thr, exact=exact, width=ex["widths"][str(th)][str(i)], order=order)
        if th not in circs: circs[th] = circuit(th)
        try:
            r = bq.run(circs[th], device="pauli-path", pauli_sum=[(zstring(i, order), 1.0)],
                       options={"pauli_path_truncation_threshold": thr})
        except Exception as e:
            if "FAILED_VALIDATION" not in str(e): raise
            with open(RESULTS, "a") as f: f.write(json.dumps(dict(base, status="refused", value=None, error=None, cost=0.0)) + "\n")
            print(f"[{k}/{len(todo)}] theta={th} site={i} thr={thr:g}: REFUSED", flush=True); continue
        cost = float(getattr(r, "cost", 0) or 0)
        v = r.expectation_value; v = float(v[0] if isinstance(v, list) else v)
        with open(RESULTS, "a") as f:
            f.write(json.dumps(dict(base, status="ok", value=v, error=v - exact, cost=cost, job_id=r.job_id)) + "\n")
        print(f"[{k}/{len(todo)}] theta={th} site={i} thr={thr:g}: {v:.6f} vs {exact:.6f} (err {v - exact:+.2e}) ${cost}", flush=True)
        if cost > 0: sys.exit(f"STOPPED: job {r.job_id} cost ${cost}")


def summary():
    rows = list(load_done().values())
    if not rows: return print("no results yet")
    print(f"\nper-site comparison, heavy-hex, {T} steps, {len({r['site'] for r in rows})} sites (hardest exactly-checkable)")
    print(f"{'theta':>6} {'thr':>6} {'answered':>9} {'refused':>8} {'max|err|':>9} {'mean|err|':>10} {'>0.01':>6} {'>10% rel':>9} {'low/high':>9}")
    for th in sorted({r["theta"] for r in rows}):
        for thr in THRESHOLDS:
            g = [r for r in rows if r["theta"] == th and r["threshold"] == thr]
            ok = [r for r in g if r["status"] == "ok"]
            if not ok:
                print(f"{th:>6} {thr:>6g} {0:>9} {len(g):>8}"); continue
            e = [abs(r["error"]) for r in ok]
            rel = sum(abs(r["error"]) > 0.1 * abs(r["exact"]) for r in ok if abs(r["exact"]) > 1e-12)
            lo = sum(r["error"] < -1e-12 for r in ok); hi = sum(r["error"] > 1e-12 for r in ok)
            print(f"{th:>6} {thr:>6g} {len(ok):>9} {len(g) - len(ok):>8} {max(e):9.4f} {sum(e) / len(e):10.5f} "
                  f"{sum(x > TOL for x in e):>6} {rel:>9} {f'{lo}/{hi}':>9}")
    pairs, done = [], load_done()
    for (th, i, thr), r in done.items():
        if thr != 1e-5 or r["status"] != "ok": continue
        r4 = done.get((th, i, 1e-4))
        if r4 and r4["status"] == "ok":
            pairs.append((abs(r["value"] - r4["value"]), abs(r["error"]), r))
    fc = [(d, e, r) for d, e, r in pairs if e > d and e > TOL]
    print(f"\nfalse convergence (change 1e-4 -> 1e-5 smaller than the remaining error, and error > {TOL}): {len(fc)} of {len(pairs)} site-angle pairs")
    for d, e, r in sorted(fc, key=lambda x: -x[1])[:10]:
        print(f"  theta={r['theta']} site={r['site']}: changed {d:.4f}, still off by {e:.4f} (exact {r['exact']:.4f}, {100 * e / max(abs(r['exact']), 1e-12):.0f}% relative)")
    zeros = [r for r in rows if r["status"] == "ok" and r["value"] == 0.0 and abs(r["exact"]) > 1e-12]
    print(f"silent zeros: {len(zeros)}; total cost ${sum(r['cost'] for r in rows):.2f}")


if __name__ == "__main__":
    if not (len(sys.argv) > 1 and sys.argv[1] == "summary"):
        run()
    summary()
