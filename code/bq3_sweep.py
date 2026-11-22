"""Project #3 sweep: BlueQubit pauli-path vs exact ground truth. Resumable; stops if any job costs > $0.
A threshold BlueQubit refuses (FAILED_VALIDATION) is recorded as 'refused' and the sweep continues;
any other error stops the run (re-run to resume)."""
import os, sys, json, math, time

THRESHOLDS = (1e-3, 1e-4, 1e-5)
TOL = 0.01


def kicked_ising(n, steps, theta_h):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for _ in range(steps):
        for q in range(n): qc.rx(theta_h, q)
        for start in (0, 1):
            for q in range(start, n - 1, 2): qc.rzz(-math.pi / 2, q, q + 1)
    return qc


def load_done():
    done = {}
    if os.path.exists("sweep_results.jsonl"):
        for line in open("sweep_results.jsonl"):
            r = json.loads(line); done[(r["steps"], r["theta"], r["threshold"])] = r
    return done


def run():
    import bluequbit
    truth = json.load(open("exact_truth.json"))["truth"]
    done = load_done()
    bq = bluequbit.init(open(os.path.expanduser("~/.bq_token")).read().strip())
    todo = [(t, thr) for t in truth for thr in THRESHOLDS if (t["steps"], t["theta"], thr) not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    for k, (t, thr) in enumerate(todo, 1):
        n = t["n"]
        obs = [("I" * i + "Z" + "I" * (n - i - 1), 1.0 / n) for i in range(n)]
        s = time.time()
        base = dict(n=n, steps=t["steps"], theta=t["theta"], threshold=thr, exact=t["exact"])
        try:
            r = bq.run(kicked_ising(n, t["steps"], t["theta"]), device="pauli-path", pauli_sum=obs,
                       options={"pauli_path_truncation_threshold": thr})
        except Exception as e:
            if "FAILED_VALIDATION" not in str(e):
                raise
            row = dict(base, status="refused", value=None, error=None, rel_error=None, cost=0.0,
                       message=str(e)[:300], wall_s=round(time.time() - s, 1))
            with open("sweep_results.jsonl", "a") as f: f.write(json.dumps(row) + "\n")
            print(f"[{k}/{len(todo)}] T={t['steps']} theta={t['theta']} thr={thr:g}: REFUSED by BlueQubit", flush=True)
            continue
        cost = float(getattr(r, "cost", 0) or 0)
        v = r.expectation_value; v = float(v[0] if isinstance(v, list) else v)
        row = dict(base, status="ok", value=v, error=v - t["exact"],
                   rel_error=((v - t["exact"]) / t["exact"] if t["exact"] else None), cost=cost,
                   run_ms=getattr(r, "run_time_ms", None), wall_s=round(time.time() - s, 1), job_id=r.job_id)
        with open("sweep_results.jsonl", "a") as f: f.write(json.dumps(row) + "\n")
        print(f"[{k}/{len(todo)}] T={t['steps']} theta={t['theta']} thr={thr:g}: {v:.6f} vs {t['exact']:.6f} "
              f"(err {v - t['exact']:+.2e}) ${cost}", flush=True)
        if cost > 0:
            sys.exit(f"STOPPED: job {r.job_id} cost ${cost} - pauli-path is no longer free; nothing more submitted")


def summary():
    rows = list(load_done().values())
    if not rows: return print("no results yet")
    ok_rows = [r for r in rows if r.get("status", "ok") == "ok"]
    steps = sorted({r["steps"] for r in rows}); thetas = sorted({r["theta"] for r in rows})
    get = lambda T, th, thr: next((x for x in rows if x["steps"] == T and x["theta"] == th and x["threshold"] == thr), None)
    for thr in THRESHOLDS:
        print(f"\nerror x 100 (value - exact) at threshold {thr:g}   (R = refused by BlueQubit)")
        print("      " + "".join(f"{th:>9}" for th in thetas))
        for T in steps:
            cells = []
            for th in thetas:
                r = get(T, th, thr)
                cells.append(f"{'-':>9}" if r is None else f"{'R':>9}" if r.get("status") == "refused" else f"{100 * r['error']:9.3f}")
            print(f"T={T:<3} " + "".join(cells))
    print("\nbest accuracy BlueQubit allowed: |error| x 100 at the tightest ACCEPTED threshold (threshold in brackets)")
    print("      " + "".join(f"{th:>14}" for th in thetas))
    best = {}
    for T in steps:
        cells = []
        for th in thetas:
            acc = [x for x in ok_rows if x["steps"] == T and x["theta"] == th]
            if not acc:
                cells.append(f"{'-':>14}"); continue
            b = min(acc, key=lambda x: x["threshold"]); best[(T, th)] = b
            cells.append(f"{100 * abs(b['error']):8.3f} [{b['threshold']:.0e}]")
        print(f"T={T:<3} " + "".join(cells))
    print(f"\ntrust boundary: steps where the best allowed |error| < {TOL}")
    for th in thetas:
        good = [T for T in steps if (T, th) in best and abs(best[(T, th)]["error"]) < TOL]
        bad = [T for T in steps if (T, th) in best and T not in good]
        print(f"  theta={th}: within tolerance at T={good}; outside at T={bad}")
    refused = [r for r in rows if r.get("status") == "refused"]
    zeros = [r for r in ok_rows if r["value"] == 0.0 and r["exact"] != 0]
    neg = sum(r["error"] < -1e-12 for r in ok_rows); exact_hits = sum(abs(r["error"]) < 1e-12 for r in ok_rows)
    print(f"\nrefused by BlueQubit: {len(refused)}  " + ", ".join(f"T={r['steps']}/th={r['theta']}/{r['threshold']:g}" for r in refused))
    print(f"silent zeros (returned exactly 0 while the truth is not 0): {len(zeros)}")
    print(f"of {len(ok_rows)} answered: {exact_hits} exact (|err|<1e-12), {neg} underestimate, "
          f"{len(ok_rows) - exact_hits - neg} overestimate; total cost ${sum(r['cost'] for r in rows):.2f}")


if __name__ == "__main__":
    if not (len(sys.argv) > 1 and sys.argv[1] == "summary"):
        run()
    summary()
