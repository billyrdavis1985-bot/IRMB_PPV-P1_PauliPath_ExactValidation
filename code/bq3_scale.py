"""Scaling test: does normalizing the observable (coefficients 1/n) make pauli-path truncation n times coarser?
Re-runs the hardest 1D points with the UNSCALED sum (coefficient 1 per term), divides by n, compares. Free."""
import os, sys, json
from bq3_sweep import kicked_ising, load_done

POINTS = [(T, th) for T in (5, 6) for th in (0.9, 1.1, 1.3)]
THRESHOLDS = (1e-4, 1e-5)


def main():
    import bluequbit
    truth = {(t["steps"], t["theta"]): t for t in json.load(open("exact_truth.json"))["truth"]}
    scaled = load_done()
    bq = bluequbit.init(open(os.path.expanduser("~/.bq_token")).read().strip())
    out = []
    print(f"{'T':>2} {'theta':>5} {'thr':>6} | {'scaled (1/n) error':>19} | {'unscaled error':>15}")
    for T, th in POINTS:
        t = truth[(T, th)]; n = t["n"]
        obs = [("I" * i + "Z" + "I" * (n - i - 1), 1.0) for i in range(n)]
        for thr in THRESHOLDS:
            rec = dict(steps=T, theta=th, threshold=thr, exact=t["exact"])
            s = scaled.get((T, th, thr), {})
            rec["scaled_status"] = s.get("status", "ok") if s else "missing"
            rec["scaled_error"] = s.get("error") if s else None
            try:
                r = bq.run(kicked_ising(n, T, th), device="pauli-path", pauli_sum=obs,
                           options={"pauli_path_truncation_threshold": thr})
                v = r.expectation_value; v = float(v[0] if isinstance(v, list) else v) / n
                cost = float(getattr(r, "cost", 0) or 0)
                rec.update(unscaled_status="ok", unscaled_value=v, unscaled_error=v - t["exact"], cost=cost, job_id=r.job_id)
            except Exception as e:
                if "FAILED_VALIDATION" not in str(e): raise
                rec.update(unscaled_status="refused", unscaled_value=None, unscaled_error=None, cost=0.0)
            out.append(rec)
            fmt = lambda st, e: "REFUSED" if st == "refused" else ("-" if e is None else f"{e:+.5f}")
            print(f"{T:>2} {th:>5} {thr:>6g} | {fmt(rec['scaled_status'], rec['scaled_error']):>19} | "
                  f"{fmt(rec['unscaled_status'], rec['unscaled_error']):>15}", flush=True)
            json.dump(out, open("scale_results.json", "w"), indent=1)
            if rec["cost"] > 0: sys.exit(f"STOPPED: job was charged ${rec['cost']}")
    both = [r for r in out if r["unscaled_status"] == "ok" and r["scaled_status"] == "ok" and r["scaled_error"] is not None]
    if both:
        better = sum(abs(r["unscaled_error"]) < abs(r["scaled_error"]) for r in both)
        print(f"\nunscaled more accurate in {better} of {len(both)} comparable cases; "
              f"max |error| scaled {max(abs(r['scaled_error']) for r in both):.5f} vs unscaled {max(abs(r['unscaled_error']) for r in both):.5f}")
    print(f"refused: scaled {sum(r['scaled_status'] == 'refused' for r in out)}, unscaled {sum(r['unscaled_status'] == 'refused' for r in out)}; "
          f"total cost ${sum(r['cost'] for r in out):.2f}")


if __name__ == "__main__":
    main()
