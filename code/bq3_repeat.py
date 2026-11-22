"""Are BlueQubit pauli-path answers and refusals deterministic? Re-runs refused, silent zeros, 6 controls. Free."""
import os, sys, json, time
from bq3_sweep import kicked_ising, load_done


def main():
    import bluequbit
    rows = list(load_done().values())
    refused = [r for r in rows if r.get("status") == "refused"]
    zeros = [r for r in rows if r.get("status", "ok") == "ok" and r["value"] == 0.0 and r["exact"] != 0]
    normal = sorted([r for r in rows if r.get("status", "ok") == "ok" and r not in zeros], key=lambda r: -abs(r["error"]))[:6]
    picks = [("refused", r) for r in refused] + [("silent_zero", r) for r in zeros] + [("control", r) for r in normal]
    bq = bluequbit.init(open(os.path.expanduser("~/.bq_token")).read().strip())
    out = []
    for kind, r in picks:
        n = r["n"]; obs = [("I" * i + "Z" + "I" * (n - i - 1), 1.0 / n) for i in range(n)]
        rec = dict(kind=kind, steps=r["steps"], theta=r["theta"], threshold=r["threshold"],
                   first_status=r.get("status", "ok"), first_value=r.get("value"))
        try:
            j = bq.run(kicked_ising(n, r["steps"], r["theta"]), device="pauli-path", pauli_sum=obs,
                       options={"pauli_path_truncation_threshold": r["threshold"]})
            v = j.expectation_value; v = float(v[0] if isinstance(v, list) else v)
            cost = float(getattr(j, "cost", 0) or 0)
            rec.update(second_status="ok", second_value=v, cost=cost, job_id=j.job_id,
                       identical=(rec["first_status"] == "ok" and v == rec["first_value"]))
            if cost > 0:
                out.append(rec); break
        except Exception as e:
            if "FAILED_VALIDATION" not in str(e): raise
            rec.update(second_status="refused", second_value=None, cost=0.0, identical=(rec["first_status"] == "refused"))
        out.append(rec)
        print(f"{kind:11s} T={rec['steps']} theta={rec['theta']} thr={rec['threshold']:g}: "
              f"first {rec['first_status']} {rec['first_value']}  ->  second {rec['second_status']} {rec['second_value']}  "
              f"{'SAME' if rec['identical'] else 'DIFFERENT'}", flush=True)
    json.dump(out, open("repeat_results.json", "w"), indent=1)
    same = sum(r["identical"] for r in out)
    print(f"\n{same} of {len(out)} identical on repeat; cost ${sum(r['cost'] for r in out):.2f}")
    if any(r["cost"] > 0 for r in out): print("STOPPED: a job was charged")


if __name__ == "__main__":
    main()
