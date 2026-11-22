"""Threshold-equivalence check (run inline on 2026-11-21; saved here verbatim as a script).
If the pauli-path truncation threshold is absolute, then sum_i Z_i (coefficient 1) at threshold 1e-3, divided by n,
must equal the 1/n-scaled average at threshold 1e-5 exactly. Run from the folder holding bq3_sweep.py and
sweep_results.jsonl."""
import os, json, bluequbit
from bq3_sweep import kicked_ising, load_done
bq = bluequbit.init(open(os.path.expanduser("~/.bq_token")).read().strip())
scaled = load_done(); out = []
print(f"{'T':>2} {'theta':>5} | {'scaled 1/n @1e-5':>17} | {'unscaled/n @1e-3':>17} | same?")
for T in (5, 6):
    for th in (0.9, 1.1, 1.3):
        s = scaled[(T, th, 1e-5)]
        obs = [("I" * i + "Z" + "I" * (99 - i), 1.0) for i in range(100)]
        try:
            r = bq.run(kicked_ising(100, T, th), device="pauli-path", pauli_sum=obs, options={"pauli_path_truncation_threshold": 1e-3})
            v = r.expectation_value; v = float(v[0] if isinstance(v, list) else v) / 100
            if float(getattr(r, "cost", 0) or 0) > 0: raise SystemExit("STOPPED: charged")
        except Exception as e:
            if "FAILED_VALIDATION" not in str(e): raise
            v = None
        sv = s.get("value")
        same = (v is not None and sv is not None and abs(v - sv) < 1e-12)
        out.append(dict(steps=T, theta=th, scaled_1e5=sv, unscaled_1e3_over_n=v, same=same))
        print(f"{T:>2} {th:>5} | {str(sv)[:17]:>17} | {str(v)[:17]:>17} | {'SAME' if same else 'DIFFERENT'}", flush=True)
json.dump(out, open("equivalence_results.json", "w"), indent=1)
print(f"\n{sum(o['same'] for o in out)} of {len(out)} identical")
