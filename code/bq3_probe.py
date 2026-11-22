"""
Project #3 probe: BlueQubit's large-circuit simulators on a kicked-Ising chain (the circuit family used in
"beyond-classical" hardware demonstrations), checked against exact answers we compute ourselves.

Circuit: n-qubit open chain, `steps` Trotter steps of  RX(theta_h) on every qubit, then RZZ(-pi/2) on even bonds,
then on odd bonds.  Observable: average magnetization <Z> = (1/n) sum_i <Z_i>  (independent of qubit ordering).
At theta_h = pi/2 the whole circuit is Clifford -> the exact answer is computable for ANY n (stabilizer simulation),
so every simulator can be checked against ground truth at 100+ qubits.

Decision it informs: which BlueQubit simulator (pauli-path, mps.cpu, tensor-network) handles which circuit size and
depth, at what cost - i.e. the design of the #3 study (sizes, depths, simulators, budget).

  python bq3_probe.py                 # FREE: local self-test + BlueQubit estimate() for n = 50/100/150
  python bq3_probe.py check --confirm # about $0.80: n=100, pauli-path and mps.cpu - Clifford case vs exact 1, generic case vs each other
Token from ~/.bq_token; never printed.
"""
import os, sys, json, math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, StabilizerState, Pauli, Statevector


def kicked_ising(n, steps, theta_h, clifford_form=False):
    """clifford_form=True builds the identical circuit (up to global phase) from cx/sdg/sx, for exact stabilizer
    simulation; only valid at theta_h = pi/2."""
    qc = QuantumCircuit(n)
    for _ in range(steps):
        for q in range(n):
            if clifford_form:
                qc.sx(q)                     # RX(pi/2) = e^{-i pi/4} SX
            else:
                qc.rx(theta_h, q)
        for start in (0, 1):
            for q in range(start, n - 1, 2):
                if clifford_form:            # RZZ(-pi/2) = CX . RZ(-pi/2) on target . CX ;  RZ(-pi/2) ~ Sdg
                    qc.cx(q, q + 1); qc.sdg(q + 1); qc.cx(q, q + 1)
                else:
                    qc.rzz(-math.pi / 2, q, q + 1)
    return qc


def avg_z_terms(n):
    return [("I" * i + "Z" + "I" * (n - i - 1), 1.0 / n) for i in range(n)]


def stabilizer_sum(n, steps):
    """(1/n) * sum of the n stabilizer generators of the final state (signed): exact value exactly 1 at theta_h=pi/2.
    The set is mirror-symmetric (the circuit is, for even n), so the value does not depend on qubit-ordering
    conventions. Labels are Qiskit order; reversing every string gives the same set."""
    labs = Clifford(kicked_ising(n, steps, math.pi / 2, clifford_form=True)).to_labels(mode="S")
    return [(l[1:], (1.0 if l[0] == "+" else -1.0) / n) for l in labs]


def exact_avg_z_clifford(n, steps):
    st = StabilizerState(Clifford(kicked_ising(n, steps, math.pi / 2, clifford_form=True)))
    return float(np.mean([st.expectation_value(Pauli("I" * (n - i - 1) + "Z" + "I" * i)) for i in range(n)]))


def self_test():
    """Clifford form == native circuit (statevector, n=10), for the exact-answer machinery to be trusted."""
    n, steps = 10, 4
    sv = Statevector(kicked_ising(n, steps, math.pi / 2))
    native = float(np.mean([sv.expectation_value(Pauli("I" * (n - i - 1) + "Z" + "I" * i)).real for i in range(n)]))
    exact = exact_avg_z_clifford(n, steps)
    from qiskit.quantum_info import SparsePauliOp
    terms = stabilizer_sum(n, steps)
    ssum = float(np.real(sv.expectation_value(SparsePauliOp([p for p, _ in terms], [c for _, c in terms]))))
    sym = sorted(terms) == sorted((p[::-1], c) for p, c in terms)
    return dict(n=n, steps=steps, avg_z_match=abs(native - exact) < 1e-9, stabilizer_sum=round(ssum, 12),
                mirror_symmetric=sym, match=abs(native - exact) < 1e-9 and abs(ssum - 1) < 1e-9 and sym)


def main():
    st = self_test(); print("self-test (Clifford form vs statevector, n=10):", st)
    if not st["match"]:
        sys.exit("self-test failed - do not use the exact answers")
    import bluequbit
    bq = bluequbit.init(open(os.path.expanduser("~/.bq_token")).read().strip())
    out = {"self_test": st}
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        if "--confirm" not in sys.argv:
            sys.exit("Add --confirm to spend about $0.80 (4 jobs).")
        n, steps = 100, 5
        cases = {"clifford_exact_1": (kicked_ising(n, steps, math.pi / 2), stabilizer_sum(n, steps), 1.0),
                 "generic_theta0.3_avgZ": (kicked_ising(n, steps, 0.3), avg_z_terms(n), None)}
        res = {}
        for case, (qc, obs, exact) in cases.items():
            for dev, opts in (("pauli-path", {"pauli_path_truncation_threshold": 1e-3}), ("mps.cpu", {"mps_bond_dimension": 64})):
                try:
                    r = bq.run(qc, device=dev, pauli_sum=obs, options=opts)
                    ev = r.expectation_value; ev = float(ev[0] if isinstance(ev, list) else ev) if ev is not None else None
                    res[f"{case}|{dev}"] = dict(value=ev, exact=exact, abs_error=(abs(ev - exact) if (ev is not None and exact is not None) else None),
                                                cost=str(getattr(r, "cost", None)), run_time_ms=getattr(r, "run_time_ms", None))
                except Exception as e:
                    res[f"{case}|{dev}"] = dict(error=str(e)[:300])
        out["check_n100_steps5"] = res
    else:
        est = {}
        for n in (50, 100, 150):
            for steps in (5, 10):
                qc = kicked_ising(n, steps, 0.3)       # generic (non-Clifford) angle
                for dev in ("pauli-path", "mps.cpu", "tensor-network"):
                    try:
                        e = bq.estimate(qc, device=dev)
                        est[f"n{n}_s{steps}_{dev}"] = dict(cost=e.estimated_cost, runtime_ms=e.estimated_runtime,
                                                           warning=e.warning_message, error=e.error_message)
                    except Exception as ex:
                        est[f"n{n}_s{steps}_{dev}"] = dict(error=str(ex)[:200])
        out["estimates"] = est
    json.dump(out, open("bq3_probe_result.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "self_test"}, indent=1))


if __name__ == "__main__":
    main()
