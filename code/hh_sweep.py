"""Heavy-hex (ibm_marrakesh, 156 qubits) pauli-path sweep: reuses the tested 1D sweep with the heavy-hex circuit.
Results go to ~/bq3/hh/sweep_results.jsonl (separate from the 1D results)."""
import os, sys, json, math, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
sys.path.insert(0, os.path.dirname(HERE))
import bq3_sweep as S
LAYERS = json.load(open("marrakesh_layers.json"))["layers"]


def kicked_ising_hh(n, steps, theta_h):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for _ in range(steps):
        for q in range(n): qc.rx(theta_h, q)
        for L in LAYERS:
            for a, b in L: qc.rzz(-math.pi / 2, a, b)
    return qc


S.kicked_ising = kicked_ising_hh
shutil.copyfile("hh_exact_truth.json", "exact_truth.json")
if __name__ == "__main__":
    if not (len(sys.argv) > 1 and sys.argv[1] == "summary"):
        S.run()
    S.summary()
