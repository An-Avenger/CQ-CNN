"""
Quantum Circuit & Variational Ansatz Library for CQ-CNN
=======================================================
Implements and benchmarks multiple variational quantum circuits (PQCs):
  1. RealAmplitudes (beta_8 baseline from the paper)
  2. EfficientSU2 (Strongly Entangling ansatz with Ry/Rz and full connectivity)
  3. Hardware-Efficient Ansatz (HEA with nearest-neighbor CNOT ladders)
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator


def create_pqc_circuit(n_qubits=2, ansatz_type="real_amplitudes", reps=1, entanglement="linear"):
    """
    Constructs a composite QuantumCircuit with ZZFeatureMap + Parametric Ansatz.
    """
    feature_map = ZZFeatureMap(n_qubits)

    if ansatz_type == "real_amplitudes":
        ansatz = RealAmplitudes(n_qubits, reps=reps, entanglement=entanglement)
    elif ansatz_type == "efficient_su2":
        ansatz = EfficientSU2(n_qubits, su2_gates=['ry', 'rz'], reps=reps, entanglement=entanglement)
    elif ansatz_type == "hea":
        ansatz = EfficientSU2(n_qubits, su2_gates=['rx', 'ry'], reps=reps, entanglement='linear')
    else:
        raise ValueError(f"Unknown ansatz_type: {ansatz_type}")

    qc = QuantumCircuit(n_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    return qc, feature_map.parameters, ansatz.parameters


def create_variational_qnn(n_qubits=2, ansatz_type="real_amplitudes", reps=1, entanglement="linear", estimator=None):
    """
    Wraps the quantum circuit into an EstimatorQNN for PyTorch integration.
    """
    if estimator is None:
        estimator = Estimator()

    qc, in_params, weight_params = create_pqc_circuit(n_qubits, ansatz_type, reps, entanglement)

    return EstimatorQNN(
        circuit=qc,
        input_params=in_params,
        weight_params=weight_params,
        input_gradients=True,
        estimator=estimator
    )


if __name__ == "__main__":
    print("Testing Quantum Circuit Ansatz Library...")
    for atype in ["real_amplitudes", "efficient_su2", "hea"]:
        for ent in ["linear", "full"]:
            if atype == "hea" and ent == "full":
                continue
            qc, in_p, w_p = create_pqc_circuit(2, ansatz_type=atype, reps=1, entanglement=ent)
            print(f"[{atype} - {ent}] Qubits: {qc.num_qubits} | Input Params: {len(in_p)} | Weight Params: {len(w_p)} | Depth: {qc.depth()}")
    print("All quantum circuit topologies initialized successfully!")
