"""
NISQ Quantum Noise Simulation for CQ-CNN
=========================================
Evaluates the physical realism of hybrid classical-quantum models by simulating
realistic noisy intermediate-scale quantum (NISQ) hardware conditions:
  - Depolarizing errors (single-qubit and two-qubit CNOT gates)
  - Bit-flip and readout / measurement errors
  - Finite shot-sampling statistics vs. ideal statevector
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit.primitives import BackendEstimatorV2, StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def build_nisq_noise_model(p1_gate_error=0.001, p2_gate_error=0.01, readout_error=0.02):
    """
    Constructs a calibrated Qiskit Aer NoiseModel mimicking superconducting quantum processors.
    """
    noise_model = NoiseModel()

    # 1. Single-qubit depolarizing error
    if p1_gate_error > 0:
        err_1q = depolarizing_error(p1_gate_error, 1)
        noise_model.add_all_qubit_quantum_error(err_1q, ['ry', 'rz', 'x', 'h', 'sx'])

    # 2. Two-qubit entangling depolarizing error
    if p2_gate_error > 0:
        err_2q = depolarizing_error(p2_gate_error, 2)
        noise_model.add_all_qubit_quantum_error(err_2q, ['cx'])

    # 3. Readout assignment error
    if readout_error > 0:
        p0_given_1 = readout_error
        p1_given_0 = readout_error
        ro_err = ReadoutError([[1 - p1_given_0, p1_given_0], [p0_given_1, 1 - p0_given_1]])
        noise_model.add_all_qubit_readout_error(ro_err)

    return noise_model


def create_decomposed_qnn(n_qubits=2, noise_model=None, shots=1024):
    """
    Creates an EstimatorQNN with decomposed elementary basis gates
    compatible with AerSimulator and noise modeling.
    """
    fm = ZZFeatureMap(n_qubits)
    ans = RealAmplitudes(n_qubits, reps=1)
    qc = QuantumCircuit(n_qubits)
    qc.compose(fm, inplace=True)
    qc.compose(ans, inplace=True)
    
    # Decompose into basis gates for Aer execution
    qc = qc.decompose()

    if noise_model is not None:
        backend = AerSimulator(noise_model=noise_model)
        estimator = BackendEstimatorV2(backend=backend)
    else:
        estimator = StatevectorEstimator()

    return EstimatorQNN(
        circuit=qc,
        input_params=fm.parameters,
        weight_params=ans.parameters,
        input_gradients=False,
        estimator=estimator
    )


def run_nisq_noise_benchmark():
    """
    Benchmarks CQ-CNN quantum circuit output fidelity across varying noise scales.
    """
    import torch

    print("=" * 70)
    print("NISQ QUANTUM HARDWARE NOISE SIMULATION BENCHMARK")
    print("=" * 70)

    noise_scales = [
        {"label": "Ideal Statevector (0% Noise)", "p1": 0.0, "p2": 0.0, "ro": 0.0},
        {"label": "Low NISQ Noise (0.1% 1Q, 1% 2Q)", "p1": 0.001, "p2": 0.01, "ro": 0.01},
        {"label": "Medium NISQ Noise (0.5% 1Q, 3% 2Q)", "p1": 0.005, "p2": 0.03, "ro": 0.03},
        {"label": "High NISQ Noise (1.0% 1Q, 5% 2Q)", "p1": 0.010, "p2": 0.05, "ro": 0.05},
        {"label": "Severe NISQ Noise (2.0% 1Q, 10% 2Q)", "p1": 0.020, "p2": 0.10, "ro": 0.10},
    ]

    torch.manual_seed(42)
    fake_features = torch.randn(12, 2).numpy()  # 12 test samples, 2 qubits

    # 1. Ideal Statevector Baseline
    ideal_qnn = create_decomposed_qnn(n_qubits=2, noise_model=None)
    weights = np.random.randn(ideal_qnn.num_weights) * 0.1
    ideal_out = ideal_qnn.forward(fake_features, weights)

    records = []
    for scale in noise_scales:
        label = scale["label"]
        if scale["p1"] == 0.0 and scale["p2"] == 0.0:
            noisy_out = ideal_out
            mad = 0.0
            corr = 1.0
        else:
            nm = build_nisq_noise_model(scale["p1"], scale["p2"], scale["ro"])
            noisy_qnn = create_decomposed_qnn(n_qubits=2, noise_model=nm, shots=1024)
            noisy_out = noisy_qnn.forward(fake_features, weights)
            mad = float(np.mean(np.abs(ideal_out - noisy_out)))
            corr = float(np.corrcoef(ideal_out.flatten(), noisy_out.flatten())[0, 1])

        print(f"[{label}]")
        print(f"  Fidelity Deviation (MAD): {mad:.4f} | Signal Correlation: {corr:.4f}")

        records.append({
            "noise_level": label,
            "p1_error": scale["p1"],
            "p2_cnot_error": scale["p2"],
            "readout_error": scale["ro"],
            "mean_absolute_deviation": round(mad, 4),
            "signal_correlation": round(corr, 4)
        })

    report_df = pd.DataFrame(records)
    out_csv = os.path.join(BASE_DIR, "experiments", "quantum_noise_benchmark_report.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    report_df.to_csv(out_csv, index=False)
    print(f"\nSaved noise simulation report to: {out_csv}")

    # Plot Noise Degradation Curve
    plt.figure(figsize=(9, 5))
    x_labels = [r["noise_level"].split(" (")[0] for r in records]
    plt.plot(range(len(records)), [r["signal_correlation"] for r in records], 'o-', color='#4F46E5', linewidth=2.5, markersize=8, label="Signal Correlation (Fidelity)")
    plt.plot(range(len(records)), [1.0 - r["mean_absolute_deviation"] for r in records], 's--', color='#EF4444', linewidth=2, markersize=6, label="Normalized Robustness (1 - MAD)")
    plt.xticks(range(len(records)), x_labels, rotation=15, ha='right', fontsize=9)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Output Quality Metric", fontsize=11)
    plt.title("CQ-CNN Variational Quantum Circuit Degradation Under NISQ Hardware Noise", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()

    out_plot = os.path.join(BASE_DIR, "plots", "stage5_evaluation", "quantum_noise_resilience_curve.png")
    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print(f"Saved noise degradation plot to: {out_plot}")

    return report_df


if __name__ == "__main__":
    run_nisq_noise_benchmark()
