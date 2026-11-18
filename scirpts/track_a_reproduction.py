"""
Stage 3: Track A — Original Reproduction Evaluation
=====================================================
Loads ALL pre-trained classification checkpoints, verifies architecture 
compatibility, counts parameters, reads training logs, and reports
metrics vs. PLOS ONE published values (Table 1 & Table 2).

Run from E:/PROJECTS/CQCNN:
    .venv/Scripts/python.exe scirpts/track_a_reproduction.py
"""

import os
import csv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import StatevectorEstimator as Estimator
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─── Architecture definitions ───────────────────────────────────────────────

def create_qnn(n_qubits):
    feature_map = ZZFeatureMap(n_qubits)
    ansatz = RealAmplitudes(n_qubits, reps=1)
    qc = QuantumCircuit(n_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    return EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
        estimator=Estimator()
    )

class QuantumNet(nn.Module):
    def __init__(self, qnn, image_size, n_qubits, conv2_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, conv2_channels, kernel_size=5)
        self.dropout = nn.Dropout2d()
        s1 = (image_size - 5 + 1) // 2
        s2 = (s1 - 5 + 1) // 2
        self.fc1 = nn.Linear(conv2_channels * s2 * s2, 4)
        self.fc2 = nn.Linear(4, n_qubits)
        self.qnn = TorchConnector(qnn)
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)
        x = self.fc3(x)
        return cat((x, 1 - x), -1)

class ClassicalNet(nn.Module):
    def __init__(self, image_size, conv2_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, conv2_channels, kernel_size=5)
        self.dropout = nn.Dropout2d()
        s1 = (image_size - 5 + 1) // 2
        s2 = (s1 - 5 + 1) // 2
        self.fc1 = nn.Linear(conv2_channels * s2 * s2, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class TinyAlexNet(nn.Module):
    def __init__(self, num_classes=2, in_channels=1, image_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        s = image_size // 8
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8 * s * s, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

# ─── Helper: count params ────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ─── Published paper metrics (PLOS ONE Table 1 & Table 2) ───────────────────
# Format: (Precision%, F1%, Specificity%, Accuracy%)

PAPER_METRICS = {
    # 2-Qubit models (alpha)
    "2q_axial":          (87.77, 93.45, 98.43, 98.40),
    "2q_axial-ss":       (89.55, 93.42, 98.53, 98.51),
    "2q_coronal":        (94.71, 97.27, 99.30, 99.37),
    "2q_coronal-ss":     (69.31, 81.43, 94.05, 94.71),
    "2q_sagittal":       (95.69, 97.75, 99.41, 99.48),
    "2q_sagittal-ss":    (70.68, 80.88, 94.04, 93.70),
    "2q_3-plane":        (90.34, 92.46, 98.61, 98.23),
    "2q_3-plane-ss":     (67.21, 75.27, 95.70, 93.50),
    # 3-Qubit models (beta)
    "3q_axial":          (90.69, 95.12, 98.72, 98.86),
    "3q_axial-ss":       (78.83, 87.40, 96.86, 97.01),
    "3q_coronal":        (91.86, 95.75, 98.95, 98.96),
    "3q_coronal-ss":     (71.96, 81.23, 92.83, 93.62),
    "3q_sagittal":       (94.92, 90.94, 99.29, 98.11),
    "3q_sagittal-ss":    (78.14, 86.76, 95.75, 96.22),
    "3q_3-plane":        (90.23, 94.85, 98.64, 98.79),
    "3q_3-plane-ss":     (83.19, 89.45, 97.55, 97.50),
}

# ─── Checkpoint registry ─────────────────────────────────────────────────────

CHECKPOINTS = [
    # (label, arch_type, n_qubits, conv2_ch, pt_path, csv_path, paper_key)
    # 2-Qubit
    ("α1  2Q Axial",           "quantum", 2, 8, "models/classification/2qubits/oasis-2_axial_2qubits.pt",        "models/classification/2qubits/oasis-2_axial_2qubits.csv",        "2q_axial"),
    ("α2  2Q Axial-SS",        "quantum", 2, 8, "models/classification/2qubits/oasis-2_axial-ss_2qubits.pt",     "models/classification/2qubits/oasis-2_axial-ss_2qubits.csv",     "2q_axial-ss"),
    ("α3  2Q Coronal",         "quantum", 2, 4, "models/classification/2qubits/oasis-2_coronal_2qubits.pt",      "models/classification/2qubits/oasis-2_coronal_2qubits.csv",      "2q_coronal"),
    ("α4  2Q Coronal-SS",      "quantum", 2, 4, "models/classification/2qubits/oasis-2_coronal-ss_2qubits.pt",   "models/classification/2qubits/oasis-2_coronal-ss_2qubits.csv",   "2q_coronal-ss"),
    ("α5  2Q Sagittal",        "quantum", 2, 4, "models/classification/2qubits/oasis-2_sagittal_2qubits.pt",     "models/classification/2qubits/oasis-2_sagittal_2qubits.csv",     "2q_sagittal"),
    ("α6  2Q Sagittal-SS",     "quantum", 2, 4, "models/classification/2qubits/oasis-2_sagittal-ss_2qubits.pt",  "models/classification/2qubits/oasis-2_sagittal-ss_2qubits.csv",  "2q_sagittal-ss"),
    ("α7  2Q 3-Plane",         "quantum", 2, 4, "models/classification/2qubits/oasis-2_3-plane_2qubits.pt",      "models/classification/2qubits/oasis-2_3-plane_2qubits.csv",      "2q_3-plane"),
    ("α8  2Q 3-Plane-SS",      "quantum", 2, 4, "models/classification/2qubits/oasis-2_3-plane-ss_2qubits.pt",   "models/classification/2qubits/oasis-2_3-plane-ss_2qubits.csv",   "2q_3-plane-ss"),
    # 3-Qubit
    ("β1  3Q Axial",           "quantum", 3, 4, "models/classification/3qubits/oasis-2_axial_3qubits.pt",        "models/classification/3qubits/oasis-2_axial_3qubits.csv",        "3q_axial"),
    ("β2  3Q Axial-SS",        "quantum", 3, 8, "models/classification/3qubits/oasis-2_axial-ss_3qubits.pt",     "models/classification/3qubits/oasis-2_axial-ss_3qubits.csv",     "3q_axial-ss"),
    ("β3  3Q Coronal",         "quantum", 3, 4, "models/classification/3qubits/oasis-2_coronal_3qubits.pt",      "models/classification/3qubits/oasis-2_coronal_3qubits.csv",      "3q_coronal"),
    ("β4  3Q Coronal-SS",      "quantum", 3, 4, "models/classification/3qubits/oasis-2_coronal-ss_3qubits.pt",   "models/classification/3qubits/oasis-2_coronal-ss_3qubits.csv",   "3q_coronal-ss"),
    ("β5  3Q Sagittal",        "quantum", 3, 4, "models/classification/3qubits/oasis-2_sagittal_3qubits.pt",     "models/classification/3qubits/oasis-2_sagittal_3qubits.csv",     "3q_sagittal"),
    ("β6  3Q Sagittal-SS",     "quantum", 3, 4, "models/classification/3qubits/oasis-2_sagittal-ss_3qubits.pt",  "models/classification/3qubits/oasis-2_sagittal-ss_3qubits.csv",  "3q_sagittal-ss"),
    ("β7  3Q 3-Plane",         "quantum", 3, 8, "models/classification/3qubits/oasis-2_3-plane_3qubits.pt",      "models/classification/3qubits/oasis-2_3-plane_3qubits.csv",      "3q_3-plane"),
    ("β8  3Q 3-Plane-SS ★",   "quantum", 3, 8, "models/classification/3qubits/oasis-2_3-plane-ss_3qubits.pt",   "models/classification/3qubits/oasis-2_3-plane-ss_3qubits.csv",   "3q_3-plane-ss"),
    # Classical baselines
    ("γ1  Classical Axial",    "classical", None, 4, "models/classification/classical/oasis-2_axial.pt",         "models/classification/classical/oasis-2_axial.csv",         None),
    ("γ2  Classical Axial-SS", "classical", None, 4, "models/classification/classical/oasis-2_axial-ss.pt",      "models/classification/classical/oasis-2_axial-ss.csv",      None),
    ("γ3  Classical Coronal",  "classical", None, 4, "models/classification/classical/oasis-2_coronal.pt",       "models/classification/classical/oasis-2_coronal.csv",       None),
    ("γ4  Classical Coronal-SS","classical",None, 4, "models/classification/classical/oasis-2_coronal-ss.pt",    "models/classification/classical/oasis-2_coronal-ss.csv",    None),
    ("γ5  Classical Sagittal", "classical", None, 4, "models/classification/classical/oasis-2_sagittal.pt",      "models/classification/classical/oasis-2_sagittal.csv",      None),
    ("γ6  Classical Sagittal-SS","classical",None,4, "models/classification/classical/oasis-2_sagittal-ss.pt",   "models/classification/classical/oasis-2_sagittal-ss.csv",   None),
    ("γ7  Classical 3-Plane",  "classical", None, 4, "models/classification/classical/oasis-2_3-plane.pt",       "models/classification/classical/oasis-2_3-plane.csv",       None),
    ("γ8  Classical 3-Plane-SS","classical",None,4,  "models/classification/classical/oasis-2_3-plane-ss.pt",    "models/classification/classical/oasis-2_3-plane-ss.csv",    None),
    # TinyAlexNet SOTA
    ("SOTA TinyAlexNet 3-Plane-SS","alexnet", None, None, "models/classification/classical-sota/oasis-2_3-plane-ss-alexnet.pt", "models/classification/classical-sota/oasis-2_3-plane-ss-alexnet.csv", None),
]

# ─── Main evaluation loop ────────────────────────────────────────────────────

def run_track_a():
    print("=" * 90)
    print("TRACK A: Original Reproduction — Checkpoint Audit & Training Log Analysis")
    print("=" * 90)

    results = []

    # Pre-build QNN instances to avoid re-creating per checkpoint
    qnn_cache = {}
    def get_qnn(n):
        if n not in qnn_cache:
            qnn_cache[n] = create_qnn(n)
        return qnn_cache[n]

    for (label, arch_type, n_qubits, conv2_ch, pt_path, csv_path, paper_key) in CHECKPOINTS:
        entry = {"label": label, "arch": arch_type}
        
        # 1. Checkpoint existence
        if not os.path.exists(pt_path):
            entry.update({"status": "MISSING .pt", "error": pt_path})
            results.append(entry)
            print(f"[MISSING]  {label}: {pt_path}")
            continue
        
        # 2. Load state dict
        try:
            sd = torch.load(pt_path, map_location="cpu", weights_only=True)
        except Exception as e:
            entry.update({"status": "LOAD ERROR", "error": str(e)})
            results.append(entry)
            print(f"[ERROR]    {label}: {e}")
            continue

        # 3. Build model and load weights
        try:
            if arch_type == "quantum":
                qnn = get_qnn(n_qubits)
                model = QuantumNet(qnn, 128, n_qubits, conv2_channels=conv2_ch)
            elif arch_type == "classical":
                model = ClassicalNet(128, conv2_channels=conv2_ch)
            elif arch_type == "alexnet":
                model = TinyAlexNet()
            
            model.load_state_dict(sd)
            model.eval()
            params = count_params(model)
            size_kb = os.path.getsize(pt_path) / 1024

            # 4. Micro-batch smoke test (no NaN check)
            dummy = torch.randn(2, 1, 128, 128)
            with torch.no_grad():
                out = model(dummy)
            assert not torch.isnan(out).any(), "NaN in output"
            status = "OK"
        except Exception as e:
            entry.update({"status": "ARCH ERROR", "error": str(e), "params": "?", "size_kb": "?"})
            results.append(entry)
            print(f"[ARCH ERR] {label}: {e}")
            continue

        entry.update({"status": status, "params": params, "size_kb": round(size_kb, 1)})

        # 5. Read training log CSV
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                best_val_acc = df["val_accuracy"].max()
                last_val_acc = df["val_accuracy"].iloc[-1]
                epochs_trained = len(df)
                entry.update({
                    "epochs": epochs_trained,
                    "best_val_acc": round(best_val_acc, 2),
                    "last_val_acc": round(last_val_acc, 2)
                })
            except Exception as e:
                entry.update({"epochs": "?", "best_val_acc": "?", "last_val_acc": "?"})
        else:
            entry.update({"epochs": "?", "best_val_acc": "N/A", "last_val_acc": "N/A"})

        # 6. Compare with published paper metrics
        if paper_key and paper_key in PAPER_METRICS:
            pre, f1, spec, acc = PAPER_METRICS[paper_key]
            entry.update({"paper_acc": acc, "paper_f1": f1, "paper_prec": pre, "paper_spec": spec})
        else:
            entry.update({"paper_acc": "N/A", "paper_f1": "N/A", "paper_prec": "N/A", "paper_spec": "N/A"})

        results.append(entry)

    # ─── Print summary table ─────────────────────────────────────────────────
    print()
    print(f"{'Model':<28} {'Arch':<10} {'Params':>7} {'SizeKB':>7} {'Epochs':>6} {'BestValAcc':>10} {'PaperAcc':>10} {'PaperF1':>8} {'Status':<8}")
    print("-" * 100)
    for r in results:
        print(f"{r['label']:<28} {r['arch']:<10} {str(r.get('params','?')):>7} "
              f"{str(r.get('size_kb','?')):>7} {str(r.get('epochs','?')):>6} "
              f"{str(r.get('best_val_acc','?')):>10} {str(r.get('paper_acc','N/A')):>10} "
              f"{str(r.get('paper_f1','N/A')):>8} {r.get('status','?'):<8}")

    # ─── Save to CSV ─────────────────────────────────────────────────────────
    out_path = "experiments/track_a_reproduction_report.csv"
    os.makedirs("experiments", exist_ok=True)
    keys = ["label", "arch", "params", "size_kb", "epochs",
            "best_val_acc", "last_val_acc",
            "paper_acc", "paper_f1", "paper_prec", "paper_spec", "status"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nFull report saved → {out_path}")
    print("\n[IMPLEMENTED ✓] [EXECUTED ✓] Awaiting validation...")

if __name__ == "__main__":
    run_track_a()

