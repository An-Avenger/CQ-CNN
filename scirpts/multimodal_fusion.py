"""
Multimodal Clinical-Quantum Integration Network
===============================================
Fuses high-dimensional MRI quantum representations with OASIS-2 tabular
clinical biomarkers (Age, MMSE cognitive score, Education, SES, eTIV, nWBV).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scirpts.quantum_circuits import create_variational_qnn
from qiskit_machine_learning.connectors import TorchConnector


class TabularClinicalEncoder(nn.Module):
    """
    Encodes 6 key clinical biomarkers:
    Age, EDUC (Years of Education), SES (Socioeconomic Status),
    MMSE (Mini-Mental State Exam), eTIV (Intracranial Volume), nWBV (Normalized Brain Volume).
    """
    def __init__(self, in_features=6, embed_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class MultimodalCQCNN(nn.Module):
    """
    Multimodal Decision Fusion Architecture:
    MRI Slices -> Conv2D -> PQC Quantum Circuit (Quantum Features)
    Clinical Biomarkers -> Tabular MLP (Clinical Features)
    Late Fusion Layer -> Calibrated Dementia Risk
    """
    def __init__(self, n_qubits=2, tabular_dim=6, fused_dim=16):
        super().__init__()
        # 1. Vision + Quantum Branch
        self.conv = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 4, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        s = ((128 - 5 + 1) // 2 - 5 + 1) // 2
        self.img_fc = nn.Linear(4 * s * s, n_qubits)

        qnn = create_variational_qnn(n_qubits=n_qubits, ansatz_type="real_amplitudes", reps=1)
        self.qnn = TorchConnector(qnn)

        # 2. Tabular Clinical Branch
        self.clin_encoder = TabularClinicalEncoder(in_features=tabular_dim, embed_dim=fused_dim)

        # 3. Late Fusion Multi-Modal Head
        # Quantum output is 1-dim expectation value + fused clinical features
        self.fusion_head = nn.Sequential(
            nn.Linear(1 + fused_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2)
        )

    def forward(self, img_x, clin_x):
        # Vision branch
        feat = self.conv(img_x)
        feat = feat.view(feat.size(0), -1)
        q_in = self.img_fc(feat)
        q_out = self.qnn(q_in)  # (Batch, 1)

        # Tabular branch
        c_out = self.clin_encoder(clin_x)  # (Batch, fused_dim)

        # Concatenate multi-modal tokens
        fused = torch.cat([q_out, c_out], dim=-1)
        return self.fusion_head(fused)


def load_clinical_metadata_lookup():
    """
    Builds a clean normalized lookup dictionary for OASIS-2 subjects.
    Features: [Age, EDUC, SES, MMSE, eTIV, nWBV]
    """
    csv_path = os.path.join(BASE_DIR, "datasets", "OASIS-2", "downloads", "oasis_longitudinal.csv")
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    # Impute missing values with medians
    df["SES"] = df["SES"].fillna(df["SES"].median())
    df["MMSE"] = df["MMSE"].fillna(df["MMSE"].median())

    lookup = {}
    for _, row in df.iterrows():
        sub_id = row["Subject ID"]
        feat = [
            float(row["Age"]) / 100.0,
            float(row["EDUC"]) / 25.0,
            float(row["SES"]) / 5.0,
            float(row["MMSE"]) / 30.0,
            float(row["eTIV"]) / 2000.0,
            float(row["nWBV"])
        ]
        lookup[sub_id] = np.array(feat, dtype=np.float32)

    return lookup


if __name__ == "__main__":
    print("Testing Multimodal Clinical-Quantum Integration Network...")
    model = MultimodalCQCNN(n_qubits=2, tabular_dim=6)
    b_size = 2
    fake_img = torch.randn(b_size, 1, 128, 128)
    fake_clin = torch.randn(b_size, 6)

    out = model(fake_img, fake_clin)
    print(f"Output shape: {out.shape} | Softmax probs: {F.softmax(out, dim=1).detach().numpy()}")
    lookup = load_clinical_metadata_lookup()
    print(f"Loaded clinical biomarkers for {len(lookup)} OASIS-2 subjects successfully!")
    print("Multimodal Self-Test Passed!")
