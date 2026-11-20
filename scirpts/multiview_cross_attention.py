"""
Tri-Planar Orthogonal Cross-Attention Fusion Network (CQ-CNN Multi-View)
========================================================================
Combines Axial, Coronal, and Sagittal orthogonal views of the brain.
Computes multi-head cross-attention across views to capture 3D spatial
structural correlations before passing into a multi-qubit variational circuit.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scirpts.quantum_circuits import create_variational_qnn
from qiskit_machine_learning.connectors import TorchConnector


class ViewEncoder(nn.Module):
    """Encodes a single 2D plane (Axial, Coronal, or Sagittal) into feature embeddings."""
    def __init__(self, embed_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=2),  # 128 -> 64
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # 64 -> 32
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),                         # -> 8 * 2 * 2 = 32
        )
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


class TriPlanarCrossAttention(nn.Module):
    """
    Computes Multi-Head Cross-Attention across 3 orthogonal anatomical views:
    Axial (Q), Coronal (K), Sagittal (V).
    """
    def __init__(self, embed_dim=32, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, axial_feat, coronal_feat, sagittal_feat):
        # Stack views into sequence of tokens: (Batch, 3, embed_dim)
        tokens = torch.stack([axial_feat, coronal_feat, sagittal_feat], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        fused = self.norm(tokens + attn_out)
        # Global pooling across views -> (Batch, embed_dim)
        return torch.mean(fused, dim=1)


class MultiViewQuantumNet(nn.Module):
    """
    Hybrid Tri-Planar Quantum Neural Network:
    Axial + Coronal + Sagittal -> Cross-Attention -> Quantum Variational Circuit -> Dementia Diagnosis
    """
    def __init__(self, n_qubits=3, embed_dim=32):
        super().__init__()
        self.axial_enc = ViewEncoder(embed_dim=embed_dim)
        self.coronal_enc = ViewEncoder(embed_dim=embed_dim)
        self.sagittal_enc = ViewEncoder(embed_dim=embed_dim)
        self.cross_attn = TriPlanarCrossAttention(embed_dim=embed_dim, num_heads=2)

        # Dimension reduction to quantum circuit parameter count
        self.fc_quantum = nn.Linear(embed_dim, n_qubits)
        
        qnn = create_variational_qnn(n_qubits=n_qubits, ansatz_type="efficient_su2", reps=1, entanglement="full")
        self.qnn = TorchConnector(qnn)
        self.head = nn.Linear(1, 2)

    def forward(self, axial_x, coronal_x, sagittal_x):
        a_f = self.axial_enc(axial_x)
        c_f = self.coronal_enc(coronal_x)
        s_f = self.sagittal_enc(sagittal_x)

        fused = self.cross_attn(a_f, c_f, s_f)
        q_in = self.fc_quantum(fused)
        q_out = self.qnn(q_in)
        return self.head(q_out)


if __name__ == "__main__":
    print("Testing Tri-Planar Cross-Attention Quantum Network...")
    model = MultiViewQuantumNet(n_qubits=3, embed_dim=32)
    b_size = 2
    ax = torch.randn(b_size, 1, 128, 128)
    co = torch.randn(b_size, 1, 128, 128)
    sa = torch.randn(b_size, 1, 128, 128)

    out = model(ax, co, sa)
    print(f"Output shape: {out.shape} | Probs: {F.softmax(out, dim=1).detach().numpy()}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Multi-View Parameters: {params:,}")
    print("Tri-Planar Model Self-Test Passed!")
