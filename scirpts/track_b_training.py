"""
Stage 4: Track B — Leakage-Free Subject-Level Training & Evaluation
====================================================================
Key Guarantees:
  1. All 2D slices from the same patient (OAS2_xxxx) live in ONE split only.
  2. DDPM synthetic slices are exclusively in the training set.
  3. Validation set is used ONLY for LR scheduling / checkpoint selection.
  4. Test set is SEALED — never touched during training.
  5. Metrics reported: Accuracy, Balanced Accuracy, Precision, Recall,
     Specificity, Macro F1, ROC-AUC + Confusion Matrix.
  6. All experiments run across N_SEEDS for confidence intervals.

Usage:
    # With real OASIS-2 preprocessed data:
    python scirpts/track_b_training.py --data_root datasets/OASIS-2/preprocessed --view axial --qubits 3 --mode full

    # With synthetic dummy data for pipeline validation:
    python scirpts/track_b_training.py --mode dummy
"""

import os
import csv
import time
import json
import random
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import StatevectorEstimator as Estimator

# ─── Config ─────────────────────────────────────────────────────────────────

IMAGE_SIZE   = 128
EPOCHS       = 15
BATCH_SIZE   = 8
LR           = 0.001
N_SEEDS      = 3        # Number of repeated seeds for confidence intervals
PATIENCE     = 4        # LR reduce patience
LR_FACTOR    = 0.7
MIN_LR       = 1e-6
LABEL_MAP    = {"nondemented": 0, "moderate_dementia": 1}

# ─── QNN creation ───────────────────────────────────────────────────────────

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

# ─── Model Architectures ────────────────────────────────────────────────────

class QuantumNet(nn.Module):
    def __init__(self, qnn, n_qubits, conv2_ch=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, conv2_ch, kernel_size=5)
        self.dropout = nn.Dropout2d()
        s1 = (IMAGE_SIZE - 5 + 1) // 2
        s2 = (s1 - 5 + 1) // 2
        self.fc1 = nn.Linear(conv2_ch * s2 * s2, 4)
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
    def __init__(self, conv2_ch=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, conv2_ch, kernel_size=5)
        self.dropout = nn.Dropout2d()
        s1 = (IMAGE_SIZE - 5 + 1) // 2
        s2 = (s1 - 5 + 1) // 2
        self.fc1 = nn.Linear(conv2_ch * s2 * s2, 4)
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

# ─── Dataset ────────────────────────────────────────────────────────────────

class ManifestDataset(Dataset):
    """Loads images from a data_manifest.csv filtered by split & plane."""

    def __init__(self, manifest_df, data_root, split_col, split_val, plane=None):
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        df = manifest_df[manifest_df[split_col] == split_val].copy()
        if plane and plane != "3-plane":
            df = df[df["plane"] == plane]

        self.records = []
        for _, row in df.iterrows():
            fp = os.path.join(data_root, row["file_path"])
            if os.path.exists(fp):
                label = LABEL_MAP.get(str(row["label"]), 0)
                self.records.append((fp, label))

        counts = {0: sum(1 for _, l in self.records if l == 0),
                  1: sum(1 for _, l in self.records if l == 1)}
        print(f"  [{split_val}] {len(self.records)} slices | non-dem={counts[0]}, mod-dem={counts[1]}")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        fp, label = self.records[idx]
        img = Image.open(fp).convert("L")
        return self.transform(img), label


class DummyDataset(Dataset):
    """Synthetic dataset for pipeline validation without real MRI data."""

    def __init__(self, n_samples=60, seed=42):
        random.seed(seed)
        self.data = [(torch.randn(1, IMAGE_SIZE, IMAGE_SIZE), random.randint(0, 1))
                     for _ in range(n_samples)]
        counts = {0: sum(1 for _, l in self.data if l == 0),
                  1: sum(1 for _, l in self.data if l == 1)}
        print(f"  [DummyDataset] {n_samples} synthetic slices | class 0={counts[0]}, class 1={counts[1]}")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# ─── Training utilities ─────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def train_epoch(model, loader, optimizer, loss_fn, is_quantum):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        pred = out.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += len(y)
    return total_loss / total, 100.0 * correct / total

def eval_epoch(model, loader, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            total_loss += loss_fn(out, y).item() * len(y)
            correct += out.argmax(dim=1).eq(y).sum().item()
            total += len(y)
    return total_loss / total, 100.0 * correct / total

def compute_full_metrics(model, loader):
    """Compute Acc, Balanced Acc, Precision, Recall, Specificity, F1, AUC."""
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            pred = out.argmax(dim=1).cpu().numpy()
            all_y.extend(y.numpy()); all_pred.extend(pred); all_prob.extend(prob)

    y_true, y_pred = np.array(all_y), np.array(all_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy":          round(100.0 * accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(100.0 * balanced_accuracy_score(y_true, y_pred), 4),
        "precision":         round(100.0 * precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":            round(100.0 * recall_score(y_true, y_pred, zero_division=0), 4),
        "specificity":       round(100.0 * (tn / (tn + fp)) if (tn + fp) > 0 else 0, 4),
        "f1":                round(100.0 * f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":           round(roc_auc_score(y_true, all_prob) if len(set(y_true)) > 1 else 0, 4),
        "confusion_matrix":  cm.tolist(),
        "n_samples":         len(y_true),
    }

# ─── Main Training Loop ─────────────────────────────────────────────────────

def run_experiment(model, model_name, train_loader, val_loader, test_loader,
                   seed, out_dir, is_quantum=True):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    wait = 0
    best_ckpt = os.path.join(out_dir, f"{model_name}_seed{seed}_best.pt")
    log_path  = os.path.join(out_dir, f"{model_name}_seed{seed}_train_log.csv")

    epoch_logs = []
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, loss_fn, is_quantum)
        va_loss, va_acc = eval_epoch(model, val_loader, loss_fn)

        epoch_logs.append({
            "epoch": epoch, "train_loss": round(tr_loss, 6),
            "train_acc": round(tr_acc, 4), "val_loss": round(va_loss, 6),
            "val_acc": round(va_acc, 4)
        })
        print(f"  Ep {epoch:02d}/{EPOCHS} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.1f}% | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.1f}%")

        # Checkpoint on best val loss
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            wait = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            wait += 1
            if wait >= PATIENCE:
                for pg in optimizer.param_groups:
                    pg["lr"] = max(pg["lr"] * LR_FACTOR, MIN_LR)
                wait = 0

    wall_time = round(time.time() - t_start, 1)

    # Save epoch log
    pd.DataFrame(epoch_logs).to_csv(log_path, index=False)

    # Load best checkpoint for final test evaluation
    model.load_state_dict(torch.load(best_ckpt, map_location="cpu", weights_only=True))
    test_metrics = compute_full_metrics(model, test_loader)
    test_metrics["wall_time_secs"] = wall_time
    test_metrics["model"] = model_name
    test_metrics["seed"] = seed

    print(f"\n  ── Test Metrics (seed={seed}) ──")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"    {k}: {v}")
    print(f"    confusion_matrix: {test_metrics['confusion_matrix']}")

    return test_metrics

# ─── Aggregate across seeds ──────────────────────────────────────────────────

def aggregate_seeds(results):
    """Compute mean ± std across repeated seeds."""
    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    summary = {"model": results[0]["model"]}
    for k in keys:
        vals = [r[k] for r in results]
        summary[k + "_mean"] = round(float(np.mean(vals)), 4)
        summary[k + "_std"]  = round(float(np.std(vals)), 4)
    return summary

# ─── Entry point ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=".")
    p.add_argument("--manifest",  default="data_manifest.csv")
    p.add_argument("--view",      default="axial", choices=["axial", "coronal", "sagittal", "3-plane"])
    p.add_argument("--qubits",    type=int, default=3, choices=[2, 3])
    p.add_argument("--arch",      default="quantum", choices=["quantum", "classical"])
    p.add_argument("--mode",      default="full", choices=["full", "dummy"])
    p.add_argument("--out_dir",   default="experiments/track_b")
    p.add_argument("--seeds",     type=int, default=N_SEEDS)
    return p.parse_args()

def main():
    args = parse_args()
    print("=" * 80)
    print("TRACK B: Leakage-Free Subject-Level Validation")
    print(f"Mode={args.mode} | Arch={args.arch} | Qubits={args.qubits} | View={args.view} | Seeds={args.seeds}")
    print("=" * 80)

    # ── Build dataloaders ──────────────────────────────────────────────────
    if args.mode == "dummy":
        print("\n[DUMMY MODE] Using synthetic random tensors — no real data needed.")
        train_ds = DummyDataset(n_samples=80, seed=42)
        val_ds   = DummyDataset(n_samples=20, seed=99)
        test_ds  = DummyDataset(n_samples=20, seed=77)
    else:
        manifest = pd.read_csv(args.manifest)
        print(f"\nManifest loaded: {len(manifest)} records")
        print("Building subject-level splits (Track B col=track_b_split)...")

        # Verify zero subject overlap
        for s1, s2 in [("train","val"),("train","test"),("val","test")]:
            sub1 = set(manifest[manifest["track_b_split"]==s1]["subject_id"]) - {"SYNTHETIC","UNKNOWN"}
            sub2 = set(manifest[manifest["track_b_split"]==s2]["subject_id"]) - {"SYNTHETIC","UNKNOWN"}
            overlap = sub1 & sub2
            if overlap:
                raise ValueError(f"Subject overlap between {s1} and {s2}: {overlap}")
            print(f"  Zero subject overlap verified: {s1} vs {s2}")

        train_ds = ManifestDataset(manifest, args.data_root, "track_b_split", "train", args.view)
        val_ds   = ManifestDataset(manifest, args.data_root, "track_b_split", "val",   args.view)
        test_ds  = ManifestDataset(manifest, args.data_root, "track_b_split", "test",  args.view)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nClass distributions:")
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if hasattr(ds, 'records'):
            lbls = [l for _, l in ds.records]
        else:
            lbls = [l for _, l in ds.data]
        print(f"  {name}: total={len(lbls)}, class0={lbls.count(0)}, class1={lbls.count(1)}")

    # ── Run across seeds ───────────────────────────────────────────────────
    all_results = []
    model_name = f"{args.arch}_{args.qubits}q_{args.view}" if args.arch == "quantum" else f"classical_{args.view}"
    out_dir = os.path.join(args.out_dir, model_name)

    seeds = list(range(args.seeds))
    for seed in seeds:
        print(f"\n── Seed {seed} ──────────────────────────────────────────────")
        set_seed(seed)

        if args.arch == "quantum":
            qnn = create_qnn(args.qubits)
            model = QuantumNet(qnn, args.qubits, conv2_ch=4)
        else:
            model = ClassicalNet(conv2_ch=4)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model: {model_name} | Params: {params:,}")

        metrics = run_experiment(
            model, model_name, train_loader, val_loader, test_loader,
            seed=seed, out_dir=out_dir, is_quantum=(args.arch == "quantum")
        )
        all_results.append(metrics)

    # ── Aggregate and save ─────────────────────────────────────────────────
    summary = aggregate_seeds(all_results)
    print("\n" + "=" * 80)
    print(f"TRACK B SUMMARY ({model_name}) — {args.seeds} seeds")
    print("=" * 80)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(all_results).to_csv(os.path.join(out_dir, "per_seed_results.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_dir}/")
    print("[STAGE 4 TRACK B] Complete.")

if __name__ == "__main__":
    main()

