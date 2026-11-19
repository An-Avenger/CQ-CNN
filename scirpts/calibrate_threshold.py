"""
Optimal Decision Threshold Calibration for Track B Models
==========================================================
Under extreme class imbalance (~120:1), the default classification threshold (0.5)
leads to trivial majority-class prediction.
This script:
  1. Computes raw validation set prediction probabilities.
  2. Finds the optimal classification threshold using:
     - Youden's J statistic (Sensitivity + Specificity - 1)
     - F1-score maximization
  3. Evaluates the calibrated threshold on the sealed unseen-patient test set.
  4. Generates an ROC and Precision-Recall curve with the optimal operating point.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score
)
from torch.utils.data import DataLoader

from scirpts.track_b_training import ClassicalNet, ManifestDataset, BATCH_SIZE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE_DIR, "experiments")
PLOTS_DIR = os.path.join(BASE_DIR, "plots", "stage5_evaluation")
os.makedirs(PLOTS_DIR, exist_ok=True)

def evaluate_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(acc) * 100, 2),
        "balanced_acc": round(float(bal_acc) * 100, 2),
        "sensitivity_recall": round(float(rec) * 100, 2),
        "specificity": round(float(spec) * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "f1": round(float(f1) * 100, 2),
        "confusion_matrix": cm.tolist()
    }

def get_predictions(model, loader):
    model.eval()
    all_y, all_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_y.extend(y.numpy())
            all_prob.extend(prob)
    return np.array(all_y), np.array(all_prob)

def main():
    print("=" * 70)
    print("OPTIMAL THRESHOLD CALIBRATION FOR TRACK B (CLASSICAL AXIAL)")
    print("=" * 70)

    manifest_path = os.path.join(BASE_DIR, "data_manifest.csv")
    manifest = pd.read_csv(manifest_path)

    val_ds  = ManifestDataset(manifest, BASE_DIR, "track_b_split", "val",  "axial")
    test_ds = ManifestDataset(manifest, BASE_DIR, "track_b_split", "test", "axial")

    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ClassicalNet(conv2_ch=4)
    ckpt_path = os.path.join(EXP_DIR, "track_b", "classical_axial", "classical_axial_seed0_best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    print(f"Loaded trained checkpoint: {ckpt_path}")

    # 1. Validation Set Predictions & Calibration
    y_val, p_val = get_predictions(model, val_loader)
    fpr, tpr, roc_thresh = roc_curve(y_val, p_val)
    
    # Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    optimal_thresh_j = roc_thresh[best_j_idx]
    
    # F1 Optimization
    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = [f1_score(y_val, (p_val >= t).astype(int), zero_division=0) for t in thresholds]
    optimal_thresh_f1 = thresholds[np.argmax(f1_scores)]

    print(f"\n[Validation Tuning] Youden's J Optimal Threshold: {optimal_thresh_j:.4f}")
    print(f"[Validation Tuning] Max F1 Optimal Threshold:      {optimal_thresh_f1:.4f}")

    # 2. Evaluate on Sealed Test Set (Unseen Patients)
    y_test, p_test = get_predictions(model, test_loader)
    
    res_default = evaluate_threshold(y_test, p_test, 0.50)
    res_j       = evaluate_threshold(y_test, p_test, optimal_thresh_j)
    res_f1      = evaluate_threshold(y_test, p_test, optimal_thresh_f1)

    print("\n--- Test Set Performance on 15 Unseen Patients ---")
    print(f"1. Default Threshold (0.50):")
    print(f"   Accuracy: {res_default['accuracy']}% | Balanced Acc: {res_default['balanced_acc']}% | Sensitivity: {res_default['sensitivity_recall']}% | Specificity: {res_default['specificity']}% | F1: {res_default['f1']}%")
    print(f"   CM: {res_default['confusion_matrix']}")

    print(f"\n2. Youden's J Calibrated ({optimal_thresh_j:.4f}):")
    print(f"   Accuracy: {res_j['accuracy']}% | Balanced Acc: {res_j['balanced_acc']}% | Sensitivity: {res_j['sensitivity_recall']}% | Specificity: {res_j['specificity']}% | F1: {res_j['f1']}%")
    print(f"   CM: {res_j['confusion_matrix']}")

    print(f"\n3. F1-Optimal Calibrated ({optimal_thresh_f1:.4f}):")
    print(f"   Accuracy: {res_f1['accuracy']}% | Balanced Acc: {res_f1['balanced_acc']}% | Sensitivity: {res_f1['sensitivity_recall']}% | Specificity: {res_f1['specificity']}% | F1: {res_f1['f1']}%")
    print(f"   CM: {res_f1['confusion_matrix']}")

    # 3. Plot ROC Curve with Operating Point
    plt.figure(figsize=(7, 6))
    fpr_test, tpr_test, _ = roc_curve(y_test, p_test)
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    
    # Plot Youden's point on test
    y_pred_j = (p_test >= optimal_thresh_j).astype(int)
    cm_j = confusion_matrix(y_test, y_pred_j)
    test_tpr_j = cm_j[1, 1] / (cm_j[1, 0] + cm_j[1, 1])
    test_fpr_j = cm_j[0, 1] / (cm_j[0, 0] + cm_j[0, 1])
    plt.scatter([test_fpr_j], [test_tpr_j], color='red', s=80, zorder=5, label=f"Operating Point (Thresh={optimal_thresh_j:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Track B Test ROC Curve with Calibrated Operating Point')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    out_plot = os.path.join(PLOTS_DIR, "track_b_calibrated_roc_curve.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"\nSaved Calibrated ROC Curve: {out_plot}")

if __name__ == "__main__":
    main()
