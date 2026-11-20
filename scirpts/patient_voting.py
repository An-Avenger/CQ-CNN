"""
Patient-Level Volumetric Aggregation for CQ-CNN
===============================================
Aggregates slice-level predictions across full 3D MRI volumes to produce
a single unified clinical diagnosis per patient:
  - Mean-probability voting
  - Majority threshold voting
  - Max-pooling aggregation
  - Top-K highest risk slice pooling
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scirpts.track_b_training import ClassicalNet, IMAGE_SIZE, LABEL_MAP


def run_patient_aggregation(
    manifest_path="data_manifest_sample.csv",
    checkpoint_path="experiments/track_b/classical_axial/classical_axial_seed0_best.pt",
    plane="axial",
    threshold=0.0005,
    device="cpu"
):
    """
    Evaluates slice-level probabilities across the unseen test subjects
    and aggregates them into patient-level predictions.
    """
    print(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    test_df = df[df["track_b_split"] == "test"].copy()
    if plane and plane != "3-plane":
        test_df = test_df[test_df["plane"] == plane]

    print(f"Unseen test slices matching criteria: {len(test_df)} across {test_df['subject_id'].nunique()} subjects")

    # Load model
    model = ClassicalNet(conv2_ch=4)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"[Warning] Checkpoint not found at {checkpoint_path}, using initialized weights.")
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Run inference per slice
    results = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            fp = os.path.join(BASE_DIR, row["file_path"])
            if not os.path.exists(fp):
                continue
            img = Image.open(fp).convert("L")
            tensor = transform(img).unsqueeze(0).to(device)
            out = model(tensor)
            prob_mod = float(F.softmax(out, dim=1)[0, 1].cpu().item())
            
            results.append({
                "subject_id": row["subject_id"],
                "slice_id": row["slice_id"],
                "plane": row["plane"],
                "true_label_str": row["label"],
                "true_label": 1 if row["label"] == "moderate_dementia" else 0,
                "prob_mod": prob_mod
            })

    pred_df = pd.DataFrame(results)
    if pred_df.empty:
        print("No valid test slices found on disk.")
        return None

    # Aggregate by patient
    patient_records = []
    for subject_id, group in pred_df.groupby("subject_id"):
        true_label = group["true_label"].iloc[0]
        probs = group["prob_mod"].values
        n_slices = len(probs)
        
        # Aggregation strategies
        prob_mean = float(np.mean(probs))
        prob_max = float(np.max(probs))
        prob_median = float(np.median(probs))
        top_k = float(np.mean(np.sort(probs)[-max(1, int(0.1 * n_slices)):]))  # top 10% slices
        pct_slices_positive = float(np.mean(probs >= threshold))

        patient_records.append({
            "subject_id": subject_id,
            "true_label": true_label,
            "n_slices": n_slices,
            "prob_mean": prob_mean,
            "prob_max": prob_max,
            "prob_median": prob_median,
            "prob_top10pct": top_k,
            "pct_positive_slices": pct_slices_positive,
            "pred_mean_thresh": int(prob_mean >= threshold),
            "pred_majority_vote": int(pct_slices_positive >= 0.50),
            "pred_top10pct_thresh": int(top_k >= threshold)
        })

    pat_df = pd.DataFrame(patient_records)

    # Calculate metrics
    y_true = pat_df["true_label"].values
    
    print("\n" + "=" * 70)
    print("PATIENT-LEVEL CLINICAL DIAGNOSIS BENCHMARK (UNSEEN SUBJECTS)")
    print("=" * 70)

    strategies = {
        "Mean Probability Voting": pat_df["pred_mean_thresh"].values,
        "Majority Slice Voting": pat_df["pred_majority_vote"].values,
        "Top-10% Focal Atrophy Voting": pat_df["pred_top10pct_thresh"].values
    }

    summary_rows = []
    for strat_name, y_pred in strategies.items():
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred, zero_division=0)
        tn, fp_cnt, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp_cnt) if (tn + fp_cnt) > 0 else 0.0
        
        try:
            auc = roc_auc_score(y_true, pat_df["prob_mean"].values if "Mean" in strat_name else pat_df["prob_top10pct"].values)
        except Exception:
            auc = 0.5

        print(f"Strategy: {strat_name}")
        print(f"  Accuracy: {acc*100:.1f}% | Balanced Acc: {bal_acc*100:.1f}%")
        print(f"  Sensitivity (Recall): {sens*100:.1f}% | Specificity: {spec*100:.1f}%")
        print(f"  Patient Confusion Matrix: TP={tp}, FP={fp_cnt}, TN={tn}, FN={fn}")
        print(f"  Subject-Level ROC-AUC: {auc:.4f}\n")

        summary_rows.append({
            "strategy": strat_name,
            "patient_accuracy": acc,
            "balanced_accuracy": bal_acc,
            "sensitivity": sens,
            "specificity": spec,
            "roc_auc": auc,
            "tp": tp, "fp": fp_cnt, "tn": tn, "fn": fn
        })

    out_csv = os.path.join(BASE_DIR, "experiments", "patient_level_evaluation_report.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"Saved patient-level metrics report to: {out_csv}")

    return pat_df, pd.DataFrame(summary_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data_manifest_sample.csv")
    parser.add_argument("--checkpoint", default="experiments/track_b/classical_axial/classical_axial_seed0_best.pt")
    parser.add_argument("--plane", default="axial")
    parser.add_argument("--threshold", type=float, default=0.0005)
    args = parser.parse_args()

    run_patient_aggregation(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        plane=args.plane,
        threshold=args.threshold
    )
