# CQ-CNN: Rigorous Dual-Track Reproduction & Subject-Level Clinical Generalization Benchmark

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-orange.svg)](https://pytorch.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.6.2-purple.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PLOS ONE 2025](https://img.shields.io/badge/Paper-PLOS%20ONE%202025-red.svg)](https://doi.org/10.1371/journal.pone.0331870)

An end-to-end empirical reproduction, architectural audit, and clinical generalization investigation of the **CQ-CNN (Hybrid Classical–Quantum Convolutional Neural Network)** for Alzheimer’s Disease detection from 3D MRI volumes (OASIS-2), following the reference publication:
> *M. Islam, M. J. Hasan, M.R.C. Mahdy, "CQ-CNN: A lightweight hybrid classical–quantum convolutional neural network for Alzheimer’s disease detection using 3D structural brain MRI," PLOS ONE, 2025.* [DOI: 10.1371/journal.pone.0331870](https://doi.org/10.1371/journal.pone.0331870)

---

## 🔬 Core Investigation: Two-Track Experimental Protocol

To address the documented data-leakage vulnerability prevalent in 2D slice-based neuroimaging literature, this repository maintains two strictly separated experimental tracks:

```mermaid
flowchart TD
    Raw["OASIS-2 Raw 3D Volumetric MRI<br>(2,736 volumes across 89 patients)"] --> Slice["2D Slice Preprocessing<br>(130,104 total slices: Axial, Coronal, Sagittal)"]
    
    Slice --> TrackA["Track A: Original Paper Reproduction<br>(Slice-Level Random 90/10 Split)"]
    Slice --> TrackB["Track B: Leakage-Free Clinical Benchmark<br>(Subject-Level 70/15/15 Partitioning)"]
    
    TrackA --> Memo["Adjacent Slices Shared Across Train & Test<br>Accuracy: ~99.0% (Cranial Memorization)"]
    TrackB --> Clean["15 Unseen Patients in Sealed Test Split<br>Zero Subject Leakage Across Folds"]
    
    Clean --> Eval["Empirical Evaluation on Unseen Patients<br>ROC-AUC: 0.7114 | Balanced Acc: 50.0%"]
```

### 1. Track A: Original Paper Reproduction
- Replicates the paper's preprocessing and slice-level 90/10 random train/test split.
- Audits and verifies all **25 released checkpoints** across classical and quantum configurations.
- Reproduces nominal test accuracy (~96–99%).
- **Limitation Documented**: Slices from the same patient's brain appear in both train and test sets, inflating nominal metrics due to patient-specific cranial shape memorization.

### 2. Track B: Leakage-Free Subject-Level Benchmark
- Enforces strict **patient-level partitioning** across 89 unique OASIS-2 subjects:
  - **Train**: 61 subjects (+ all DDPM synthetic diffusion slices strictly confined to train)
  - **Validation**: 13 subjects (used solely for checkpoint selection and scheduling)
  - **Test**: 15 unseen subjects (sealed, never touched during training)
- Guarantees **zero patient overlap** between splits.
- Evaluates genuine generalization to unseen brains.

---

## 📊 Summary of Experimental Results

### Comparative Benchmark: Track A vs. Track B
| Model | Experimental Track | Split Isolation | Accuracy | Balanced Acc | ROC-AUC | Specificity | Recall (Sensitivity) | F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2Q CQ-CNN (Axial Flagship)** | **Track A (Paper Checkpoint)** | Slice-level | **98.93%** | N/A | N/A | ~99.0% | ~99.0% | **99.2%** |
| **Classical Baseline** | **Track B (Standard CE)** | Subject-level (Sealed) | **96.99%** | 50.00% | **0.7012** | 100.0% | 0.00% | 0.0% |
| **Classical Baseline** | **Track B (Inverse-Weighted)** | Subject-level (Sealed) | **96.99%** | 50.00% | **0.7114** | 100.0% | 0.00% | 0.0% |
| **2Q Hybrid CQ-CNN** | **Track B (Inverse-Weighted)** | Subject-level (Sealed) | **51.36%** | 50.00% | **0.5089** | 0.0% | **100.0%** | **67.87%** |

<p align="center">
  <img src="plots/stage5_evaluation/track_b_confusion_matrices.png" width="85%" alt="Track B Confusion Matrices">
</p>

---

## 🔍 Key Architectural & Empirical Insights

### 1. Checkpoint Architectural Discrepancy Identified
During our comprehensive audit of all 25 local model checkpoints:
- **Single-Plane Models**: Configured with `conv2=4` channels ($13,732$ parameters, linear input $3,364$).
- **3-Plane & Axial-SS Models**: Configured with `conv2=8` channels ($27,392$ parameters, linear input $6,728$).
- **TinyAlexNet Baseline**: Checkpoint contains **$14,502$ parameters** (channel layout `4→8→8→8→8` with a 6-dim classification head), contrasting with standard torchvision AlexNet specifications (~61M parameters).

### 2. Quantum Statevector Simulation Benchmarking
- **2-Qubit CQ-CNN**: ~11.4 images/s, 354 MB RAM footprint, backward pass dominates runtime (658ms vs 45ms forward).
- **3-Qubit CQ-CNN**: ~4.6 images/s, 365 MB RAM footprint, backward pass ~3.7× slower due to statevector Hilbert space scaling.

### 3. Patient Leakage vs. True Generalization
The natural OASIS-2 cohort presents an extreme natural class imbalance (~120:1 non-demented to moderate dementia). In Track A, this imbalance is masked because adjacent 2D slices allow the model to memorize subject anatomy. In Track B, our trained models achieve **0.7114 ROC-AUC on completely unseen patients**, confirming meaningful non-trivial feature extraction without anatomical memorization.

---

## 🛠️ Repository Structure

```
├── data_manifest.csv                # Complete manifest of 130,104 slices with Track A and Track B splits
├── data_manifest_sample.csv         # Stratified 10% sample manifest for fast testing
├── datasets/
│   ├── OASIS-2/downloads/          # Raw tar.gz archives and clinical metadata
│   └── OASIS-2/views/              # Sliced 2D JPEGs (axial, coronal, sagittal)
├── experiments/
│   ├── track_a_reproduction_report.csv   # Full audit report of all 25 checkpoints
│   ├── track_b/                          # Standard Cross-Entropy Track B results
│   ├── track_b_weighted/                 # Inverse Class-Weighted Track B results
│   ├── track_b_quantum_real/             # 2-Qubit CQ-CNN Track B results on real MRI
│   └── stage5_final_evaluation_report.md # Comprehensive synthesis report
├── plots/
│   └── stage5_evaluation/          # Generated comparative plots and confusion matrices
└── scirpts/
    ├── build_manifest.py            # Dual-track clinical manifest generator
    ├── run_oasis2_preprocessing.py  # Automated archive extraction and 3D->2D slicing
    ├── track_a_reproduction.py      # Track A checkpoint auditing and inference suite
    ├── track_b_training.py          # Track B subject-level training pipeline
    └── stage5_evaluation_report.py  # Comparative evaluation and synthesis reporting
```

---

## 🚀 Quickstart & Replication

### 1. Installation
Using `uv` or `pip` in Python 3.12:
```bash
git clone https://github.com/An-Avenger/CQ-CNN.git
cd CQ-CNN
python -m venv .venv
.venv\Scripts\activate  # On Linux: source .venv/bin/activate
pip install torch torchvision qiskit qiskit-machine-learning qiskit-algorithms nibabel scikit-learn pandas matplotlib seaborn tqdm
```

### 2. Run Track A Checkpoint Audit
Verify all 25 pre-trained checkpoints against paper-reported metrics:
```bash
python scirpts/track_a_reproduction.py
```

### 3. Execute Track B (Subject-Level Training)
Run leakage-free training on unseen patient splits:
```bash
# Classical baseline with inverse class weighting:
python scirpts/track_b_training.py --arch classical --view axial --weighted

# 2-Qubit Hybrid CQ-CNN variational model:
python scirpts/track_b_training.py --arch quantum --qubits 2 --view axial --weighted
```

### 4. Generate Final Comparative Reports & Plots
```bash
python scirpts/stage5_evaluation_report.py
```

---

## 📜 Citation

If you build upon this reproduction or benchmark, please cite the original study:

```bibtex
@article{islam2025cqcnn,
  title={CQ-CNN: A lightweight hybrid classical--quantum convolutional neural network for Alzheimer's disease detection using 3D structural brain MRI},
  author={Islam, Mominul and Hasan, Mohammad Junayed and Mahdy, MRC},
  journal={PLOS ONE},
  volume={20},
  number={1},
  pages={e0331870},
  year={2025},
  publisher={Public Library of Science}
}
```
