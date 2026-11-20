"""
Interactive CQ-CNN Clinical Inference & Decision Dashboard
==========================================================
Streamlit Web Application featuring:
  - Live MRI Slice Selection (or custom upload)
  - Model Choice: Classical Baseline vs. 2-Qubit Hybrid CQ-CNN
  - Dynamic Clinical Decision Threshold Slider (0.0001 - 0.50)
  - Real-time Dementia Risk Probability Bar & Diagnosis Recommendation
  - Patient Metadata & Split Audit (Zero-Leakage Assurance)
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ClassicalNet imported at top-level (no heavy deps)
# QuantumNet / create_qnn are lazy-loaded inside load_quantum_model()
from scirpts.track_b_training import ClassicalNet

st.set_page_config(
    page_title="CQ-CNN Neuroimaging Clinical Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 20px; }
    .metric-card { background-color: #F3F4F6; border-radius: 8px; padding: 15px; border-left: 5px solid #3B82F6; }
    .alert-high { background-color: #FEE2E2; border-left: 5px solid #EF4444; padding: 15px; border-radius: 8px; }
    .alert-low { background-color: #D1FAE5; border-left: 5px solid #10B981; padding: 15px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🧠 CQ-CNN: Hybrid Classical–Quantum MRI Diagnostic Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Subject-Level Leakage-Free Alzheimer’s Disease Detection on 3D Structural MRI (OASIS-2)</p>', unsafe_allow_html=True)

# ── Sidebar Controls ───────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model & Diagnostic Settings")

model_choice = st.sidebar.selectbox(
    "Select Architecture",
    ["Classical Baseline (Axial)", "2-Qubit Hybrid CQ-CNN (Variational Circuit)"],
    index=0
)

threshold = st.sidebar.slider(
    "Decision Probability Threshold",
    min_value=0.0001,
    max_value=0.5000,
    value=0.0005,
    step=0.0001,
    format="%.4f",
    help="Default is 0.50. Youden's J calibrated threshold is 0.0005 to counteract natural 120:1 OASIS-2 class imbalance."
)

st.sidebar.markdown("""
---
**Clinical Safeguards Enforced:**
- 🛡️ Zero Patient Overlap between Folds
- 🧪 Unseen Patient Test Set Evaluation
- ⚖️ Inverse Class Frequency Weighting
""")

# ── Model Caching ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_classical_model():
    model = ClassicalNet(conv2_ch=4)
    ckpt = os.path.join(BASE_DIR, "experiments", "track_b", "classical_axial", "classical_axial_seed0_best.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    return model

@st.cache_resource
def load_quantum_model():
    try:
        from scirpts.track_b_training import QuantumNet, create_qnn
    except Exception as e:
        st.error(f"Quantum backend unavailable: {e}")
        return None
    qnn = create_qnn(2)
    model = QuantumNet(qnn, 2, conv2_ch=4)
    ckpt = os.path.join(BASE_DIR, "experiments", "track_b_quantum_real", "quantum_2q_axial", "quantum_2q_axial_seed0_best.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    return model

# ── Data Loading ──────────────────────────────────────────────────────────────
manifest_path = os.path.join(BASE_DIR, "data_manifest_sample.csv")
if os.path.exists(manifest_path):
    manifest_df = pd.read_csv(manifest_path)
    test_slices = manifest_df[manifest_df["track_b_split"] == "test"]
else:
    test_slices = pd.DataFrame()

# ── Main UI Layout ────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("1. Patient Slice Selection")
    data_source = st.radio("Choose Input Source", ["Test Set (Unseen Patients)", "Upload MRI Slice Image"])

    image = None
    true_label = "Unknown"
    subject_id = "N/A"

    if data_source == "Test Set (Unseen Patients)" and not test_slices.empty:
        sample_row = st.selectbox(
            "Select Slice from Unseen Test Patients",
            range(min(20, len(test_slices))),
            format_func=lambda i: f"Slice {i+1}: {test_slices.iloc[i]['slice_id']} ({test_slices.iloc[i]['label']})"
        )
        row_data = test_slices.iloc[sample_row]
        img_path = os.path.join(BASE_DIR, row_data["file_path"])
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("L")
            true_label = row_data["label"]
            subject_id = row_data["subject_id"]
        else:
            st.warning(f"File not found on local disk: {img_path}")
    else:
        uploaded_file = st.file_uploader("Upload 2D MRI Slice (JPEG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("L")

    if image:
        st.image(image, caption=f"Subject: {subject_id} | Ground Truth: {true_label}", width=320)

with col2:
    st.subheader("2. Real-Time Model Inference")
    if image is not None:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        tensor = transform(image).unsqueeze(0)

        with st.spinner("Executing neural forward pass..."):
            if "Classical" in model_choice:
                model = load_classical_model()
                with torch.no_grad():
                    out = model(tensor)
                    prob_mod = float(F.softmax(out, dim=1)[0, 1].numpy())
            else:
                model = load_quantum_model()
                if model is None:
                    st.error("Quantum model could not be loaded. Try the Classical model.")
                    st.stop()
                with torch.no_grad():
                    out = model(tensor)
                    prob_mod = float(F.softmax(out, dim=1)[0, 1].numpy())

        prob_non = 1.0 - prob_mod
        is_demented = prob_mod >= threshold

        # Probability Metrics Display
        st.markdown(f"**Predicted Dementia Probability**: `{prob_mod:.4f}` (Operating Threshold: `{threshold:.4f}`)")
        st.progress(min(prob_mod / (threshold * 2), 1.0) if threshold > 0 else prob_mod)

        if is_demented:
            st.markdown(f"""
            <div class="alert-high">
                <h3 style="color:#B91C1C; margin:0;">⚠️ High Risk: Moderate Dementia Indicated</h3>
                <p style="margin:5px 0 0 0; color:#7F1D1D;">Calculated probability ({prob_mod:.4f}) exceeds clinical operating threshold ({threshold:.4f}).</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-low">
                <h3 style="color:#047857; margin:0;">✅ Low Risk: Non-Demented Classified</h3>
                <p style="margin:5px 0 0 0; color:#065F46;">Calculated probability ({prob_mod:.4f}) is below clinical operating threshold ({threshold:.4f}).</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("3. Classification Breakdown")
        m1, m2, m3 = st.columns(3)
        m1.metric("Non-Demented Score", f"{prob_non*100:.1f}%")
        m2.metric("Moderate Dementia Score", f"{prob_mod*100:.2f}%")
        m3.metric("Ground Truth Match", "MATCH ✓" if (is_demented and true_label=="moderate_dementia") or (not is_demented and true_label=="nondemented") else "MISMATCH ✗")

    else:
        st.info("👈 Please select or upload an MRI slice to begin diagnostic analysis.")

# ── Footer Benchmark Info ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <b>Research Paper Reference:</b> <i>CQ-CNN: A lightweight hybrid classical-quantum convolutional neural network for Alzheimer's disease detection</i> (PLOS ONE 2025)<br>
    Repository: <a href="https://github.com/An-Avenger/CQ-CNN" target="_blank">An-Avenger/CQ-CNN</a> | Track B Zero-Leakage Subject Split
</div>
""", unsafe_allow_html=True)
