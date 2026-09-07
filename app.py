"""
OralGuard AI — Streamlit Dashboard
===================================
Two tabs:
  1. Results Dashboard — shows Phase 1-4 accuracy/F1/AUC comparisons from
     the CSV/JSON files your training notebooks already saved.
  2. Live Demo — upload a histopathology image and get a live prediction
     from any of the three trained models (or a weighted ensemble).

Run with:
    streamlit run app.py

Model checkpoints (.pth) and result files (.csv / .json) are NOT bundled —
point the sidebar paths at wherever you saved them (local folder or a
mounted Google Drive path if running inside Colab + ngrok / ssh tunnel).
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
import timm
from efficientnet_pytorch import EfficientNet


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="OralGuard AI", page_icon="🦷", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "OSCC"]


# ============================================================================
# MODEL ARCHITECTURES — must match training exactly (Phase 1 / 2 / 3)
# ============================================================================
class OralGuardModel(nn.Module):
    """EfficientNet-B0 via efficientnet_pytorch (Phase 1)"""
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        if pretrained:
            self.backbone = EfficientNet.from_pretrained("efficientnet-b0")
        else:
            self.backbone = EfficientNet.from_name("efficientnet-b0")
        num_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def create_vgg16_model(num_classes=2, dropout=0.5):
    """VGG16-BN exactly as built in Phase 2"""
    model = models.vgg16_bn(pretrained=False)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 4096),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(4096, num_classes),
    )
    return model


class OralGuardViT(nn.Module):
    """Vision Transformer (Phase 3)"""
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=0
        )
        embed_dim = self.backbone.embed_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


MODEL_BUILDERS = {
    "EfficientNet-B0": lambda: OralGuardModel(num_classes=2, pretrained=False),
    "VGG16": lambda: create_vgg16_model(num_classes=2),
    "ViT-B/16": lambda: OralGuardViT(num_classes=2, pretrained=False),
}


# ============================================================================
# TRANSFORMS (must match test-time transforms used in training)
# ============================================================================
def get_test_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ============================================================================
# MODEL LOADING (cached so it only loads once per checkpoint path)
# ============================================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model(model_name: str, checkpoint_path: str):
    model = MODEL_BUILDERS[model_name]()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def predict(model, image: Image.Image):
    transform = get_test_transforms()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    return probs


# ============================================================================
# SIDEBAR — paths
# ============================================================================
st.sidebar.title("🦷 OralGuard AI")
st.sidebar.markdown("### Checkpoint & Results Paths")

phase_dirs = {
    "Phase-1 (EfficientNet-B0)": st.sidebar.text_input(
        "Phase-1 folder", "oralguard_models/Phase-1"
    ),
    "Phase-2 (VGG16)": st.sidebar.text_input("Phase-2 folder", "oralguard_models/Phase-2"),
    "Phase-3 (ViT-B/16)": st.sidebar.text_input("Phase-3 folder", "oralguard_models/Phase-3"),
    "Phase-4 (Ensemble)": st.sidebar.text_input("Phase-4 folder", "oralguard_models/Phase-4"),
}

CHECKPOINT_FILES = {
    ("EfficientNet-B0", "100x"): ("Phase-1 (EfficientNet-B0)", "oralguard_efficientnet_b0_100x_best.pth"),
    ("EfficientNet-B0", "400x"): ("Phase-1 (EfficientNet-B0)", "oralguard_efficientnet_b0_400x_best.pth"),
    ("VGG16", "100x"): ("Phase-2 (VGG16)", "oralguard_vgg16_100x_best.pth"),
    ("VGG16", "400x"): ("Phase-2 (VGG16)", "oralguard_vgg16_400x_best.pth"),
    ("ViT-B/16", "100x"): ("Phase-3 (ViT-B/16)", "oralguard_vit_100x_best.pth"),
    ("ViT-B/16", "400x"): ("Phase-3 (ViT-B/16)", "oralguard_vit_400x_best.pth"),
}

st.title("🦷 OralGuard AI — Oral Cancer Histopathology Classifier")

tab1, tab2 = st.tabs(["📊 Results Dashboard", "🔬 Live Demo"])


# ============================================================================
# TAB 1 — RESULTS DASHBOARD
# ============================================================================
with tab1:
    st.header("Model Performance Overview")

    comparison_path = os.path.join(phase_dirs["Phase-4 (Ensemble)"], "ensemble_comparison.csv")
    summary_path = os.path.join(phase_dirs["Phase-4 (Ensemble)"], "phase4_summary.json")

    if os.path.exists(comparison_path):
        df = pd.read_csv(comparison_path)
        st.subheader("Individual Models vs Ensemble")
        st.dataframe(df, use_container_width=True)

        for mag in ["100x", "400x"]:
            mag_df = df[df["Model"].str.contains(mag)]
            if not mag_df.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ["#1f77b4" if t == "Individual" else "#ff7f0e" for t in mag_df["Type"]]
                ax.bar(mag_df["Model"], mag_df["Accuracy (%)"], color=colors)
                ax.set_ylabel("Accuracy (%)")
                ax.set_title(f"{mag} Magnification — Individual vs Ensemble")
                plt.xticks(rotation=30, ha="right")
                st.pyplot(fig)
    else:
        st.info(
            f"Couldn't find `{comparison_path}`. Update the Phase-4 folder path in the "
            "sidebar, or upload the CSV manually below."
        )
        uploaded_csv = st.file_uploader("Upload ensemble_comparison.csv", type="csv")
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df, use_container_width=True)

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        st.subheader("Best Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best 100x Model", summary["100x_results"]["best_model"],
                       f"{summary['100x_results']['best_accuracy']*100:.2f}%")
        with col2:
            st.metric("Best 400x Model", summary["400x_results"]["best_model"],
                       f"{summary['400x_results']['best_accuracy']*100:.2f}%")


# ============================================================================
# TAB 2 — LIVE DEMO
# ============================================================================
with tab2:
    st.header("Try a Prediction")

    col1, col2 = st.columns([1, 1])
    with col1:
        magnification = st.selectbox("Magnification", ["100x", "400x"])
        mode = st.radio("Mode", ["Single Model", "Weighted Ensemble"])

        if mode == "Single Model":
            model_choice = st.selectbox("Model", list(MODEL_BUILDERS.keys()))
        else:
            st.caption("Ensemble weights (auto-normalized)")
            w_eff = st.slider("EfficientNet-B0 weight", 0.0, 1.0, 0.33)
            w_vgg = st.slider("VGG16 weight", 0.0, 1.0, 0.33)
            w_vit = st.slider("ViT-B/16 weight", 0.0, 1.0, 0.34)

    with col2:
        uploaded_image = st.file_uploader(
            "Upload a histopathology image", type=["jpg", "jpeg", "png"]
        )
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded image", use_container_width=True)

    if uploaded_image and st.button("🔍 Predict", type="primary"):
        image = Image.open(uploaded_image)

        try:
            if mode == "Single Model":
                folder_key, filename = CHECKPOINT_FILES[(model_choice, magnification)]
                ckpt_path = os.path.join(phase_dirs[folder_key], filename)
                model = load_model(model_choice, ckpt_path)
                probs = predict(model, image)
            else:
                weights = np.array([w_eff, w_vgg, w_vit])
                weights = weights / weights.sum()
                probs_list = []
                for name in ["EfficientNet-B0", "VGG16", "ViT-B/16"]:
                    folder_key, filename = CHECKPOINT_FILES[(name, magnification)]
                    ckpt_path = os.path.join(phase_dirs[folder_key], filename)
                    model = load_model(name, ckpt_path)
                    probs_list.append(predict(model, image))
                probs = sum(w * p for w, p in zip(weights, probs_list))

            pred_idx = int(np.argmax(probs))
            pred_label = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100

            st.success(f"**Prediction: {pred_label}** ({confidence:.1f}% confidence)")

            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.barh(CLASS_NAMES, probs * 100, color=["#2ca02c", "#d62728"])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Confidence (%)")
            for i, v in enumerate(probs * 100):
                ax.text(v + 1, i, f"{v:.1f}%", va="center")
            st.pyplot(fig)

        except FileNotFoundError as e:
            st.error(
                f"Checkpoint not found: {e}. Check the folder paths in the sidebar — "
                "they should point to wherever your `.pth` files are saved."
            )

    st.caption(
        "⚠️ Research prototype — not a diagnostic tool. Predictions should not be used "
        "for actual clinical decisions."
    )
