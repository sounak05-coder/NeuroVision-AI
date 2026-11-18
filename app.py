
# ============================================================
# NEUROVISION AI
# BRAIN TUMOUR SEGMENTATION USING U-NET
# PHASE 11 - REDESIGNED STREAMLIT APPLICATION
# ============================================================

import os
import io
import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit.components.v1 as components

from PIL import Image
from pathlib import Path
from report_generator import generate_segmentation_report
from session_storage import save_segmentation_session
from session_storage import get_segmentation_history


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

WEBAPP_DIR = BASE_DIR
RESULTS_DIR = BASE_DIR / "results"

MODEL_PATH = WEBAPP_DIR / "best_brain_tumor_unet.keras"

METRICS_PATH = RESULTS_DIR / "final_test_metrics.csv"
CONFIG_PATH = RESULTS_DIR / "final_model_config.json"

LOSS_GRAPH = RESULTS_DIR / "training_loss.png"
DICE_GRAPH = RESULTS_DIR / "training_dice.png"
THRESHOLD_GRAPH = RESULTS_DIR / "threshold_comparison.png"
TRAINING_LOG_PATH = RESULTS_DIR / "training_log.csv"

IMG_SIZE = 256
DEFAULT_THRESHOLD = 0.60


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeuroVision AI | Brain Tumour Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. FUTURISTIC NEUROVISION AI THEME

st.markdown(
    '<style>\n\n/* ==========================================\n   NEUROVISION AI - FUTURISTIC MEDICAL THEME\n   ========================================== */\n\n.stApp {\n    background:\n        radial-gradient(ellipse at 15% 10%, rgba(0, 153, 255, 0.13), transparent 35%),\n        radial-gradient(ellipse at 85% 85%, rgba(0, 220, 255, 0.09), transparent 35%),\n        linear-gradient(135deg, #050B18 0%, #091629 50%, #07101F 100%);\n    color: #E8F5FF;\n}\n\n/* Main content */\n.main .block-container {\n    max-width: 1500px;\n    padding-top: 2rem;\n    padding-bottom: 3rem;\n}\n\n/* Sidebar */\n[data-testid="stSidebar"] {\n    background: linear-gradient(180deg, #09172B 0%, #0D2038 100%);\n    border-right: 1px solid rgba(0, 195, 255, 0.25);\n}\n\n/* Headings */\n.stApp h1, .stApp h2, .stApp h3, .stApp h4 {\n    color: #F2FAFF;\n    font-weight: 700;\n    letter-spacing: 0.3px;\n}\n\n/* General text */\n.stApp p, .stApp li, .stApp label {\n    color: #C6D9ED;\n}\n\n[data-testid="stCaptionContainer"] {\n    color: #91AAC5;\n}\n\n/* Glowing metric cards */\n[data-testid="stMetric"] {\n    background: linear-gradient(\n        145deg,\n        rgba(18, 44, 72, 0.92),\n        rgba(9, 27, 49, 0.92)\n    );\n    border: 1px solid rgba(48, 180, 255, 0.32);\n    border-radius: 16px;\n    padding: 20px;\n    box-shadow: 0 0 18px rgba(0, 145, 255, 0.08);\n    transition: all 0.3s ease;\n}\n\n[data-testid="stMetric"]:hover {\n    transform: translateY(-4px);\n    border-color: #32CFFF;\n    box-shadow: 0 0 24px rgba(0, 195, 255, 0.20);\n}\n\n[data-testid="stMetricLabel"] {\n    color: #91B9D8;\n}\n\n[data-testid="stMetricValue"] {\n    color: #53D9FF;\n}\n\n/* Buttons */\n.stButton > button,\n.stDownloadButton > button {\n    background: linear-gradient(100deg, #087FC9, #12B9D6);\n    color: white;\n    border: 1px solid rgba(94, 225, 255, 0.5);\n    border-radius: 10px;\n    min-height: 42px;\n    font-weight: 650;\n    transition: all 0.25s ease;\n}\n\n.stButton > button:hover,\n.stDownloadButton > button:hover {\n    transform: translateY(-2px);\n    box-shadow: 0 0 18px rgba(0, 195, 255, 0.35);\n    border-color: #71E8FF;\n    color: white;\n}\n\n/* File uploader */\n[data-testid="stFileUploader"] {\n    background: rgba(13, 35, 59, 0.72);\n    border: 1px dashed #20BFEA;\n    border-radius: 14px;\n    padding: 16px;\n}\n\n/* Custom information cards */\n.info-card {\n    background: linear-gradient(\n        145deg,\n        rgba(18, 45, 73, 0.85),\n        rgba(8, 25, 46, 0.92)\n    );\n    border: 1px solid rgba(50, 190, 255, 0.25);\n    border-radius: 16px;\n    padding: 22px;\n    margin-bottom: 16px;\n    box-shadow: 0 5px 25px rgba(0, 0, 0, 0.18);\n    transition: all 0.3s ease;\n}\n\n.info-card:hover {\n    transform: translateY(-4px);\n    border-color: rgba(61, 216, 255, 0.75);\n    box-shadow: 0 0 22px rgba(0, 180, 255, 0.13);\n}\n\n/* Sidebar navigation */\n[data-testid="stSidebar"] .stRadio label {\n    color: #D5E9FA;\n}\n\n/* Dividers */\nhr {\n    border-color: rgba(65, 175, 225, 0.22);\n}\n\n/* Smooth appearance */\n@keyframes neuroFadeIn {\n    from {\n        opacity: 0;\n        transform: translateY(10px);\n    }\n    to {\n        opacity: 1;\n        transform: translateY(0);\n    }\n}\n\n.main .block-container {\n    animation: neuroFadeIn 0.65s ease-out;\n}\n\n/* Subtle animated background glow */\n@keyframes neuroGlow {\n    0%, 100% { opacity: 0.45; }\n    50% { opacity: 0.85; }\n}\n\n[data-testid="stSidebar"]::after {\n    content: "";\n    display: block;\n    height: 3px;\n    margin: 12px;\n    background: linear-gradient(90deg, transparent, #16C9F5, transparent);\n    animation: neuroGlow 3s ease-in-out infinite;\n}\n\n/* Respect reduced-motion preferences */\n@media (prefers-reduced-motion: reduce) {\n    *, *::before, *::after {\n        animation-duration: 0.01ms !important;\n        transition-duration: 0.01ms !important;\n    }\n}\n\n\n\n/* ==========================================\n   DASHBOARD - FUTURISTIC ANIMATIONS\n   ========================================== */\n\n/* Glowing dashboard heading */\n.neuro-hero-title {\n    font-size: clamp(2rem, 4vw, 3.2rem);\n    font-weight: 800;\n    color: #EAFBFF;\n    letter-spacing: 1px;\n    text-shadow:\n        0 0 10px rgba(40, 205, 255, 0.45),\n        0 0 28px rgba(0, 145, 255, 0.25);\n    animation: neuroTitleIn 1s ease-out both;\n}\n\n/* Subtitle */\n.neuro-hero-subtitle {\n    color: #8FDFFF;\n    font-size: 1.08rem;\n    letter-spacing: 0.4px;\n    margin-top: 8px;\n}\n\n/* AI system status */\n.neuro-status {\n    display: inline-flex;\n    align-items: center;\n    gap: 9px;\n    padding: 9px 15px;\n    border-radius: 30px;\n    background: rgba(0, 190, 220, 0.09);\n    border: 1px solid rgba(42, 220, 255, 0.35);\n    color: #8FEFFF;\n    font-size: 0.85rem;\n    letter-spacing: 1px;\n}\n\n.neuro-status-dot {\n    width: 9px;\n    height: 9px;\n    background: #29F0C5;\n    border-radius: 50%;\n    box-shadow: 0 0 12px #29F0C5;\n    animation: neuroPulse 1.8s infinite;\n}\n\n/* Animated scanning panel */\n.neuro-scan-panel {\n    position: relative;\n    height: 5px;\n    width: 100%;\n    overflow: hidden;\n    border-radius: 10px;\n    background: rgba(35, 150, 210, 0.12);\n    margin: 20px 0;\n}\n\n.neuro-scan-line {\n    position: absolute;\n    height: 100%;\n    width: 28%;\n    background: linear-gradient(\n        90deg,\n        transparent,\n        #25DFFF,\n        #FFFFFF,\n        #25DFFF,\n        transparent\n    );\n    box-shadow: 0 0 15px #16CFFF;\n    animation: neuroScan 4s linear infinite;\n}\n\n/* Animation keyframes */\n@keyframes neuroPulse {\n    0%, 100% {\n        opacity: 1;\n        transform: scale(1);\n    }\n    50% {\n        opacity: 0.45;\n        transform: scale(0.75);\n    }\n}\n\n@keyframes neuroScan {\n    from { left: -30%; }\n    to { left: 110%; }\n}\n\n@keyframes neuroTitleIn {\n    from {\n        opacity: 0;\n        transform: translateY(16px);\n    }\n    to {\n        opacity: 1;\n        transform: translateY(0);\n    }\n}\n\n/* Accessibility */\n@media (prefers-reduced-motion: reduce) {\n    .neuro-hero-title,\n    .neuro-status-dot,\n    .neuro-scan-line {\n        animation: none !important;\n    }\n}\n\n\n\n/* ==========================================\n   NEUROVISION AI - MRI SCANNING INTERFACE\n   ========================================== */\n\n.neuro-segmentation-header {\n    padding: 22px 24px;\n    border: 1px solid rgba(39, 205, 255, 0.35);\n    border-radius: 16px;\n    background: linear-gradient(\n        120deg,\n        rgba(8, 43, 75, 0.92),\n        rgba(8, 22, 43, 0.95)\n    );\n    box-shadow: 0 0 25px rgba(0, 175, 255, 0.10);\n    margin-bottom: 20px;\n    animation: neuroTitleIn 0.8s ease-out both;\n}\n\n.neuro-segmentation-title {\n    font-size: 2rem;\n    font-weight: 800;\n    color: #EAFBFF;\n    text-shadow: 0 0 15px rgba(35, 210, 255, 0.35);\n}\n\n.neuro-segmentation-subtitle {\n    color: #A8D8EF;\n    margin-top: 7px;\n    font-size: 0.95rem;\n}\n\n.neuro-analysis-status {\n    display: inline-flex;\n    align-items: center;\n    gap: 9px;\n    color: #75E8FF;\n    background: rgba(0, 190, 230, 0.08);\n    border: 1px solid rgba(0, 205, 255, 0.3);\n    padding: 8px 13px;\n    border-radius: 25px;\n    font-size: 0.82rem;\n    letter-spacing: 0.7px;\n    margin-top: 15px;\n}\n\n.neuro-analysis-dot {\n    width: 9px;\n    height: 9px;\n    border-radius: 50%;\n    background: #29F0C5;\n    box-shadow: 0 0 12px #29F0C5;\n    animation: neuroPulse 1.6s infinite;\n}\n\n/* MRI image presentation */\n[data-testid="stImage"] {\n    border: 1px solid rgba(45, 195, 255, 0.25);\n    border-radius: 13px;\n    overflow: hidden;\n    box-shadow: 0 0 18px rgba(0, 150, 255, 0.09);\n    transition: all 0.3s ease;\n}\n\n[data-testid="stImage"]:hover {\n    border-color: rgba(45, 220, 255, 0.7);\n    box-shadow: 0 0 24px rgba(0, 190, 255, 0.18);\n}\n\n/* Prediction result section */\n.neuro-result-heading {\n    color: #66DCFF;\n    font-size: 1.35rem;\n    font-weight: 750;\n    letter-spacing: 0.5px;\n    text-shadow: 0 0 12px rgba(0, 190, 255, 0.25);\n}\n\n/* Accessibility */\n@media (prefers-reduced-motion: reduce) {\n    .neuro-segmentation-header,\n    .neuro-analysis-dot {\n        animation: none !important;\n    }\n\n    [data-testid="stImage"] {\n        transition: none !important;\n    }\n}\n\n\n\n/* ==========================================\n   NEUROVISION AI - MODEL PERFORMANCE\n   ========================================== */\n\n.neuro-performance-header {\n    padding: 24px;\n    border: 1px solid rgba(39, 205, 255, 0.4);\n    border-radius: 17px;\n    background: linear-gradient(\n        120deg,\n        rgba(8, 43, 75, 0.94),\n        rgba(8, 22, 43, 0.96)\n    );\n    box-shadow: 0 0 25px rgba(0, 175, 255, 0.12);\n    margin-bottom: 22px;\n    animation: neuroTitleIn 0.8s ease-out both;\n}\n\n.neuro-performance-title {\n    color: #EAFBFF;\n    font-size: 2rem;\n    font-weight: 800;\n    text-shadow: 0 0 16px rgba(35, 210, 255, 0.4);\n}\n\n.neuro-performance-subtitle {\n    color: #A8D8EF;\n    margin-top: 8px;\n    font-size: 0.95rem;\n}\n\n.neuro-performance-status {\n    display: inline-flex;\n    align-items: center;\n    gap: 9px;\n    padding: 8px 14px;\n    margin-top: 15px;\n    border-radius: 25px;\n    border: 1px solid rgba(0, 205, 255, 0.3);\n    background: rgba(0, 190, 230, 0.08);\n    color: #75E8FF;\n    font-size: 0.82rem;\n    letter-spacing: 0.7px;\n}\n\n.neuro-performance-dot {\n    width: 9px;\n    height: 9px;\n    background: #29F0C5;\n    border-radius: 50%;\n    box-shadow: 0 0 12px #29F0C5;\n    animation: neuroPulse 1.6s infinite;\n}\n\n.neuro-section-title {\n    color: #66DCFF;\n    font-size: 1.35rem;\n    font-weight: 750;\n    letter-spacing: 0.5px;\n    text-shadow: 0 0 12px rgba(0, 190, 255, 0.25);\n    margin-top: 18px;\n    margin-bottom: 14px;\n}\n\n.neuro-evaluation-note {\n    background: rgba(0, 165, 220, 0.08);\n    border: 1px solid rgba(45, 195, 255, 0.35);\n    border-left: 4px solid #27D3F5;\n    border-radius: 10px;\n    padding: 15px 18px;\n    color: #BDEBFF;\n    margin: 14px 0;\n}\n\n/* Chart image presentation */\n[data-testid="stImage"] {\n    border: 1px solid rgba(45, 195, 255, 0.25);\n    border-radius: 13px;\n    overflow: hidden;\n    box-shadow: 0 0 18px rgba(0, 150, 255, 0.09);\n    transition: all 0.3s ease;\n}\n\n[data-testid="stImage"]:hover {\n    border-color: rgba(45, 220, 255, 0.7);\n    box-shadow: 0 0 24px rgba(0, 190, 255, 0.18);\n}\n\n@media (prefers-reduced-motion: reduce) {\n    .neuro-performance-header,\n    .neuro-performance-dot {\n        animation: none !important;\n    }\n}\n\n\n\n/* ==========================================\n   REMOVE STREAMLIT BLACK TOP BAR\n   ========================================== */\n\n/* Main Streamlit header */\n[data-testid="stHeader"] {\n    background: linear-gradient(\n        90deg,\n        #09172B 0%,\n        #081629 50%,\n        #071321 100%\n    ) !important;\n    border-bottom: 1px solid rgba(40, 190, 255, 0.18);\n    box-shadow: none !important;\n}\n\n/* Toolbar area */\n[data-testid="stToolbar"] {\n    background: transparent !important;\n}\n\n/* Header decoration */\n[data-testid="stDecoration"] {\n    background: transparent !important;\n}\n\n/* Main app background behind header */\n.stApp {\n    background-color: #071321;\n}\n\n/* Keep header controls visible */\n[data-testid="stHeader"] button {\n    color: #C9EFFF !important;\n}\n\n\n\n/* ==========================================\n   NEUROVISION AI - NOTIFICATION CONTRAST FIX\n   ========================================== */\n\n/* Success notification */\n[data-testid="stAlert"][kind="success"] {\n    background: rgba(16, 185, 129, 0.13) !important;\n    border: 1px solid #10B981 !important;\n    border-radius: 12px !important;\n}\n\n/* Information notification */\n[data-testid="stAlert"][kind="info"] {\n    background: rgba(14, 165, 233, 0.13) !important;\n    border: 1px solid #0EA5E9 !important;\n    border-radius: 12px !important;\n}\n\n/* Warning notification */\n[data-testid="stAlert"][kind="warning"] {\n    background: rgba(245, 158, 11, 0.13) !important;\n    border: 1px solid #F59E0B !important;\n    border-radius: 12px !important;\n}\n\n/* Error notification */\n[data-testid="stAlert"][kind="error"] {\n    background: rgba(239, 68, 68, 0.13) !important;\n    border: 1px solid #EF4444 !important;\n    border-radius: 12px !important;\n}\n\n/* Notification text - improve readability */\n[data-testid="stAlert"] p,\n[data-testid="stAlert"] div,\n[data-testid="stAlert"] span {\n    color: #E8F5FF !important;\n}\n\n/* Notification icons */\n[data-testid="stAlert"] svg {\n    fill: currentColor;\n}\n\n/* Consistent input labels */\n[data-testid="stWidgetLabel"] p {\n    color: #D6E9F8 !important;\n}\n\n/* Dataframe text */\n[data-testid="stDataFrame"] {\n    border: 1px solid rgba(45, 195, 255, 0.3);\n    border-radius: 12px;\n    overflow: hidden;\n}\n\n\n\n/* ==========================================\n   NATIVE STREAMLIT MENU - PRESERVE CONTROLS\n   ========================================== */\n\n/* Keep the native top toolbar visible */\n[data-testid="stHeader"] {\n    display: flex !important;\n    visibility: visible !important;\n}\n\n/* Preserve native toolbar buttons */\n[data-testid="stToolbar"] {\n    display: flex !important;\n    visibility: visible !important;\n    opacity: 1 !important;\n}\n\n/* Keep menu and toolbar interaction available */\n[data-testid="stHeader"] button,\n[data-testid="stToolbar"] button {\n    visibility: visible !important;\n    pointer-events: auto !important;\n}\n\n/* Do not hide native menu popovers or dialogs */\n[data-testid="stPopover"],\n[role="dialog"],\n[role="menu"] {\n    visibility: visible;\n}\n\n/* Maintain visible keyboard focus */\n[data-testid="stHeader"] button:focus-visible,\n[data-testid="stToolbar"] button:focus-visible {\n    outline: 2px solid #20BFEA !important;\n    outline-offset: 3px !important;\n}\n\n</style>',
    unsafe_allow_html=True
)



# ============================================================
# 4. LOAD TRAINED U-NET MODEL
# ============================================================

@st.cache_resource
def load_unet_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    return model


# ============================================================
# 5. MRI IMAGE PREPROCESSING
# ============================================================

def preprocess_mri(image_bytes):

    """
    Preprocess MRI images using the same channel extraction,
    resizing and normalization approach as model training.
    """

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise ValueError(
            "Unable to read the uploaded MRI image."
        )

    # Handle grayscale MRI
    if image.ndim == 2:

        mri_gray = image.copy()

        mri_display = cv2.cvtColor(
            mri_gray,
            cv2.COLOR_GRAY2RGB
        )

    # Handle colour MRI
    elif image.ndim == 3:

        if image.shape[2] == 4:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2BGR
            )

        if image.shape[2] != 3:
            raise ValueError(
                "Unsupported MRI image format."
            )

        # Extract green channel, matching training preprocessing
        mri_gray = image[:, :, 1]

        mri_display = cv2.cvtColor(
            mri_gray,
            cv2.COLOR_GRAY2RGB
        )

    else:

        raise ValueError(
            "Unsupported MRI image dimensions."
        )

    # Resize MRI image to 256 x 256
    resized = cv2.resize(
        mri_gray,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # Normalize pixel values
    normalized = resized.astype(
        np.float32
    ) / 255.0

    # Add batch and channel dimensions
    model_input = normalized[
        np.newaxis, ..., np.newaxis
    ]

    return mri_display, model_input


# ============================================================
# 6. GENERATE TUMOUR OVERLAY
# ============================================================

def create_overlay(display_mri, predicted_mask):

    overlay = display_mri.copy()

    # Create red colour mask
    red_mask = np.zeros_like(overlay)

    red_mask[:, :, 0] = 255

    # Apply red overlay to predicted tumour pixels
    tumour_region = predicted_mask == 1

    overlay[tumour_region] = (
        0.5 * display_mri[tumour_region]
        + 0.5 * red_mask[tumour_region]
    ).astype(np.uint8)

    return overlay


# ============================================================
# 7. IMAGE DOWNLOAD HELPER
# ============================================================

def image_to_bytes(image_array):

    buffer = io.BytesIO()

    Image.fromarray(
        image_array.astype(np.uint8)
    ).save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# 8. LOAD MODEL CONFIGURATION
# ============================================================

def load_threshold():

    if CONFIG_PATH.exists():

        try:

            with open(
                CONFIG_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                config = json.load(file)

            return float(
                config.get(
                    "threshold",
                    DEFAULT_THRESHOLD
                )
            )

        except Exception:

            return DEFAULT_THRESHOLD

    return DEFAULT_THRESHOLD


# ============================================================
# 9. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 10px 0 15px 0;">
            <h2 style="color:#55A8FF; margin-bottom:0;">
                🧠 NeuroVision AI
            </h2>
            <p style="color:#A9BCD3; font-size:13px;">
                Brain Tumour Segmentation
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    selected_page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Tumour Segmentation",
            "Model Performance",
            "About Project",
            "Session History"
        ],
        index=0
    )

    st.divider()

    # Persistent NeuroVision AI appearance selector
    app_theme = st.radio(
        "APPEARANCE",
        options=["Dark", "Light"],
        format_func=lambda value: "🌙 Dark" if value == "Dark" else "☀️ Light",
        horizontal=True,
        index=0,
        key="neuro_theme_radio"
    )

    st.divider()

    st.caption("AI Research & Development")

    st.caption(
        "Medical AI | Deep Learning"
    )



# ============================================================
# NEUROVISION AI - UI ACCESSIBILITY FIXES
# ============================================================

st.markdown(
    """<style>

/* ==========================================
   NEUROVISION AI - ALERT AND UPLOADER FIX
   ========================================== */

/* High-contrast dark-theme notifications */
[data-testid="stAlert"],
[data-testid="stAlert"] * {
    color: #F8FAFC !important;
}

[data-testid="stAlert"] svg {
    color: #F8FAFC !important;
}

/* Readable file uploader */
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: #DCEAF7 !important;
}

[data-testid="stFileUploader"] button {
    background: #087DB5 !important;
    color: #FFFFFF !important;
    border: 1px solid #43C8F0 !important;
    border-radius: 9px !important;
}

[data-testid="stFileUploader"] button *,
[data-testid="stFileUploader"] button svg {
    color: #FFFFFF !important;
    fill: currentColor !important;
}

</style>""",
    unsafe_allow_html=True
)

if app_theme == "Light":
    st.markdown(
        """<style>

/* ==========================================
   NEUROVISION AI - LIGHT THEME CONTRAST FIX
   ========================================== */

/* Dark readable text on light notification backgrounds */
[data-testid="stAlert"],
[data-testid="stAlert"] * {
    color: #17324B !important;
}

[data-testid="stAlert"] svg {
    color: #17324B !important;
}

/* Light-theme uploader text */
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: #17324B !important;
}

/* Blue upload button with white text */
[data-testid="stFileUploader"] button {
    background: #087DB5 !important;
    color: #FFFFFF !important;
    border: 1px solid #087DB5 !important;
    border-radius: 9px !important;
}

[data-testid="stFileUploader"] button *,
[data-testid="stFileUploader"] button svg {
    color: #FFFFFF !important;
    fill: currentColor !important;
}

/* Segmented theme selector */
[data-testid="stSidebar"] [data-testid="stSegmentedControl"] {
    border: 1px solid #A8D9ED;
    border-radius: 10px;
}

</style>""",
        unsafe_allow_html=True
    )

# NEUROVISION AI - LIGHT THEME OVERRIDES
# ============================================================

if app_theme == "Light":

    st.markdown(
        """
        <style>

        /* Main application */
        .stApp {
            background:
                radial-gradient(
                    ellipse at 15% 10%,
                    rgba(0, 160, 220, 0.08),
                    transparent 38%
                ),
                linear-gradient(
                    135deg,
                    #F4F9FD,
                    #EAF3FA
                ) !important;
            color: #17324B !important;
        }

        /* Main header and toolbar */
        [data-testid="stHeader"] {
            background: #EAF3FA !important;
            border-bottom: 1px solid #D1E2EF !important;
        }

        [data-testid="stToolbar"] {
            background: transparent !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #FFFFFF,
                #EAF3FA
            ) !important;
            border-right: 1px solid #D3E2ED !important;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: #17324B !important;
        }

        /* Main text and headings */
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4 {
            color: #17324B !important;
        }

        [data-testid="stCaptionContainer"] {
            color: #526D83 !important;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background: #FFFFFF !important;
            border: 1px solid #C9DDEB !important;
            box-shadow: 0 4px 18px rgba(31, 85, 120, 0.07);
        }

        [data-testid="stMetricLabel"] {
            color: #526D83 !important;
        }

        [data-testid="stMetricValue"] {
            color: #087DB5 !important;
        }

        /* Dashboard and page headings */
        .neuro-hero-title,
        .neuro-segmentation-title,
        .neuro-performance-title {
            color: #123B59 !important;
            text-shadow: 0 0 12px rgba(0, 150, 210, 0.12) !important;
        }

        .neuro-hero-subtitle,
        .neuro-segmentation-subtitle,
        .neuro-performance-subtitle {
            color: #42677F !important;
        }

        /* Futuristic page headers */
        .neuro-segmentation-header,
        .neuro-performance-header {
            background: linear-gradient(
                120deg,
                #FFFFFF,
                #E6F4FC
            ) !important;
            border-color: #A8D9ED !important;
        }

        /* Information cards */
        .info-card {
            background: #FFFFFF !important;
            border-color: #C9DDEB !important;
        }

        .info-card h3,
        .info-card p {
            color: #17324B !important;
        }

        /* Buttons */
        .stButton > button,
        .stDownloadButton > button {
            color: #FFFFFF !important;
        }

        /* Upload controls */
        [data-testid="stFileUploader"] {
            background: #FFFFFF !important;
            border-color: #48A9D2 !important;
        }

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] p {
            color: #17324B !important;
        }

        /* Inputs and sliders */
        [data-testid="stWidgetLabel"] p {
            color: #17324B !important;
        }

        input, textarea {
            color: #17324B !important;
        }

        /* Alerts - readable light-theme colours */
        [data-testid="stAlert"][kind="success"] {
            background: #E5F7EF !important;
            border-color: #27A66D !important;
        }

        [data-testid="stAlert"][kind="info"] {
            background: #E5F3FC !important;
            border-color: #238BC4 !important;
        }

        [data-testid="stAlert"][kind="warning"] {
            background: #FFF5DE !important;
            border-color: #D99A20 !important;
        }

        [data-testid="stAlert"][kind="error"] {
            background: #FDECEC !important;
            border-color: #D94A4A !important;
        }

        [data-testid="stAlert"] p,
        [data-testid="stAlert"] div,
        [data-testid="stAlert"] span {
            color: #17324B !important;
        }

        /* Tables */
        [data-testid="stDataFrame"] {
            border: 1px solid #C9DDEB !important;
            border-radius: 10px;
        }

        /* Dividers */
        hr {
            border-color: #CADCE9 !important;
        }

        /* Status indicators */
        .neuro-status,
        .neuro-analysis-status,
        .neuro-performance-status {
            color: #08769F !important;
            background: #E2F5FB !important;
            border-color: #A8DDEA !important;
        }

        .neuro-evaluation-note {
            color: #17324B !important;
            background: #E5F3FC !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# 10. DASHBOARD PAGE
# ============================================================



# NEUROVISION ANIMATED PLOTLY CHARTS

def get_chart_theme():
    """Return chart colours based on the selected application theme."""
    if app_theme == "Light":
        return {
            "template": "plotly_white",
            "paper": "#FFFFFF",
            "plot": "#FFFFFF",
            "text": "#17324B",
            "grid": "#DCE6EF",
            "colors": ["#087FC9", "#E87518", "#159B72", "#9254C8"]
        }

    return {
        "template": "plotly_dark",
        "paper": "#09172B",
        "plot": "#0D2038",
        "text": "#E8F5FF",
        "grid": "#29445F",
        "colors": ["#22D3EE", "#FF9F43", "#39D98A", "#B794F4"]
    }


def render_animated_training_chart(
    csv_path, y_columns, labels, title, y_axis_title
):
    if not csv_path.exists():
        st.warning(f"Training data not found: {csv_path.name}")
        return

    df = pd.read_csv(csv_path)

    if not all(column in df.columns for column in y_columns):
        st.error(f"Required chart columns are missing for {title}.")
        return

    if len(df) == 0:
        st.warning("No training records are available.")
        return

    epochs = (
        df["epoch"].tolist()
        if "epoch" in df.columns
        else list(range(len(df)))
    )

    theme = get_chart_theme()
    colors = theme["colors"]

    fig = go.Figure()

    # Initial traces.
    for index, (column, label) in enumerate(zip(y_columns, labels)):
        fig.add_trace(
            go.Scatter(
                x=[epochs[0]],
                y=[float(df[column].iloc[0])],
                mode="lines+markers",
                name=label,
                line=dict(color=colors[index], width=3),
                marker=dict(size=7),
                hovertemplate=(
                    "Epoch: %{x}<br>"
                    + label + ": %{y:.4f}<extra></extra>"
                )
            )
        )

    # Cumulative animation frames.
    frames = []

    for end_index in range(len(df)):
        frame_traces = []

        for column in y_columns:
            frame_traces.append(
                go.Scatter(
                    x=epochs[:end_index + 1],
                    y=df[column].iloc[:end_index + 1].astype(float).tolist()
                )
            )

        frames.append(
            go.Frame(
                name=str(end_index),
                data=frame_traces,
                traces=list(range(len(y_columns)))
            )
        )

    fig.frames = frames

    # Calculate stable axis limits using ALL recorded epochs.
    x_min = float(min(epochs))
    x_max = float(max(epochs))

    x_range = [
        x_min - 0.5,
        x_max + 0.5
    ]

    all_y_values = pd.concat(
        [pd.to_numeric(df[column], errors="coerce") for column in y_columns],
        ignore_index=True
    ).dropna()

    y_min = float(all_y_values.min())
    y_max = float(all_y_values.max())

    y_span = y_max - y_min
    y_padding = max(y_span * 0.08, abs(y_max) * 0.02, 0.000001)

    y_range = [
        y_min - y_padding,
        y_max + y_padding
    ]

    # Preserve the epoch slider.
    slider_steps = [
        {
            "method": "animate",
            "label": str(epochs[index]),
            "args": [
                [str(index)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 450, "redraw": False},
                    "transition": {"duration": 250}
                }
            ]
        }
        for index in range(len(epochs))
    ]

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=21, color=theme["text"])
        ),
        template=theme["template"],
        paper_bgcolor=theme["paper"],
        plot_bgcolor=theme["plot"],
        font=dict(color=theme["text"], size=13),
        xaxis=dict(
            title=dict(text="Epoch", font=dict(color=theme["text"])),
            tickfont=dict(color=theme["text"]),
            gridcolor=theme["grid"],
            zerolinecolor=theme["grid"],
            tickmode="array",
            tickvals=epochs,
            range=x_range,
            autorange=False
        ),
        yaxis=dict(
            title=dict(text=y_axis_title, font=dict(color=theme["text"])),
            tickfont=dict(color=theme["text"]),
            gridcolor=theme["grid"],
            zerolinecolor=theme["grid"],
            range=y_range,
            autorange=False
        ),
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme["text"])
        ),
        margin=dict(l=55, r=25, t=70, b=100),
        height=440,
        uirevision=f"stable-{title}",
        sliders=[
            {
                "active": 0,
                "currentvalue": {
                    "prefix": "Epoch: ",
                    "font": {"color": theme["text"]}
                },
                "font": {"color": theme["text"]},
                "pad": {"t": 35},
                "steps": slider_steps
            }
        ],
        transition={"duration": 0}
    )

    # Convert the Plotly figure, including frames, into JSON.
    figure_json = fig.to_json().replace("</", "<\\/")

    # Theme-aware player colours.
    button_bg = "#087FC9" if app_theme == "Light" else "#123D5C"
    button_text = "#FFFFFF" if app_theme == "Light" else "#E8F5FF"
    status_text = "#17324B" if app_theme == "Light" else "#E8F5FF"
    border_color = "#DCE6EF" if app_theme == "Light" else "#29445F"

    html_template = r'''
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
      <style>
        * { box-sizing: border-box; }

        body {
          margin: 0;
          padding: 0;
          background: transparent;
          font-family: Arial, sans-serif;
        }

        #chart {
          width: 100%;
          height: 440px;
        }

        .player {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 12px;
          padding: 10px 0 8px;
        }

        #toggle {
          width: 48px;
          height: 48px;
          border-radius: 50%;
          border: 1px solid BORDER_COLOR;
          background: BUTTON_BG;
          color: BUTTON_TEXT;
          font-size: 21px;
          cursor: pointer;
          display: flex;
          justify-content: center;
          align-items: center;
          transition: transform .18s ease;
        }

        #toggle:hover {
          transform: scale(1.08);
        }

        #toggle:active {
          transform: scale(.94);
        }

        #status {
          color: STATUS_TEXT;
          font-size: 13px;
          font-weight: 600;
          opacity: 1;
        }

        .extra-controls {
          display: flex;
          justify-content: center;
          align-items: center;
          flex-wrap: wrap;
          gap: 9px;
          padding: 5px 0 10px;
        }

        .extra-controls button {
          background: BUTTON_BG;
          color: BUTTON_TEXT;
          border: 1px solid BORDER_COLOR;
          border-radius: 9px;
          padding: 9px 14px;
          font-size: 13px;
          font-weight: 600;
          cursor: pointer;
          transition: opacity .15s ease, transform .15s ease;
        }

        .extra-controls button:hover {
          opacity: .82;
          transform: translateY(-1px);
        }

        .extra-controls button:active {
          transform: scale(.97);
        }

        #toggle {
          color: BUTTON_TEXT !important;
          font-weight: 700;
          opacity: 1;
        }
      </style>
    </head>

    <body>
      <div id="chart"></div>

      <div class="player">
        <button id="toggle" title="Play" aria-label="Play animation">
          ▶
        </button>
        <span id="status">Paused</span>
      </div>

      <div class="extra-controls">
        <button id="refreshGraph">↻ Refresh Graph</button>
        <button id="clearSelection">⌫ Clear Selection</button>
        <button id="normalMode">↖ Normal Mode</button>
      </div>

      <script>
        const figure = FIGURE_JSON;
        const chart = document.getElementById("chart");
        const button = document.getElementById("toggle");
        const status = document.getElementById("status");
        const refreshButton = document.getElementById("refreshGraph");
        const clearButton = document.getElementById("clearSelection");
        const normalButton = document.getElementById("normalMode");

        let isPlaying = false;
        let animationRun = 0;

        // Preserve untouched chart data for every refresh.
        const originalData = JSON.parse(JSON.stringify(figure.data));
        const originalLayout = JSON.parse(JSON.stringify(figure.layout));
        const originalFrames = JSON.parse(JSON.stringify(figure.frames || []));

        function updateButton(playing) {
          isPlaying = playing;

          button.textContent = playing ? "Ⅱ" : "▶";
          button.title = playing ? "Pause" : "Play";
          button.setAttribute(
            "aria-label",
            playing ? "Pause animation" : "Play animation"
          );

          button.style.setProperty("color", "BUTTON_TEXT", "important");
          button.style.setProperty(
            "-webkit-text-fill-color", "BUTTON_TEXT", "important"
          );
          button.style.setProperty("opacity", "1", "important");

          status.textContent = playing ? "Playing" : "Paused";

          // Ensure status text remains readable in Light and Dark themes.
          status.style.setProperty("color", "STATUS_TEXT", "important");
          status.style.setProperty("-webkit-text-fill-color", "STATUS_TEXT", "important");
          status.style.setProperty("font-weight", "700", "important");
          status.style.setProperty("opacity", "1", "important");
          status.style.setProperty("text-shadow", "none", "important");
        }

        function stopAnimation() {
          animationRun++;

          Plotly.animate(chart, [null], {
            mode: "immediate",
            frame: {duration: 0, redraw: false},
            transition: {duration: 0}
          });

          updateButton(false);
        }

        function startAnimation() {
          const currentRun = ++animationRun;

          updateButton(true);

          Plotly.animate(chart, null, {
            fromcurrent: true,
            frame: {
              duration: 450,
              redraw: false
            },
            transition: {
              duration: 0
            }
          }).then(() => {
            if (currentRun === animationRun) {
              updateButton(false);
            }
          }).catch(error => {
            if (currentRun === animationRun) {
              updateButton(false);
            }
            console.error("Animation error:", error);
          });
        }

        Plotly.newPlot(
          chart,
          JSON.parse(JSON.stringify(originalData)),
          JSON.parse(JSON.stringify(originalLayout)),
          {
            responsive: true,
            displaylogo: false,
            modeBarButtonsToRemove: ["zoom2d"]
          }
        ).then(() => {
          if (originalFrames.length) {
            return Plotly.addFrames(
              chart,
              JSON.parse(JSON.stringify(originalFrames))
            );
          }
        }).then(() => {
          updateButton(false);
          button.disabled = false;
        }).catch(error => {
          button.disabled = true;
          status.textContent = "Chart loading error";
          console.error(error);
        });

        button.addEventListener("click", () => {
          if (isPlaying) {
            stopAnimation();
          } else {
            startAnimation();
          }
        });

        // Clear selected points on every trace and remove persistent selections.
        clearButton.addEventListener("click", async () => {
          if (isPlaying) stopAnimation();

          try {
            for (let i = 0; i < chart.data.length; i++) {
              await Plotly.restyle(
                chart, {selectedpoints: null}, [i]
              );
            }

            await Plotly.relayout(chart, {
              selections: [],
              dragmode: false
            });

            if (Plotly.Fx && Plotly.Fx.unhover) {
              Plotly.Fx.unhover(chart);
            }
          } catch (error) {
            console.error("Clear selection error:", error);
          }
        });

        // Exit Pan, Box Select or Lasso Select without resetting zoom.
        normalButton.addEventListener("click", () => {
          Plotly.relayout(chart, {dragmode: false});
        });

        // Refresh only this graph, preserving the rest of the page.
        refreshButton.addEventListener("click", async () => {
          if (isPlaying) stopAnimation();

          refreshButton.disabled = true;
          refreshButton.textContent = "↻ Refreshing...";

          try {
            Plotly.purge(chart);

            await Plotly.newPlot(
              chart,
              JSON.parse(JSON.stringify(originalData)),
              JSON.parse(JSON.stringify(originalLayout)),
              {
                responsive: true,
                displaylogo: false,
                modeBarButtonsToRemove: ["zoom2d"]
              }
            );

            if (originalFrames.length) {
              await Plotly.addFrames(
                chart,
                JSON.parse(JSON.stringify(originalFrames))
              );
            }

            updateButton(false);
            status.textContent = "Paused";
          } catch (error) {
            console.error("Graph refresh error:", error);
            status.textContent = "Refresh failed";
          } finally {
            refreshButton.disabled = false;
            refreshButton.textContent = "↻ Refresh Graph";
          }
        });
      </script>
    </body>
    </html>
    '''

    html_content = (
        html_template
        .replace("FIGURE_JSON", figure_json)
        .replace("BUTTON_BG", button_bg)
        .replace("BUTTON_TEXT", button_text)
        .replace("STATUS_TEXT", status_text)
        .replace("BORDER_COLOR", border_color)
    )

    components.html(
        html_content,
        height=565,
        scrolling=False
    )


def render_threshold_chart(csv_path):
    """Display interactive threshold comparison using actual CSV values."""
    if not csv_path.exists():
        st.warning("Threshold comparison data not found.")
        return

    df = pd.read_csv(csv_path)

    required = [
        "Threshold", "Dice", "IoU", "Precision", "Sensitivity"
    ]

    if not all(column in df.columns for column in required):
        st.error("Required threshold comparison columns are missing.")
        return

    theme = get_chart_theme()
    fig = go.Figure()

    metric_colors = {
        "Dice": theme["colors"][0],
        "IoU": theme["colors"][1],
        "Precision": theme["colors"][2],
        "Sensitivity": theme["colors"][3]
    }

    for metric, color in metric_colors.items():
        fig.add_trace(
            go.Scatter(
                x=df["Threshold"],
                y=df[metric],
                mode="lines+markers",
                name=metric,
                line=dict(color=color, width=3),
                marker=dict(size=8),
                hovertemplate=(
                    "Threshold: %{x:.2f}<br>"
                    + metric + ": %{y:.4f}<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title=dict(
            text="Model Performance Across Thresholds",
            font=dict(size=21, color=theme["text"])
        ),
        template=theme["template"],
        paper_bgcolor=theme["paper"],
        plot_bgcolor=theme["plot"],
        font=dict(color=theme["text"], size=13),
        xaxis=dict(
            title=dict(text="Segmentation Threshold", font=dict(color=theme["text"])),
            tickfont=dict(color=theme["text"]),
            gridcolor=theme["grid"]
        ),
        yaxis=dict(
            title=dict(text="Metric Score", font=dict(color=theme["text"])),
            tickfont=dict(color=theme["text"]),
            gridcolor=theme["grid"]
        ),
        hovermode="x unified",
        legend=dict(font=dict(color=theme["text"])),
        margin=dict(l=45, r=25, t=65, b=55),
        height=460
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"responsive": True, "displaylogo": False}
    )


# NEUROVISION REUSABLE THEME-AWARE NOTIFICATIONS
def render_notice(message, kind="info"):
    """Render a readable notification in the selected application theme."""
    from html import escape

    light_palette = {
        "info":    ("#E8F4FC", "#2B8CC4", "#174A70", "ℹ"),
        "success": ("#E5F7EF", "#27A66D", "#145C3B", "✓"),
        "warning": ("#FFF5DE", "#D99A20", "#69440B", "⚠"),
        "error":   ("#FDECEC", "#D94A4A", "#8B2525", "✕"),
    }

    dark_palette = {
        "info":    ("#102D49", "#278FC4", "#D8EEFF", "ℹ"),
        "success": ("#103B30", "#27B985", "#C9F7E6", "✓"),
        "warning": ("#3A2C13", "#D99A20", "#FFE6B0", "⚠"),
        "error":   ("#401E24", "#E45B65", "#FFD8DC", "✕"),
    }

    palette = light_palette if app_theme == "Light" else dark_palette
    background, border, foreground, icon = palette.get(
        kind, palette["info"]
    )

    safe_message = escape(str(message))
    role = "alert" if kind in ("warning", "error") else "status"

    st.markdown(
        f"""
        <div role="{role}" style="
            background:{background};
            border:1px solid {border};
            border-left:6px solid {border};
            border-radius:10px;
            padding:15px 19px;
            margin:12px 0;
            color:{foreground};
            font-size:15px;
            font-weight:600;
            line-height:1.6;
            opacity:1;
        ">
            <span style="color:{foreground};">
                {icon}&nbsp; {safe_message}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


if selected_page == "Dashboard":

    # Futuristic animated dashboard header
    st.markdown(
        """
        <div class="neuro-hero-title">
            🧠 NEUROVISION AI
        </div>
        <div class="neuro-hero-subtitle">
            AI-powered Brain MRI Analysis | U-Net Segmentation
        </div>
        <br>
        <div class="neuro-status">
            <span class="neuro-status-dot"></span>
            AI SYSTEM INTERFACE ACTIVE
        </div>
        <div class="neuro-scan-panel">
            <div class="neuro-scan-line"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
        A deep learning-based academic research project
        designed to segment suspected tumour regions
        from brain FLAIR MRI images.
        """
    )

    st.divider()

    # Dataset and model overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "MRI Images",
            "3,929"
        )

    with col2:
        st.metric(
            "Unique Patients",
            "110"
        )

    with col3:
        st.metric(
            "Image Resolution",
            "256 × 256"
        )

    with col4:
        st.metric(
            "Model Architecture",
            "U-Net"
        )

    st.divider()

    st.subheader("Project Overview")

    card1, card2 = st.columns(2)

    with card1:

        st.markdown(
            """
            <div class="info-card">
                <h3>🎯 Project Objective</h3>
                <p>
                Develop a deep learning model to identify
                and segment suspected tumour regions
                from brain MRI images.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with card2:

        st.markdown(
            """
            <div class="info-card">
                <h3>⚙️ Technology Stack</h3>
                <p>
                Python, TensorFlow, U-Net, OpenCV,
                NumPy, Pandas, Matplotlib and Streamlit.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("Dataset Distribution")

    split1, split2, split3 = st.columns(3)

    with split1:
        st.metric(
            "Training Images",
            "3,151"
        )

    with split2:
        st.metric(
            "Validation Images",
            "447"
        )

    with split3:
        st.metric(
            "Testing Images",
            "331"
        )

    render_notice(
        "Patient-level splitting was used to avoid "
        "patient overlap between training, validation "
        "and testing datasets.",
        kind="info"
    )

    st.divider()

    st.subheader("How the System Works")

    st.markdown(
        """
        **01 — Upload MRI**

        Upload a compatible brain MRI image.

        **02 — Preprocessing**

        The image is resized and normalized.

        **03 — U-Net Prediction**

        The trained model generates a pixel-wise
        probability map.

        **04 — Segmentation**

        A threshold converts probabilities into a
        binary segmentation mask.

        **05 — Visualization**

        The predicted region is displayed on the MRI
        image using a red overlay.
        """
    )


# ============================================================
# 11. TUMOUR SEGMENTATION PAGE
# ============================================================

elif selected_page == "Tumour Segmentation":

    # Futuristic MRI analysis header
    st.markdown(
        """
        <div class="neuro-segmentation-header">
            <div class="neuro-segmentation-title">
                🧠 MRI ANALYSIS CENTER
            </div>
            <div class="neuro-segmentation-subtitle">
                Deep Learning | U-Net | Brain MRI Segmentation
            </div>
            <div class="neuro-analysis-status">
                <span class="neuro-analysis-dot"></span>
                SEGMENTATION ENGINE READY
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "Upload a brain MRI image to generate "
        "a predicted tumour segmentation mask."
    )

    st.divider()

    # Load trained model
    try:

        with st.spinner(
            "Loading trained U-Net model..."
        ):

            model = load_unet_model()

        render_notice(
            "U-Net model loaded successfully!",
            kind="success"
        )

    except Exception as error:

        st.error(
            f"Unable to load model: {error}"
        )

        st.stop()

    # Upload MRI image
    uploaded_file = st.file_uploader(
        "Upload Brain MRI Image",
        type=[
            "tif",
            "tiff",
            "png",
            "jpg",
            "jpeg"
        ],
        key="mri_uploader"
    )

    # Select segmentation threshold
    default_threshold = load_threshold()

    threshold = st.slider(
        "Segmentation Threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(default_threshold),
        step=0.05,
        help="Adjust the probability threshold used to create the binary mask."
    )

    if uploaded_file is not None:

        try:

            image_bytes = uploaded_file.getvalue()

            # Display uploaded MRI
            original_image = Image.open(
                io.BytesIO(image_bytes)
            )

            st.subheader("Uploaded MRI Image")

            st.image(
                original_image,
                caption=uploaded_file.name,
                width=400
            )

            # Generate segmentation
            if st.button(
                "Generate Segmentation",
                type="primary",
                use_container_width=False
            ):

                with st.spinner(
                    "Analysing MRI image..."
                ):

                    # Preprocess MRI
                    display_mri, model_input = preprocess_mri(
                        image_bytes
                    )

                    # Generate model prediction
                    prediction = model.predict(
                        model_input,
                        verbose=0
                    )[0, :, :, 0]

                    # Create binary mask
                    predicted_mask = (
                        prediction >= threshold
                    ).astype(np.uint8)

                    # Create overlay
                    final_overlay = create_overlay(
                        display_mri,
                        predicted_mask
                    )

                    # Calculate predicted pixel area
                    tumour_pixels = int(
                        np.sum(predicted_mask)
                    )

                    total_pixels = IMG_SIZE * IMG_SIZE

                    tumour_percentage = (
                        tumour_pixels / total_pixels
                    ) * 100

                    # Save prediction results in Streamlit session.
                    # Original MRI bytes are retained for persistent storage.
                    st.session_state["segmentation_result"] = {
                        "filename": uploaded_file.name,
                        "original_image_bytes": image_bytes,
                        "display_mri": display_mri,
                        "predicted_mask": predicted_mask,
                        "overlay": final_overlay,
                        "probability_map": prediction,
                        "tumour_pixels": tumour_pixels,
                        "tumour_percentage": tumour_percentage,
                        "threshold": threshold
                    }

                    # Generate the PDF once from the existing prediction.
                    pdf_bytes, report_id = generate_segmentation_report(
                        filename=uploaded_file.name,
                        display_mri=display_mri,
                        predicted_mask=predicted_mask,
                        overlay=final_overlay,
                        tumour_pixels=tumour_pixels,
                        tumour_percentage=tumour_percentage,
                        threshold=threshold
                    )

                    # Keep the PDF and report ID for subsequent reruns.
                    st.session_state["segmentation_result"]["pdf_bytes"] = pdf_bytes
                    st.session_state["segmentation_result"]["report_id"] = report_id

                    # Save the completed session to Google Drive and SQLite.
                    try:
                        saved_session = save_segmentation_session(
                            filename=uploaded_file.name,
                            original_image_bytes=image_bytes,
                            display_mri=display_mri,
                            predicted_mask=predicted_mask,
                            overlay=final_overlay,
                            threshold=threshold,
                            tumour_pixels=tumour_pixels,
                            tumour_percentage=tumour_percentage,
                            pdf_bytes=pdf_bytes,
                            report_id=report_id
                        )

                        st.session_state["segmentation_result"]["storage"] = saved_session

                    except Exception as storage_error:
                        # Preserve prediction and PDF even if storage fails.
                        st.session_state["segmentation_result"]["storage_error"] = str(storage_error)


                st.success(
                    "Segmentation prediction completed!"
                )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )

    # Display prediction results
    result = st.session_state.get(
        "segmentation_result"
    )

    if result is not None:

        # Ensure results correspond to the currently uploaded file
        if (
            uploaded_file is None
            or result["filename"] != uploaded_file.name
        ):

            st.info(
                "Upload the MRI image and generate a new "
                "prediction to display its results."
            )

        else:

            st.divider()

            st.subheader("Segmentation Results")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.image(
                    result["display_mri"],
                    caption="Original MRI",
                    use_container_width=True
                )

            with col2:

                st.image(
                    result["predicted_mask"] * 255,
                    caption="Predicted Tumour Mask",
                    clamp=True,
                    use_container_width=True
                )

            with col3:

                st.image(
                    result["overlay"],
                    caption="MRI with Tumour Overlay",
                    use_container_width=True
                )

            st.divider()

            st.subheader("Prediction Information")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:

                st.metric(
                    "Predicted Tumour Pixels",
                    f"{result['tumour_pixels']:,}"
                )

            with metric2:

                st.metric(
                    "Predicted Area",
                    f"{result['tumour_percentage']:.2f}%"
                )

            with metric3:

                st.metric(
                    "Threshold",
                    f"{result['threshold']:.2f}"
                )

            if result["tumour_pixels"] > 0:

                st.warning(
                    "The model predicted a segmented region. "
                    "This is an algorithmic output, not a medical diagnosis."
                )

            else:

                st.info(
                    "The model did not predict a tumour region "
                    "at the selected threshold."
                )

            st.divider()

            st.subheader("Download Results")

            download1, download2 = st.columns(2)

            with download1:

                mask_bytes = image_to_bytes(
                    result["predicted_mask"] * 255
                )

                st.download_button(
                    label="Download Tumour Mask",
                    data=mask_bytes,
                    file_name="predicted_tumour_mask.png",
                    mime="image/png",
                    use_container_width=True
                )

            with download2:

                overlay_bytes = image_to_bytes(
                    result["overlay"]
                )

                st.download_button(
                    label="Download MRI Overlay",
                    data=overlay_bytes,
                    file_name="mri_tumour_overlay.png",
                    mime="image/png",
                    use_container_width=True
                )

            st.caption(
                "Predicted area is the percentage of image pixels "
                "classified as tumour. It is not a clinical tumour "
                "volume measurement."
            )

            # Reuse the PDF generated when this prediction completed.
            pdf_bytes = result["pdf_bytes"]
            report_id = result["report_id"]

            report_filename = (
                f"neurovision_report_"
                f"{Path(result['filename']).stem}.pdf"
            )

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=report_filename,
                mime="application/pdf",
                use_container_width=True,
                key="download_segmentation_pdf"
            )

            st.caption(
                f"Report ID: {report_id} | "
                "PDF includes MRI, predicted mask, overlay and "
                "segmentation statistics."
            )

            if result.get("storage"):
                st.success(
                    "Session saved successfully to SQLite and Google Drive."
                )
                st.caption(
                    f"Session ID: {result['storage']['session_id']}"
                )

            elif result.get("storage_error"):
                st.warning(
                    "Prediction and PDF are available, but persistent "
                    "session storage failed: "
                    + result["storage_error"]
                )



    st.divider()

    render_notice(
        "Academic research prototype only. This application is "
        "not intended for clinical diagnosis or treatment decisions.",
        kind="warning"
    )


# ============================================================
# 12. MODEL PERFORMANCE PAGE
# ============================================================

elif selected_page == "Model Performance":

    # Futuristic model performance header
    st.markdown(
        """
        <div class="neuro-performance-header">
            <div class="neuro-performance-title">
                📊 MODEL PERFORMANCE CENTER
            </div>
            <div class="neuro-performance-subtitle">
                U-Net Evaluation | Test Dataset | Deep Learning Analytics
            </div>
            <div class="neuro-performance-status">
                <span class="neuro-performance-dot"></span>
                MODEL EVALUATION REPORT
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "Evaluation results of the trained U-Net model "
        "on the independent test dataset."
    )

    st.divider()

    st.subheader("Final Test Set Metrics")

    # Recorded test-set metrics
    metric_values = [
        ("Dice Score", "0.7799"),
        ("IoU Score", "0.6392"),
        ("Precision", "0.7442"),
        ("Sensitivity", "0.8191"),
        ("Specificity", "0.9963"),
        ("Positive-Slice Mean Dice", "0.6973")
    ]

    metric_rows = [
        st.columns(3),
        st.columns(3)
    ]

    for index, (label, value) in enumerate(metric_values):

        row_index = index // 3
        column_index = index % 3

        with metric_rows[row_index][column_index]:

            st.metric(
                label,
                value
            )

    render_notice(
        "These test-set metrics were calculated using "
        "the selected binary threshold of 0.60.",
        kind="info"
    )

    st.divider()

    # Display saved detailed metrics when available
    if METRICS_PATH.exists():

        st.subheader("Saved Evaluation Results")

        try:

            metrics_df = pd.read_csv(
                METRICS_PATH
            )

            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:

            st.warning(
                f"Could not read metrics CSV: {error}"
            )

    # Training loss graph
    st.divider()

    st.subheader("Training and Validation Loss")

    if LOSS_GRAPH.exists():

        render_animated_training_chart(
            TRAINING_LOG_PATH,
            ["loss", "val_loss"],
            ["Training Loss", "Validation Loss"],
            "U-Net Training and Validation Loss",
            "Loss"
        )

    else:

        st.warning(
            "Training loss graph not found."
        )

    # Training Dice graph
    st.subheader("Training and Validation Dice")

    if DICE_GRAPH.exists():

        render_animated_training_chart(
            TRAINING_LOG_PATH,
            ["dice_coefficient", "val_dice_coefficient"],
            ["Training Dice", "Validation Dice"],
            "U-Net Training and Validation Dice",
            "Dice Score"
        )

    else:

        st.warning(
            "Training Dice graph not found."
        )

    # Threshold comparison graph
    st.subheader("Threshold Comparison")

    if THRESHOLD_GRAPH.exists():

        render_threshold_chart(
            RESULTS_DIR / "threshold_comparison.csv"
        )

    else:

        st.warning(
            "Threshold comparison graph not found."
        )

    # Threshold comparison CSV
    threshold_csv = RESULTS_DIR / "threshold_comparison.csv"

    if threshold_csv.exists():

        st.subheader("Threshold Comparison Data")

        try:

            threshold_df = pd.read_csv(
                threshold_csv
            )

            st.dataframe(
                threshold_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:

            st.warning(
                f"Could not read threshold data: {error}"
            )

    st.divider()

    st.subheader("Evaluation Metrics Explained")

    st.markdown(
        """
        - **Dice Score:** Measures overlap between predicted
          and ground-truth tumour regions.

        - **IoU:** Measures intersection divided by union.

        - **Precision:** Measures the proportion of predicted
          tumour pixels that are correct.

        - **Sensitivity:** Measures the proportion of actual
          tumour pixels detected by the model.

        - **Specificity:** Measures the proportion of background
          pixels correctly identified.
        """
    )


# ============================================================
# 13. ABOUT PROJECT PAGE
# ============================================================

elif selected_page == "About Project":

    # ============================================================
    # ABOUT PROJECT - NEUROVISION AI
    # ============================================================

    # Futuristic About Project header - matching Model Performance
    st.markdown(
        """
        <div class="neuro-performance-header">
            <div class="neuro-performance-title">
                🧠 ABOUT NEUROVISION AI
            </div>
            <div class="neuro-performance-subtitle">
                Brain Tumour Segmentation | U-Net | Deep Learning Research
            </div>
            <div class="neuro-performance-status">
                <span class="neuro-performance-dot"></span>
                PROJECT OVERVIEW
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "NeuroVision AI is an academic deep learning project "
        "designed to segment suspected tumour regions from "
        "FLAIR brain MRI images using a U-Net architecture. "
        "The application supports MRI upload, segmentation "
        "prediction and visualization of model outputs."
    )

    st.divider()

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # PROJECT AT A GLANCE
    # ------------------------------------------------------------

    st.markdown(
        """
        <style>
        .st-key-about_project_overview [data-testid="stMetric"] {
            background: linear-gradient(
                145deg,
                rgba(12, 48, 78, 0.96),
                rgba(7, 27, 49, 0.98)
            );
            border: 1px solid rgba(18, 185, 214, 0.42);
            border-radius: 15px;
            padding: 20px 16px;
            min-height: 112px;
            box-shadow: 0 5px 20px rgba(0, 160, 220, 0.10);
            transition: all 0.25s ease;
        }

        .st-key-about_project_overview
        [data-testid="stMetric"]:hover {
            border-color: #12B9D6;
            box-shadow: 0 0 20px rgba(18, 185, 214, 0.22);
            transform: translateY(-3px);
        }

        .st-key-about_project_overview
        [data-testid="stMetricLabel"] {
            color: #A8D8EF !important;
            font-weight: 600;
        }

        .st-key-about_project_overview
        [data-testid="stMetricValue"] {
            color: #EAFBFF !important;
            font-size: 1.8rem;
            font-weight: 800;
        }

        [data-theme="light"]
        .st-key-about_project_overview
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, #FFFFFF, #EAF7FC);
            border: 1px solid #A8DDEA;
            box-shadow: 0 5px 18px rgba(20, 100, 150, 0.08);
        }

        [data-theme="light"]
        .st-key-about_project_overview
        [data-testid="stMetricLabel"] {
            color: #42677F !important;
        }

        [data-theme="light"]
        .st-key-about_project_overview
        [data-testid="stMetricValue"] {
            color: #123B59 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container(key="about_project_overview"):

        st.subheader("Project at a Glance")

        overview_cols = st.columns(4)

        overview_items = [
            ("3,929", "MRI Images"),
            ("3,929", "Tumour Masks"),
            ("110", "Unique Patients"),
            ("256 × 256", "Input Resolution")
        ]

        for col, (value, label) in zip(overview_cols, overview_items):
            with col:
                st.metric(label=label, value=value)

    st.divider()

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # PROJECT OBJECTIVES
    # ------------------------------------------------------------

    st.markdown(
        """
        <style>
        /* PROJECT OBJECTIVES - FUTURISTIC GLASS CARDS */

        .st-key-about_project_objectives
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(
                145deg,
                rgba(12, 48, 78, 0.92),
                rgba(7, 27, 49, 0.98)
            );
            border: 1px solid rgba(18, 185, 214, 0.38) !important;
            border-radius: 15px !important;
            box-shadow: 0 5px 20px rgba(0, 160, 220, 0.08);
            padding: 10px 12px;
            transition: all 0.25s ease;
        }

        .st-key-about_project_objectives
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: #12B9D6 !important;
            box-shadow: 0 0 20px rgba(18, 185, 214, 0.18);
            transform: translateY(-3px);
        }

        .st-key-about_project_objectives
        [data-testid="stMarkdownContainer"] p {
            color: #BBD3E6;
            line-height: 1.7;
        }

        .st-key-about_project_objectives strong {
            color: #75E8FF;
            font-size: 1.02rem;
            font-weight: 750;
        }

        /* LIGHT THEME */

        [data-theme="light"]
        .st-key-about_project_objectives
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(
                145deg,
                #FFFFFF,
                #EDF8FC
            );
            border: 1px solid #A8DDEA !important;
            box-shadow: 0 5px 18px rgba(20, 100, 150, 0.07);
        }

        [data-theme="light"]
        .st-key-about_project_objectives
        [data-testid="stMarkdownContainer"] p {
            color: #42677F;
        }

        [data-theme="light"]
        .st-key-about_project_objectives strong {
            color: #087FA5;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container(key="about_project_objectives"):

        st.subheader("Project Objectives")

        objective_cols = st.columns(2)

        objectives = [
            (
                "01",
                "Data Collection",
                "Collect brain MRI images and their corresponding masks."
            ),
            (
                "02",
                "Image Preprocessing",
                "Prepare MRI images by resizing and normalizing pixel values."
            ),
            (
                "03",
                "U-Net Development",
                "Develop a segmentation model using encoder and decoder blocks."
            ),
            (
                "04",
                "Model Training",
                "Train the model using labelled MRI images and masks."
            ),
            (
                "05",
                "Performance Evaluation",
                "Evaluate segmentation quality using standard metrics."
            ),
            (
                "06",
                "Application Development",
                "Build a Streamlit interface for image upload and prediction."
            )
        ]

        for i, (number, heading, description) in enumerate(objectives):
            with objective_cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"**{number} | {heading}**")
                    st.write(description)

    st.divider()

    # ------------------------------------------------------------
    # DATASET INFORMATION
    # ------------------------------------------------------------

    st.subheader("Dataset Information")

    dataset_col1, dataset_col2 = st.columns(2)

    with dataset_col1:
        with st.container(border=True):
            st.markdown("#### Dataset Details")
            st.write("**Dataset:** LGG MRI Segmentation Dataset")
            st.write("**MRI Modality:** FLAIR")
            st.write("**MRI Images:** 3,929")
            st.write("**Tumour Masks:** 3,929")
            st.write("**Unique Patients:** 110")

    with dataset_col2:
        with st.container(border=True):
            st.markdown("#### Image Processing")
            st.write("**Input Resolution:** 256 × 256")
            st.write("**Image Channels:** 1")
            st.write("**Normalization:** 0 to 1")
            st.write("**Mask Format:** Binary")
            st.write("**Segmentation Model:** U-Net")

    st.markdown("#### Dataset Distribution")

    split_df = pd.DataFrame({
        "Dataset": ["Training", "Validation", "Testing"],
        "Images": [3151, 447, 331]
    })

    st.bar_chart(
        split_df.set_index("Dataset"),
        y="Images",
        height=280
    )

    st.caption(
        "The dataset was divided at patient level to avoid "
        "patient overlap between training, validation and testing."
    )

    st.divider()

    # ------------------------------------------------------------
    # PROJECT METHODOLOGY
    # ------------------------------------------------------------

    st.subheader("Project Methodology")

    methodology = [
        (
            "01. Data Collection",
            "Collect MRI images and their corresponding ground-truth masks."
        ),
        (
            "02. Preprocessing",
            "Resize images to 256 × 256 and normalize pixel values between 0 and 1."
        ),
        (
            "03. U-Net Architecture",
            "Use encoder, decoder and skip connections for image segmentation."
        ),
        (
            "04. Model Training",
            "Train the network and monitor validation performance."
        ),
        (
            "05. Model Evaluation",
            "Measure performance using Dice, IoU, Precision, Sensitivity and Specificity."
        ),
        (
            "06. Web Application",
            "Use Streamlit to upload MRI images and visualize segmentation results."
        )
    ]

    for i in range(0, len(methodology), 3):
        method_cols = st.columns(3)

        for col, (heading, description) in zip(
            method_cols, methodology[i:i + 3]
        ):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{heading}**")
                    st.write(description)

    st.divider()

    # ------------------------------------------------------------
    # TECHNOLOGY STACK
    # ------------------------------------------------------------

    st.subheader("Technologies Used")

    technology_cols = st.columns(3)

    technology_groups = [
        (
            "Programming",
            ["Python", "NumPy", "Pandas"]
        ),
        (
            "Deep Learning & Image Processing",
            ["TensorFlow", "U-Net", "OpenCV"]
        ),
        (
            "Application & Development",
            ["Streamlit", "Matplotlib", "Google Colab"]
        )
    ]

    for col, (heading, technologies) in zip(
        technology_cols, technology_groups
    ):
        with col:
            with st.container(border=True):
                st.markdown(f"**{heading}**")
                for technology in technologies:
                    st.markdown(f"- {technology}")

    st.divider()

    # ------------------------------------------------------------
    # MODEL PERFORMANCE SUMMARY
    # ------------------------------------------------------------

    st.subheader("Model Performance Summary")

    st.caption(
        "Recorded test-set segmentation results from the current project."
    )

    performance_cols = st.columns(5)

    performance_metrics = [
        ("Dice Score", "0.7799"),
        ("IoU Score", "0.6392"),
        ("Precision", "0.7442"),
        ("Sensitivity", "0.8191"),
        ("Specificity", "0.9963")
    ]

    for col, (label, value) in zip(
        performance_cols, performance_metrics
    ):
        with col:
            st.metric(label=label, value=value)

    st.caption(
        "These metrics describe the model's recorded test-set performance. "
        "They do not establish clinical accuracy or diagnostic suitability."
    )

    st.divider()

    # ------------------------------------------------------------
    # LIMITATIONS AND DISCLAIMER
    # ------------------------------------------------------------

    st.subheader("Project Limitations")

    limitations_col1, limitations_col2 = st.columns(2)

    with limitations_col1:
        st.markdown(
            """
            - The current model uses FLAIR MRI images only.
            - Prediction quality depends on image quality.
            - Performance may vary on images unlike the training data.
            """
        )

    with limitations_col2:
        st.markdown(
            """
            - The model has not been clinically validated.
            - The project is an academic research prototype.
            - Predictions require expert medical interpretation.
            """
        )

    render_notice(
        "This system is intended for academic research "
        "and educational demonstration only. It must not "
        "be used as a medical diagnostic tool.",
        kind="warning"
    )



# ============================================================
# SESSION HISTORY PAGE
# ============================================================

elif selected_page == "Session History":

    st.markdown(
        """
        <div class="neuro-performance-header">
            <div class="neuro-performance-title">
                📁 SEGMENTATION SESSION HISTORY
            </div>
            <div class="neuro-performance-subtitle">
                Saved MRI Results | SQLite Database | Report Archive
            </div>
            <div class="neuro-performance-status">
                <span class="neuro-performance-dot"></span>
                PERSISTENT SESSION ARCHIVE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Retrieve saved records from SQLite.
    try:
        history_records = get_segmentation_history(limit=1000)

    except Exception as error:
        st.error(f"Unable to load session history: {error}")
        history_records = []

    st.subheader("Saved Segmentation Sessions")

    st.metric(
        "Total Saved Sessions",
        len(history_records)
    )

    # Search saved sessions.
    search_query = st.text_input(
        "Search by MRI filename or Session ID",
        placeholder="Enter filename or session ID..."
    ).strip().lower()

    filtered_records = [
        record for record in history_records
        if (
            search_query in record["original_filename"].lower()
            or search_query in record["session_id"].lower()
            or search_query in record["report_id"].lower()
        )
    ]

    st.caption(
        f"Showing {len(filtered_records)} of "
        f"{len(history_records)} saved sessions."
    )

    if not filtered_records:

        if history_records:
            st.info("No sessions match your search.")
        else:
            st.info(
                "No saved segmentation sessions are available yet. "
                "Generate a segmentation to create your first record."
            )

    else:

        for index, record in enumerate(filtered_records):

            session_id = record["session_id"]

            with st.expander(
                f"{record['original_filename']} | "
                f"{record['created_at'][:19]} | "
                f"Session {session_id[:8]}",
                expanded=(index == 0)
            ):

                st.markdown("**Session Information**")

                info_col1, info_col2 = st.columns(2)

                with info_col1:
                    st.caption("Session ID")
                    st.code(session_id)

                    st.caption("Report ID")
                    st.write(record["report_id"])

                    st.caption("Original MRI")
                    st.write(record["original_filename"])

                with info_col2:
                    st.caption("Created At")
                    st.write(record["created_at"])

                    st.caption("Threshold")
                    st.write(f"{record['threshold']:.2f}")

                    st.caption("Predicted Tumour Pixels")
                    st.write(f"{record['tumour_pixels']:,}")

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric(
                        "Predicted Area",
                        f"{record['tumour_percentage']:.2f}%"
                    )

                with metric_col2:
                    st.metric(
                        "Total Image Pixels",
                        f"{record['total_pixels']:,}"
                    )

                st.markdown("**Saved Segmentation Images**")

                image_paths = [
                    record["original_path"],
                    record["mask_path"],
                    record["overlay_path"]
                ]

                image_captions = [
                    "Original MRI",
                    "Predicted Tumour Mask",
                    "MRI with Tumour Overlay"
                ]

                image_columns = st.columns(3)

                for column, image_path, caption in zip(
                    image_columns,
                    image_paths,
                    image_captions
                ):
                    with column:
                        if Path(image_path).is_file():
                            st.image(
                                str(image_path),
                                caption=caption,
                                use_container_width=True
                            )
                        else:
                            st.warning(f"{caption} file is missing.")

                st.markdown("**Download Archived Report**")

                pdf_path = Path(record["report_path"])

                if pdf_path.is_file():

                    pdf_bytes = pdf_path.read_bytes()

                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=(
                            f"neurovision_report_"
                            f"{Path(record['original_filename']).stem}.pdf"
                        ),
                        mime="application/pdf",
                        key=f"history_pdf_{session_id}",
                        use_container_width=True
                    )

                else:
                    st.error("The archived PDF file is missing.")

    st.divider()

    render_notice(
        "Session history is stored locally in the project Google Drive "
        "folder using SQLite and saved image files. "
        "This archive is intended for academic research only.",
        kind="info"
    )



# 14. APPLICATION FOOTER
# ============================================================

st.divider()

st.markdown(
    "<h3 style='text-align:center; color:#55A8FF;'>NeuroVision AI</h3>",
    unsafe_allow_html=True
)

st.caption("Brain Tumour Segmentation Using U-Net")

st.caption("Medical AI Research | Deep Learning")

st.caption("Academic Research Prototype")

# END OF APPLICATION
# ============================================================

# NEUROVISION AI FINAL ACCESSIBILITY OVERRIDES


# ============================================================
# NEUROVISION AI FINAL ACCESSIBILITY OVERRIDES
# ============================================================

if app_theme == "Light":
    alert_text_color = "#17324B"
    uploader_text_color = "#17324B"
else:
    alert_text_color = "#FFFFFF"
    uploader_text_color = "#EAF5FF"

st.markdown(
    f"""
    <style>

    /* Final alert text contrast */
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] strong {{
        color: {alert_text_color} !important;
        -webkit-text-fill-color: {alert_text_color} !important;
        opacity: 1 !important;
    }}

    /* Alert icons */
    div[data-testid="stAlert"] svg {{
        color: {alert_text_color} !important;
        fill: currentColor !important;
    }}

    /* Upload instructions and filename */
    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] small {{
        color: {uploader_text_color} !important;
        -webkit-text-fill-color: {uploader_text_color} !important;
        opacity: 1 !important;
    }}

    /* Upload button */
    div[data-testid="stFileUploader"] button {{
        background: #087DB5 !important;
        color: #FFFFFF !important;
        border: 1px solid #087DB5 !important;
        opacity: 1 !important;
    }}

    div[data-testid="stFileUploader"] button *,
    div[data-testid="stFileUploader"] button svg {{
        color: #FFFFFF !important;
        fill: currentColor !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# NEUROVISION AI LIGHT TYPOGRAPHY FINAL FIX


# ============================================================
# NEUROVISION AI LIGHT TYPOGRAPHY FINAL FIX
# ============================================================

if app_theme == "Light":

    st.markdown(
        """
        <style>

        /* Main application text */
        .stApp,
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp span,
        .stApp small,
        .stApp div {
            color: #17324B;
        }

        /* Main headings */
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {
            color: #123B59 !important;
            text-shadow: none !important;
            opacity: 1 !important;
        }

        /* Markdown content */
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li,
        .stApp [data-testid="stMarkdownContainer"] strong,
        .stApp [data-testid="stMarkdownContainer"] em {
            color: #17324B !important;
            opacity: 1 !important;
        }

        /* Captions and secondary text */
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stCaptionContainer"] p,
        .stApp .stCaption {
            color: #405B73 !important;
            opacity: 1 !important;
        }

        /* Widget labels */
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp [data-testid="stWidgetLabel"] label {
            color: #17324B !important;
            opacity: 1 !important;
        }

        /* Input values */
        .stApp input,
        .stApp textarea,
        .stApp select {
            color: #17324B !important;
            -webkit-text-fill-color: #17324B !important;
        }

        /* Selectbox and multiselect text */
        .stApp [data-baseweb="select"] *,
        .stApp [data-baseweb="input"] * {
            color: #17324B !important;
        }

        /* Slider labels and values */
        .stApp [data-testid="stSlider"] *,
        .stApp [data-testid="stNumberInput"] *,
        .stApp [data-testid="stTextInput"] * {
            color: #17324B !important;
        }

        /* Metric labels and values */
        .stApp [data-testid="stMetricLabel"],
        .stApp [data-testid="stMetricLabel"] *,
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stMetricValue"] * {
            color: #123B59 !important;
        }

        /* Custom information cards */
        .stApp .info-card,
        .stApp .info-card p,
        .stApp .info-card li,
        .stApp .info-card span,
        .stApp .info-card strong {
            color: #17324B !important;
            text-shadow: none !important;
        }

        /* Custom section headings */
        .stApp .neuro-section-title,
        .stApp .neuro-result-heading {
            color: #087DB5 !important;
            text-shadow: none !important;
        }

        /* Evaluation notes */
        .stApp .neuro-evaluation-note {
            color: #17324B !important;
        }

        /* Footer */
        .stApp [data-testid="stBottomBlockContainer"] p,
        .stApp [data-testid="stBottomBlockContainer"] h3 {
            color: #405B73 !important;
        }

        /* Tables */
        .stApp [data-testid="stDataFrame"] {
            border: 1px solid #C9DDEB !important;
            border-radius: 10px !important;
        }

        /* Remove faded text appearance */
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp span {
            -webkit-text-fill-color: currentColor;
        }

        </style>
        """,
        unsafe_allow_html=True
    )



# NEUROVISION LIGHT THEME ALERT CONTRAST FIX
if app_theme == "Light":
    st.markdown("""
    <style>
    /* Success, warning, info and error alert text */
    .stApp div[data-testid="stAlert"] p,
    .stApp div[data-testid="stAlert"] span,
    .stApp div[data-testid="stAlert"] label,
    .stApp div[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    .stApp div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] * {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }

    /* Preserve alert backgrounds; improve text and icon visibility */
    .stApp div[data-testid="stAlert"] svg {
        color: #FFFFFF !important;
        fill: currentColor !important;
    }

    /* File uploader instructions and file details */
    .stApp [data-testid="stFileUploaderDropzone"] p,
    .stApp [data-testid="stFileUploaderDropzone"] span,
    .stApp [data-testid="stFileUploaderDropzone"] small,
    .stApp [data-testid="stFileUploaderDropzone"] label,
    .stApp [data-testid="stFileUploaderFile"] p,
    .stApp [data-testid="stFileUploaderFile"] span {
        color: #EAF5FF !important;
        opacity: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)
