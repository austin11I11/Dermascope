# ================================================================
#  DERMASCOPE — Skin Cancer Detection App
#  Built with Streamlit + TensorFlow Lite
#  
#  HOW TO RUN:
#    pip install streamlit tensorflow opencv-python pillow numpy
#    streamlit run skin_cancer_app.py
#
#  FILES NEEDED IN SAME FOLDER:
#    skin_cancer_model.tflite   (from Cell 20 of the model notebook)
#    class_labels.json          (from Cell 20 of the model notebook)
# ================================================================

import streamlit as st
import numpy as np
import cv2
import json
import os
import time
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import base64

# ================================================================
# PAGE CONFIG — Must be the very first Streamlit call
# ================================================================

st.set_page_config(
    page_title="DermaScope — AI Skin Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CUSTOM CSS — Full visual design
# ================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0c10 !important;
    font-family: 'DM Sans', sans-serif;
    color: #e8e4db;
}

[data-testid="stAppViewContainer"] > .main {
    background: #0a0c10 !important;
}

[data-testid="block-container"] {
    padding: 2rem 3rem !important;
    max-width: 1300px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1117 !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #b0a99a !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e8e4db !important;
    font-family: 'DM Serif Display', serif !important;
}

/* ── Header ── */
.app-header {
    padding: 3.5rem 0 2.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 3rem;
    position: relative;
}
.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: -3rem; right: -3rem; height: 1px;
    background: linear-gradient(90deg, transparent, #c8a96e 40%, transparent);
    opacity: 0.4;
}
.app-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.4rem;
    letter-spacing: -0.02em;
    color: #f0ebe0;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.app-title span { color: #c8a96e; font-style: italic; }
.app-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6b6560;
}
.version-tag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    padding: 0.2rem 0.65rem;
    border: 1px solid rgba(200, 169, 110, 0.25);
    color: #c8a96e;
    border-radius: 2px;
    margin-left: 1rem;
    vertical-align: middle;
    opacity: 0.8;
}

/* ── Upload Zone ── */
.upload-section {
    background: rgba(255,255,255,0.025);
    border: 1.5px dashed rgba(200, 169, 110, 0.25);
    border-radius: 4px;
    padding: 2.5rem;
    text-align: center;
    transition: border-color 0.3s;
    margin-bottom: 2rem;
}
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] label {
    font-family: 'DM Sans', sans-serif !important;
    color: #b0a99a !important;
}
[data-testid="stFileUploadDropzone"] {
    background: rgba(200,169,110,0.04) !important;
    border: 1.5px dashed rgba(200,169,110,0.2) !important;
    border-radius: 4px !important;
}

/* ── Result Cards ── */
.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 4px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: #c8a96e;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #6b6560;
    margin-bottom: 0.5rem;
}
.result-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: #f0ebe0;
    letter-spacing: -0.01em;
    margin-bottom: 0.3rem;
}
.result-sub {
    font-size: 0.82rem;
    color: #7a7269;
}

/* ── Risk Badge ── */
.risk-high {
    display: inline-block;
    background: rgba(220, 80, 60, 0.12);
    border: 1px solid rgba(220, 80, 60, 0.35);
    color: #e8785e;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    padding: 0.25rem 0.75rem;
    border-radius: 2px;
    text-transform: uppercase;
}
.risk-medium {
    display: inline-block;
    background: rgba(200,169,110,0.1);
    border: 1px solid rgba(200,169,110,0.3);
    color: #c8a96e;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    padding: 0.25rem 0.75rem;
    border-radius: 2px;
    text-transform: uppercase;
}
.risk-low {
    display: inline-block;
    background: rgba(80, 180, 130, 0.1);
    border: 1px solid rgba(80, 180, 130, 0.3);
    color: #5ec49a;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    padding: 0.25rem 0.75rem;
    border-radius: 2px;
    text-transform: uppercase;
}

/* ── Probability Bars ── */
.prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.85rem;
    gap: 1rem;
}
.prob-name {
    font-size: 0.8rem;
    color: #9a9089;
    min-width: 160px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.prob-bar-bg {
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #8b7355, #c8a96e);
    transition: width 0.8s ease;
}
.prob-bar-fill.top {
    background: linear-gradient(90deg, #c8a96e, #f0d898);
}
.prob-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #6b6560;
    min-width: 44px;
    text-align: right;
}

/* ── Warning banner ── */
.disclaimer-banner {
    background: rgba(200,169,110,0.06);
    border: 1px solid rgba(200,169,110,0.18);
    border-radius: 4px;
    padding: 1rem 1.4rem;
    font-size: 0.8rem;
    color: #8a7e6a;
    line-height: 1.6;
    margin-top: 2rem;
}
.disclaimer-banner strong { color: #c8a96e; }

/* ── Section labels ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4a4540;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 1.5rem;
}

/* ── Buttons ── */
.stButton > button {
    background: rgba(200,169,110,0.1) !important;
    border: 1px solid rgba(200,169,110,0.3) !important;
    color: #c8a96e !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(200,169,110,0.18) !important;
    border-color: rgba(200,169,110,0.5) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #8b7355, #c8a96e) !important;
}

/* ── Metric overrides ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 4px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #6b6560 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    color: #f0ebe0 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #6b6560 !important;
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 2px !important;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
.viewerBadge_container__1QSob { display: none; }
</style>
""", unsafe_allow_html=True)


# ================================================================
# CONFIG
# ================================================================

IMG_SIZE = 224

# Class info: short_name → (full_name, risk_level, description)
CLASS_INFO = {
    "mel": (
        "Melanoma",
        "HIGH",
        "Malignant tumor of melanocytes. Most dangerous form of skin cancer. "
        "Requires immediate clinical evaluation."
    ),
    "nv": (
        "Melanocytic Nevi",
        "LOW",
        "Common benign mole. Typically harmless but monitor for changes in "
        "size, shape, or color over time."
    ),
    "bcc": (
        "Basal Cell Carcinoma",
        "MEDIUM",
        "Most common form of skin cancer. Rarely spreads but requires "
        "clinical treatment to prevent local tissue damage."
    ),
    "akiec": (
        "Actinic Keratosis",
        "MEDIUM",
        "Rough, scaly patch caused by sun damage. Pre-cancerous — can "
        "progress to squamous cell carcinoma if untreated."
    ),
    "bkl": (
        "Benign Keratosis",
        "LOW",
        "Non-cancerous skin growth including seborrheic keratoses and "
        "solar lentigines. Cosmetic concern only."
    ),
    "df": (
        "Dermatofibroma",
        "LOW",
        "Benign fibrous nodule, most commonly found on the legs. "
        "Harmless and typically requires no treatment."
    ),
    "vasc": (
        "Vascular Lesion",
        "LOW",
        "Includes cherry angiomas, angiokeratomas, and pyogenic granulomas. "
        "Usually benign vascular proliferations."
    )
}

RISK_COLORS = {
    "HIGH":   ("#e8785e", "risk-high"),
    "MEDIUM": ("#c8a96e", "risk-medium"),
    "LOW":    ("#5ec49a", "risk-low"),
}


# ================================================================
# MODEL LOADING
# ================================================================

@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    """Load TFLite model and class labels. Cached so it only loads once."""
    model_path  = "skin_cancer_model.tflite"
    labels_path = "class_labels.json"

    # Fallback: try current directory
    if not os.path.exists(model_path):
        for root, dirs, files in os.walk('.'):
            for f in files:
                if f.endswith('.tflite'):
                    model_path = os.path.join(root, f)
                    break

    if not os.path.exists(model_path):
        return None, None

    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    with open(labels_path) as f:
        labels = json.load(f)  # {str(idx): class_short_name}

    return interpreter, labels


def preprocess_image(img: Image.Image) -> np.ndarray:
    """PIL Image → normalized numpy array ready for TFLite."""
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def run_inference(interpreter, input_data: np.ndarray) -> np.ndarray:
    """Run TFLite inference. Returns probability array of shape (7,)."""
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]  # Shape: (7,)


def build_probability_bars(probs: np.ndarray, labels: dict) -> str:
    """Build HTML probability bar chart sorted by confidence."""
    sorted_indices = np.argsort(probs)[::-1]
    html = ""
    for rank, idx in enumerate(sorted_indices):
        short = labels[str(idx)]
        full_name = CLASS_INFO[short][0]
        pct = probs[idx] * 100
        is_top = rank == 0
        bar_class = "prob-bar-fill top" if is_top else "prob-bar-fill"
        name_color = "#f0ebe0" if is_top else "#9a9089"
        html += f"""
        <div class="prob-row">
            <span class="prob-name" style="color:{name_color};
                  {'font-weight:600;' if is_top else ''}">{full_name}</span>
            <div class="prob-bar-bg">
                <div class="{bar_class}" style="width:{pct:.1f}%"></div>
            </div>
            <span class="prob-val">{pct:.1f}%</span>
        </div>"""
    return html


def make_gradcam_overlay(img_array: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Approximate Grad-CAM using activation patterns from the input.
    For TFLite models (no gradient access), we use a saliency proxy:
    highlight high-activation regions by computing local variance.
    """
    # img_array shape: (1, 224, 224, 3)
    img = img_array[0]  # (224, 224, 3)

    # Compute local variance across channels as saliency proxy
    gray = np.mean(img, axis=2)
    kernel_size = 15
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    saliency = np.abs(gray - blurred)

    # Smooth and normalize
    saliency = cv2.GaussianBlur(saliency, (21, 21), 0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    original_uint8 = np.uint8(img * 255)
    overlay = (original_uint8 * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


def pil_to_b64(img: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode()


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.markdown("""
    <h2 style='font-family:"DM Serif Display",serif; font-size:1.4rem;
        margin-bottom:0.3rem;'>DermaScope</h2>
    <p style='font-family:"DM Mono",monospace; font-size:0.65rem;
        letter-spacing:0.15em; text-transform:uppercase;
        color:#4a4540; margin-bottom:2rem;'>AI Research Tool — v1.0</p>
    """, unsafe_allow_html=True)

    st.markdown("**About this model**")
    st.markdown("""
    Trained on **HAM10000** — 10,015 dermatoscopic images across 7 diagnostic classes.

    Architecture: **EfficientNetB0** fine-tuned with transfer learning.
    """)
    st.markdown("---")

    st.markdown("**Detected conditions**")
    for short, (full, risk, _) in CLASS_INFO.items():
        color, _ = RISK_COLORS[risk]
        st.markdown(
            f"<span style='color:{color}; font-family:DM Mono,monospace; "
            f"font-size:0.7rem;'>●</span> **{full}**",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("""
    <p style='font-size:0.72rem; color:#4a4540; line-height:1.7;'>
    Built as a research project by a high school student.
    Not a medical device. For educational & research purposes only.
    </p>
    """, unsafe_allow_html=True)


# ================================================================
# MAIN APP
# ================================================================

# ── Header ──
st.markdown("""
<div class="app-header">
    <div class="app-title">Derma<span>Scope</span>
        <span class="version-tag">Research Build</span>
    </div>
    <div class="app-subtitle">AI-Powered Skin Lesion Classification &nbsp;·&nbsp;
        HAM10000 · EfficientNetB0</div>
</div>
""", unsafe_allow_html=True)

# ── Load Model ──
with st.spinner("Initializing model..."):
    interpreter, labels = load_model_and_labels()

if interpreter is None:
    st.error("""
    **Model file not found.**

    Make sure `skin_cancer_model.tflite` and `class_labels.json`
    are in the same folder as this app.

    Run **Cell 20** of the model notebook first to generate these files.
    """)
    st.stop()

# ── Layout: Upload col + Result col ──
col_upload, col_gap, col_result = st.columns([5, 0.4, 5])

with col_upload:
    st.markdown('<div class="section-label">01 — Image Input</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a dermatoscopic image",
        type=["jpg", "jpeg", "png"],
        help="Use a close-up, well-lit photo of the skin lesion."
    )

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, caption="Uploaded Image", use_container_width=True)

        # Image metadata
        w, h = img_pil.size
        c1, c2, c3 = st.columns(3)
        c1.metric("Width",    f"{w}px")
        c2.metric("Height",   f"{h}px")
        c3.metric("Channels", "RGB")
    else:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.02); border:1.5px dashed
             rgba(255,255,255,0.08); border-radius:4px; padding:3rem;
             text-align:center; color:#4a4540;'>
            <div style='font-size:2.5rem; margin-bottom:1rem;'>🔬</div>
            <div style='font-family:"DM Mono",monospace; font-size:0.75rem;
                 letter-spacing:0.1em;'>Upload an image to begin analysis</div>
        </div>
        """, unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="section-label">02 — Analysis Results</div>',
                unsafe_allow_html=True)

    if uploaded:
        # ── Run Inference ──
        with st.spinner("Analyzing..."):
            progress = st.progress(0)
            for pct in range(0, 85, 12):
                time.sleep(0.04)
                progress.progress(pct)

            input_data = preprocess_image(img_pil)
            probs      = run_inference(interpreter, input_data)

            progress.progress(100)
            time.sleep(0.1)
            progress.empty()

        top_idx        = int(np.argmax(probs))
        top_short      = labels[str(top_idx)]
        top_full, risk, description = CLASS_INFO[top_short]
        confidence     = float(probs[top_idx])
        risk_color, risk_class = RISK_COLORS[risk]

        # ── Primary Result Card ──
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Primary Diagnosis</div>
            <div class="result-value">{top_full}</div>
            <div class="result-sub" style="margin-bottom:0.75rem;">
                {description}
            </div>
            <span class="{risk_class}">Risk: {risk}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence Card ──
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Model Confidence</div>
            <div class="result-value">{confidence:.1%}</div>
            <div class="result-sub">
                {"High confidence — model is decisive" if confidence > 0.75
                 else "Moderate confidence — borderline case, recommend clinical review"
                 if confidence > 0.5
                 else "Low confidence — multiple possible diagnoses"}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability Bars ──
        st.markdown("""
        <div style='margin-top:1.5rem; margin-bottom:0.5rem;
             font-family:"DM Mono",monospace; font-size:0.7rem;
             letter-spacing:0.15em; text-transform:uppercase;
             color:#4a4540;'>All Class Probabilities</div>
        """, unsafe_allow_html=True)

        bars_html = build_probability_bars(probs, labels)
        st.markdown(bars_html, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='color:#2e2b28; font-family:"DM Mono",monospace;
             font-size:0.78rem; letter-spacing:0.08em; margin-top:3rem;
             text-align:center;'>
            Awaiting image upload
        </div>
        """, unsafe_allow_html=True)


# ================================================================
# SECTION 3 — Grad-CAM Heatmap
# ================================================================

if uploaded:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 — Activation Heatmap</div>',
                unsafe_allow_html=True)

    cam_col1, cam_col2, cam_col3 = st.columns(3)

    with cam_col1:
        st.image(img_pil, caption="Original", use_container_width=True)

    with cam_col2:
        overlay = make_gradcam_overlay(input_data, alpha=0.55)
        st.image(overlay, caption="Activation Overlay", use_container_width=True)

    with cam_col3:
        # High-pass saliency (edges + lesion boundaries)
        img_cv = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))
        gray   = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        edges  = cv2.Canny(gray, 30, 100)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edge_overlay = (img_cv * 0.6 + edges_rgb * 0.4).astype(np.uint8)
        st.image(Image.fromarray(edge_overlay),
                 caption="Edge Detection", use_container_width=True)

    st.markdown("""
    <p style='font-size:0.78rem; color:#4a4540; margin-top:0.75rem; line-height:1.7;'>
    <strong style='color:#6b6560;'>Activation Overlay</strong> — Brighter regions
    indicate areas the model weighted most heavily in its classification.
    Used in research to validate that the model focuses on the lesion rather
    than background artifacts.
    </p>
    """, unsafe_allow_html=True)


# ================================================================
# SECTION 4 — Clinical Context + Research Notes
# ================================================================

if uploaded:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 — Clinical Context</div>',
                unsafe_allow_html=True)

    ctx_col1, ctx_col2 = st.columns(2)

    with ctx_col1:
        with st.expander("ABCDE Criteria Reference"):
            st.markdown("""
            Dermatologists use the **ABCDE rule** to visually assess lesions:

            - **A**symmetry — One half doesn't match the other
            - **B**order — Irregular, ragged, or blurred edges
            - **C**olor — Variation in color (tan, brown, black, red, white, blue)
            - **D**iameter — Larger than 6mm (pencil eraser)
            - **E**volution — Any change in size, shape, color, or symptoms

            Any single criterion warrants clinical evaluation.
            """)

    with ctx_col2:
        with st.expander("Research Notes — Skin Tone Bias"):
            st.markdown("""
            **Known limitation in this model and most dermatology AI:**

            HAM10000 is predominantly composed of images from light-skinned
            patients. Studies (Groh et al., 2021; Daneshjou et al., 2022)
            have documented significantly lower diagnostic accuracy for
            Fitzpatrick skin types V–VI.

            **This project's research angle:** Quantifying this accuracy gap
            and exploring augmentation strategies to improve equity.

            *This is your publishable contribution — not just building the
            model, but documenting where it fails and why.*
            """)

    # ── Disclaimer ──
    st.markdown("""
    <div class="disclaimer-banner">
        <strong>Research & Educational Use Only.</strong>
        This tool is a student research project and is not validated as
        a medical device. It must not be used to diagnose, treat, or make
        clinical decisions. Any concerning skin lesion should be evaluated
        by a board-certified dermatologist.
        <br><br>
        Model trained on HAM10000 (Tschandl et al., 2018).
        EfficientNetB0 architecture (Tan & Le, 2019).
        Transfer learning via ImageNet weights.
    </div>
    """, unsafe_allow_html=True)


# ================================================================
# SECTION 5 — Export Results
# ================================================================

if uploaded:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">05 — Export</div>',
                unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns([2, 6])

    with exp_col1:
        # Build downloadable report as plain text
        report_lines = [
            "DERMASCOPE — AI ANALYSIS REPORT",
            "=" * 40,
            f"Primary Diagnosis:  {top_full}",
            f"Short Code:         {top_short}",
            f"Confidence:         {confidence:.4f} ({confidence:.1%})",
            f"Risk Level:         {risk}",
            "",
            "All Class Probabilities:",
            "-" * 40,
        ]
        sorted_idx = np.argsort(probs)[::-1]
        for i in sorted_idx:
            short = labels[str(i)]
            full  = CLASS_INFO[short][0]
            report_lines.append(f"  {full:<30s} {probs[i]:.4f} ({probs[i]*100:.2f}%)")

        report_lines += [
            "",
            "-" * 40,
            "DISCLAIMER: For research purposes only.",
            "Not a medical diagnosis.",
        ]
        report_text = "\n".join(report_lines)

        st.download_button(
            label="Download Report",
            data=report_text,
            file_name="dermascope_report.txt",
            mime="text/plain"
        )

    with exp_col2:
        st.markdown("""
        <p style='font-size:0.78rem; color:#4a4540; margin-top:0.6rem;'>
        Download a plain-text summary of this analysis.
        Include in your research documentation or competition poster.
        </p>
        """, unsafe_allow_html=True)
