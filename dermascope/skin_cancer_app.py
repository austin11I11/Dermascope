# ================================================================
#  DERMASCOPE — Skin Cancer Detection App
#  Built with Streamlit + TensorFlow Lite
#  
#  HOW TO RUN:
#    pip install -r requirements.txt
#    streamlit run skin_cancer_app.py
#
#  FILES NEEDED IN SAME FOLDER:
#    skin_cancer_model_f32_features.tflite   (float32 export with two outputs: class
#                                             probabilities and pooled features, from training/ood)
#    ood_bank.npy, ood_threshold.json        (out-of-distribution gate, from training/ood)
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

# The model has two outputs, told apart by their size
NUM_CLASSES = 7      # class probabilities
FEATURE_DIM = 1280   # pooled features, used by the out-of-distribution gate

# Occlusion sensitivity: a 56px patch at stride 28 gives a 7×7 grid (49 positions)
OCC_PATCH  = 56
OCC_STRIDE = 28
# Fixed color scale: a confidence drop this large is full brightness
# (1.00 = the patch removed all of the model's confidence)
OCC_FULL_SCALE = 1.00
# Overlay opacity ramps from 0 at no drop up to its maximum at this drop
OCC_OPACITY_FULL = 0.10
# Below this, the largest confidence drop is treated as "no region stands out"
OCC_MIN_DROP = 0.05

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

def find_model_path():
    """Path to the TFLite model, or None if it can't be found."""
    model_path = os.path.join(os.path.dirname(__file__), "skin_cancer_model_f32_features.tflite")

    # Fallback: try current directory
    if not os.path.exists(model_path):
        for root, dirs, files in os.walk('.'):
            for f in files:
                if f.endswith('.tflite'):
                    model_path = os.path.join(root, f)
                    break

    return model_path if os.path.exists(model_path) else None


@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    """Load TFLite model and class labels. Cached so it only loads once."""
    model_path  = find_model_path()
    labels_path = os.path.join(os.path.dirname(__file__), "class_labels.json")

    if model_path is None:
        return None, None

    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    with open(labels_path) as f:
        labels = json.load(f)  # {str(idx): class_short_name}

    return interpreter, labels


@st.cache_resource(show_spinner=False)
def load_ood_gate():
    """
    Load the out-of-distribution gate: the bank of training-image features
    and the score threshold. Cached so it only loads once.
    Returns (bank, threshold, k), or None if a file is missing.
    """
    bank_path      = os.path.join(os.path.dirname(__file__), "ood_bank.npy")
    threshold_path = os.path.join(os.path.dirname(__file__), "ood_threshold.json")
    if not (os.path.exists(bank_path) and os.path.exists(threshold_path)):
        return None

    bank = np.load(bank_path).astype(np.float32)  # saved as float16, (3000, 1280)
    with open(threshold_path) as f:
        info = json.load(f)
    return bank, float(info["threshold"]), int(info["k"])


def preprocess_image(img: Image.Image) -> np.ndarray:
    """PIL Image → float32 array in [0, 255], resized the way the training pipeline does."""
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = arr.astype(np.float32)
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def run_inference(interpreter, input_data: np.ndarray):
    """
    Run TFLite inference. Returns (probabilities of shape (7,),
    pooled features of shape (1280,)).
    """
    input_details = interpreter.get_input_details()
    # The converter does not promise an output order, so go by size
    outputs = {int(d['shape'][-1]): d['index'] for d in interpreter.get_output_details()}

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    probs    = interpreter.get_tensor(outputs[NUM_CLASSES])[0]
    features = interpreter.get_tensor(outputs[FEATURE_DIM])[0]
    return probs, features


def ood_score(features: np.ndarray, bank: np.ndarray, k: int) -> float:
    """
    Out-of-distribution score: cosine distance from this image's pooled
    features to its k-th nearest neighbor in the bank of training images.
    Higher means less like the dermatoscopic images the model was trained on.
    Same computation as training/ood/ood_gate.ipynb.
    """
    query = features / np.linalg.norm(features)
    similarities = bank @ query
    kth_largest = np.partition(similarities, -k)[-k]
    return float(1.0 - kth_largest)


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


@st.cache_data(show_spinner=False, max_entries=32)
def compute_occlusion_map(input_data: np.ndarray, class_idx: int) -> np.ndarray:
    """
    Occlusion sensitivity for one preprocessed image, using the app's model.
    A mean-color patch is slid over the model input; each cell of the returned
    7×7 map is how much the probability of class_idx dropped with the patch
    at that position. Cached per image (keyed on input_data and class_idx).
    """
    # input_data shape: (1, 224, 224, 3), already preprocessed
    mean_color = input_data[0].mean(axis=(0, 1))
    positions  = range(0, IMG_SIZE - OCC_PATCH + 1, OCC_STRIDE)
    n = len(positions)

    # One batch: every occluded copy, plus the untouched image as the baseline
    batch = np.repeat(input_data, n * n + 1, axis=0)
    for row, y in enumerate(positions):
        for col, x in enumerate(positions):
            batch[row * n + col, y:y + OCC_PATCH, x:x + OCC_PATCH] = mean_color

    # A fresh interpreter sized for the batch, used only by this call. The
    # shared cached interpreter is never resized, so simultaneous users
    # can't collide.
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=find_model_path())
    input_index = interpreter.get_input_details()[0]['index']
    interpreter.resize_tensor_input(input_index, list(batch.shape))
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, batch)
    interpreter.invoke()
    outputs = {int(d['shape'][-1]): d['index'] for d in interpreter.get_output_details()}
    output = interpreter.get_tensor(outputs[NUM_CLASSES])

    probs = output[:, class_idx]
    return (probs[-1] - probs[:-1]).reshape(n, n)


def make_occlusion_overlay(img: Image.Image, drop_map: np.ndarray,
                           alpha: float = 0.55) -> Image.Image:
    """
    Upsample the occlusion map to the image size and blend it over the image.
    alpha is the maximum opacity; zero-drop areas are left untouched.
    """
    # Only drops count; a patch that raised the confidence shows as zero.
    # Fixed scale, the same for every image: a drop of OCC_FULL_SCALE is full
    # brightness, so brightness is comparable from one image to the next.
    heat = np.clip(drop_map / OCC_FULL_SCALE, 0, 1)

    # Each cell belongs to the center of its patch, so upsample onto the area
    # the patch centers span and replicate the edge out to the full input size.
    n    = heat.shape[0]
    pad  = (IMG_SIZE - n * OCC_STRIDE) // 2
    heat = cv2.resize(heat.astype(np.float32), (n * OCC_STRIDE, n * OCC_STRIDE),
                      interpolation=cv2.INTER_LINEAR)
    heat = cv2.copyMakeBorder(heat, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    original = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w     = original.shape[:2]
    heat     = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)

    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(heat, 0, 1)), cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay: opacity ramps up to alpha at a drop of OCC_OPACITY_FULL and
    # stays there, so zero-drop areas show the original image unchanged and
    # any meaningful hotspot is clearly visible. Color carries the drop size.
    opacity = alpha * np.clip(heat * OCC_FULL_SCALE / OCC_OPACITY_FULL, 0, 1)
    opacity = opacity[..., np.newaxis]
    overlay = (original * (1 - opacity) + heatmap * opacity).astype(np.uint8)
    return Image.fromarray(overlay)


def occlusion_summary(drop_map: np.ndarray) -> str:
    """One line under the map: the largest drop, or that no region stands out."""
    max_drop = float(drop_map.max())
    if max_drop < OCC_MIN_DROP:
        return "No single region strongly changed the prediction"
    return f"Largest confidence drop: {max_drop:.0%}"


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

    st.markdown("**Test-set performance**")
    st.markdown("""
    Held-out HAM10000 test set (1,484 images):

    - Macro AUC: **0.936**
    - Accuracy: **78.0%**
    - Melanoma recall: **59%**
    """)

    st.markdown("**Known limitations**")
    st.markdown("""
    - About 41% of test melanomas were missed.
    - HAM10000 contains mostly lighter skin tones, so performance on
      darker skin is unverified.
    - The image check wrongly rejects about 27% of vascular lesions.
    - The image check also wrongly rejects about 4% of melanomas.
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
    ood_gate = load_ood_gate()

if interpreter is None or ood_gate is None:
    st.error("""
    **Model files not found.**

    Make sure `skin_cancer_model_f32_features.tflite`, `ood_bank.npy`,
    `ood_threshold.json` and `class_labels.json` are in the same folder
    as this app.

    `training/ood/ood_gate.ipynb` generates the first three; **Cell 20** of
    the model notebook generates the labels.
    """)
    st.stop()

ood_bank, ood_threshold, ood_k = ood_gate

# True when the out-of-distribution gate rejects the uploaded image
rejected = False

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
    st.caption("Designed for dermatoscopic images (taken through a dermatoscope) only.")

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
            probs, features = run_inference(interpreter, input_data)

            # Out-of-distribution gate, checked before anything is shown
            rejected = ood_score(features, ood_bank, ood_k) > ood_threshold

            progress.progress(100)
            time.sleep(0.1)
            progress.empty()

    if uploaded and rejected:
        st.markdown("""
        <div class="result-card">
            <div class="result-label">Image Not Classified</div>
            <div class="result-sub" style="font-size:0.95rem; color:#b0a99a; line-height:1.7;">
                This doesn’t look like a dermatoscopic skin-lesion image, so
                DermaScope won’t classify it. The model was trained only on
                dermatoscope images. Regular phone photos of skin aren’t supported.
                If you’re concerned about a skin spot, see a dermatologist.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded:
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
# SECTION 3 — Occlusion Sensitivity
# ================================================================

if uploaded and not rejected:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 — Occlusion Sensitivity</div>',
                unsafe_allow_html=True)

    cam_col1, cam_col2, cam_col3 = st.columns(3)

    with cam_col1:
        st.image(img_pil, caption="Original", use_container_width=True)

    with cam_col2:
        with st.spinner("Computing occlusion sensitivity..."):
            drop_map = compute_occlusion_map(input_data, top_idx)
        overlay = make_occlusion_overlay(img_pil, drop_map, alpha=0.55)
        st.image(overlay, caption="Occlusion Sensitivity", use_container_width=True)
        st.markdown(f"""
        <p style='font-family:"DM Mono",monospace; font-size:0.75rem;
           color:#9a9089; text-align:center;'>{occlusion_summary(drop_map)}</p>
        <div style='height:6px; border-radius:2px; margin-top:0.6rem;
             background:linear-gradient(90deg, #000004, #420a68, #932667,
             #dd513a, #fca50a, #fcffa4);'></div>
        <div style='display:flex; justify-content:space-between;
             font-family:"DM Mono",monospace; font-size:0.68rem; color:#6b6560;
             margin-top:0.25rem;'>
            <span>0%</span><span>{OCC_FULL_SCALE:.0%}</span>
        </div>
        <p style='font-size:0.72rem; color:#6b6560; margin-top:0.3rem; line-height:1.6;'>
        Brightness shows how many percentage points the model’s confidence fell
        when that region was covered. Full brightness is a
        {OCC_FULL_SCALE * 100:.0f}-point drop. A prediction can’t fall further
        than its own confidence, so less confident predictions never reach full
        brightness. The scale is the same for every image.
        </p>
        """, unsafe_allow_html=True)

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
    Occlusion sensitivity: brighter regions are where covering the image most
    reduced the model’s confidence in its prediction. Coarse 7×7 grid; it shows
    what the prediction depends on, not a medical rationale.
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

            This model has not been evaluated by skin tone, so its
            performance on darker skin is unverified.

            **Planned next step for this project:** Quantifying this accuracy
            gap and exploring augmentation strategies to improve equity. This
            analysis has not been done yet.
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

if uploaded and not rejected:
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
