"""
Streamlit Dashboard for Image Captioning Studio.

Features:
 - Image upload (jpg/jpeg/png only)
 - Toggle YOLO detection
 - Encoder selector: ResNet / ViT / CLIP
 - Decoder selector: LSTM / Transformer / BLIP
 - Generate button
 - Display: caption, confidence, method, inference time, annotated image
"""

import base64
import io
import time

import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000/api/v1"
CAPTION_URL = f"{API_BASE}/caption"
HEALTH_URL = f"{API_BASE}/health"

ALLOWED_TYPES = ["image/jpeg", "image/png"]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Image Captioning Studio",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🖼️ Image Captioning Studio")
st.caption("Modular captioning with YOLO · ResNet/ViT/CLIP · LSTM/Transformer/BLIP")

# ---------------------------------------------------------------------------
# Sidebar: controls
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    help="Strictly JPG / JPEG / PNG formats only.",
)

st.sidebar.divider()

use_yolo = st.sidebar.toggle(
    "🔍 Enable YOLO Object Detection",
    value=False,
    help="Runs YOLOv8n to detect objects and optionally fuse region features.",
)

encoder_choice = st.sidebar.radio(
    "🧠 Feature Extraction (Encoder)",
    options=["resnet", "vit", "clip"],
    format_func=lambda x: {
        "resnet": "ResNet50 (CNN)",
        "vit": "Vision Transformer (ViT)",
        "clip": "CLIP Vision Encoder",
    }[x],
    index=0,
    help="Encoder is skipped when BLIP decoder is selected (BLIP is self-contained).",
)

decoder_choice = st.sidebar.radio(
    "📝 Caption Generator (Decoder)",
    options=["lstm", "transformer", "blip"],
    format_func=lambda x: {
        "lstm": "LSTM + Attention",
        "transformer": "Transformer Decoder",
        "blip": "BLIP (pretrained, recommended)",
    }[x],
    index=2,
)

st.sidebar.divider()
generate_btn = st.sidebar.button("🚀 Generate Caption", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Backend health check banner
# ---------------------------------------------------------------------------

try:
    r = requests.get(HEALTH_URL, timeout=3)
    if r.ok:
        st.sidebar.success("✅ Backend connected", icon="🟢")
    else:
        st.sidebar.error("⚠️ Backend unreachable")
except requests.exceptions.ConnectionError:
    st.sidebar.error("❌ Cannot reach backend – start FastAPI first.")

# ---------------------------------------------------------------------------
# Main area: image preview
# ---------------------------------------------------------------------------

col_img, col_results = st.columns([1, 1], gap="large")

with col_img:
    st.subheader("📷 Input Image")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{uploaded_file.name} ({image.size[0]}×{image.size[1]})", use_column_width=True)
    else:
        st.info("Upload an image using the sidebar to get started.")

# ---------------------------------------------------------------------------
# Generate caption on button press
# ---------------------------------------------------------------------------

with col_results:
    st.subheader("📋 Results")

    if generate_btn:
        if uploaded_file is None:
            st.error("Please upload an image first.")
        else:
            with st.spinner("Running inference …"):
                # Reset file pointer
                uploaded_file.seek(0)

                try:
                    response = requests.post(
                        CAPTION_URL,
                        files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
                        data={
                            "encoder": encoder_choice,
                            "decoder": decoder_choice,
                            "use_yolo": str(use_yolo).lower(),
                        },
                        timeout=120,
                    )
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Is FastAPI running?")
                    st.stop()
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Model loading may take a while on first run.")
                    st.stop()

            if response.status_code == 200:
                data = response.json()

                # ---- Caption ----
                st.success("Caption generated successfully!")
                st.markdown(
                    f"""
                    <div style="background:#1e1e2e;border-radius:10px;padding:20px;margin-bottom:10px">
                        <h3 style="color:#cdd6f4;margin:0 0 8px">📣 Caption</h3>
                        <p style="font-size:18px;color:#a6e3a1;margin:0"><em>"{data['caption']}"</em></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ---- Metrics ----
                m1, m2, m3 = st.columns(3)
                m1.metric("🎯 Confidence", f"{data['confidence']:.2%}")
                m2.metric("⏱️ Inference", f"{data['inference_time_ms']:.0f} ms")
                m3.metric("🔧 Decoder", data["decoder_used"].upper())

                st.markdown(
                    f"**Encoder used:** `{data['encoder_used']}` &nbsp;|&nbsp; "
                    f"**YOLO:** `{'enabled' if data['yolo_enabled'] else 'disabled'}`"
                )

                # ---- Detected objects ----
                if data.get("detected_objects"):
                    st.subheader("🔎 Detected Objects")
                    obj_table = [
                        {"Label": o["label"], "Confidence": f"{o['confidence']:.2%}"}
                        for o in data["detected_objects"]
                    ]
                    st.table(obj_table)

                # ---- Annotated image ----
                if data.get("annotated_image_b64"):
                    st.subheader("🖼️ Annotated Image (YOLO)")
                    img_bytes = base64.b64decode(data["annotated_image_b64"])
                    annotated = Image.open(io.BytesIO(img_bytes))
                    st.image(annotated, use_column_width=True)

            else:
                try:
                    err = response.json()
                    st.error(f"API Error {response.status_code}: {err.get('detail', 'Unknown error')}")
                except Exception:
                    st.error(f"API Error {response.status_code}: {response.text}")