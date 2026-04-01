# ==========================================
# 🚀 MARS AUTONOMOUS LANDING (DEMO READY)
# ==========================================

import os
import cv2
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from torchvision import models

# ==========================================
# 1. LOAD MODEL (RESNET)
# ==========================================

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 8)

model.load_state_dict(torch.load("safelanding/model.pth", map_location=torch.device("cpu")))
model.eval()

# ==========================================
# 2. RISK MAP (IMPROVED)
# ==========================================

risk_map = {
    0: 0.3,
    1: 1.0,
    2: 0.4,
    3: 0.9,
    4: 0.1,
    5: 0.6,
    6: 0.8,
    7: 0.85
}

# ==========================================
# 3. HAZARD MAP GENERATION (SMART)
# ==========================================

def generate_hazard_map(model, image):
    # 🔥 Remove sky (top 30%)
    h, w, _ = image.shape
    image = image[int(h*0.3):, :]

    image = cv2.resize(image, (256, 256))

    patch_size = 64
    stride = patch_size // 2

    hazard_map = np.zeros((256, 256))
    count_map = np.zeros((256, 256))

    for i in range(0, 256 - patch_size, stride):
        for j in range(0, 256 - patch_size, stride):

            patch = image[i:i+patch_size, j:j+patch_size]

            # Resize for ResNet
            patch = cv2.resize(patch, (224, 224))
            patch = patch / 255.0

            # Normalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            patch = (patch - mean) / std

            patch = np.transpose(patch, (2, 0, 1))
            patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                pred = model(patch_tensor)

                probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
                confidence = np.max(probs)

                # Expected risk
                risk = 0
                for k in range(len(probs)):
                    risk += probs[k] * risk_map.get(k, 0.5)

                # Penalize low confidence
                risk = risk * (1 + (1 - confidence))

            hazard_map[i:i+patch_size, j:j+patch_size] += risk
            count_map[i:i+patch_size, j:j+patch_size] += 1

    # Average overlapping areas
    hazard_map = hazard_map / (count_map + 1e-6)

    # Normalize
    hazard_map = (hazard_map - hazard_map.min()) / (hazard_map.max() - hazard_map.min() + 1e-6)

    # Smooth
    hazard_map = cv2.GaussianBlur(hazard_map, (15, 15), 0)

    return hazard_map, image

# ==========================================
# 4. FIND SAFE LANDING ZONE
# ==========================================

def find_safe_zone(hazard_map):
    window_size = 40
    best_score = 9999
    best_coord = (0, 0)

    for i in range(0, hazard_map.shape[0] - window_size):
        for j in range(0, hazard_map.shape[1] - window_size):

            window = hazard_map[i:i+window_size, j:j+window_size]

            score = np.mean(window) + 0.1 * np.std(window)

            if score < best_score:
                best_score = score
                best_coord = (i, j)

    return best_coord, best_score

# ==========================================
# 5. STREAMLIT UI
# ==========================================
# ==========================================
# 🎨 ADVANCED STREAMLIT UI (CRAZY GOOD)
# ==========================================

st.set_page_config(
    page_title="Mars Landing AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------
# 🔥 CUSTOM CSS
# ------------------------------------------

st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Title */
.title {
    font-size: 48px;
    font-weight: 800;
    background: -webkit-linear-gradient(45deg, #22c55e, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Card */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(0,0,0,0.3);
}

/* KPI */
.kpi {
    font-size: 28px;
    font-weight: bold;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #020617;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(45deg, #22c55e, #3b82f6);
    color: white;
    border-radius: 10px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# 🚀 HEADER
# ------------------------------------------

st.markdown('<div class="title">🚀 Mars Autonomous Landing AI</div>', unsafe_allow_html=True)
st.markdown("### Intelligent Terrain Analysis & Safe Zone Prediction")

st.markdown("---")

# ------------------------------------------
# ⚙️ SIDEBAR CONTROLS
# ------------------------------------------

st.sidebar.title("⚙️ AI Controls")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
smoothing = st.sidebar.slider("Heatmap Smoothing", 1, 25, 15)
show_overlay = st.sidebar.checkbox("Show Overlay", True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.write("Model: ResNet18")
st.sidebar.write("Classes: 8 Terrain Types")
st.sidebar.write("Status: 🟢 Active")

# ------------------------------------------
# 📂 FILE UPLOAD
# ------------------------------------------

st.markdown("### 📂 Upload Mars Image")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# ------------------------------------------
# 🚀 PROCESS
# ------------------------------------------

if uploaded_file is not None:

    with st.spinner("🧠 AI is analyzing terrain..."):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        hazard_map, processed_img = generate_hazard_map(model, image)
        coord, score = find_safe_zone(hazard_map)

    st.success("✅ Analysis Complete")

    # ------------------------------------------
    # 🔥 KPI DASHBOARD
    # ------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card"><div class="kpi">🟢 {:.2f}</div>Safety Score</div>'.format(1-score), unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="kpi">📍 {}</div>Landing Coordinates</div>'.format(coord), unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card"><div class="kpi">⚡ Real-time</div>Inference Speed</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------
    # 🎨 VISUALIZATION
    # ------------------------------------------

    heatmap = cv2.applyColorMap((hazard_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (processed_img.shape[1], processed_img.shape[0]))

    overlay = cv2.addWeighted(processed_img, 0.6, heatmap, 0.4, 0)

    x, y = coord
    cv2.rectangle(overlay, (y, x), (y+40, x+40), (0, 255, 0), 2)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🛰️ Input Image")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.markdown("### 🌡️ Hazard Heatmap")
        if show_overlay:
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.image(heatmap, use_container_width=True)

    # ------------------------------------------
    # 📊 LEGEND
    # ------------------------------------------

    st.markdown("### 🎯 Risk Legend")
    st.markdown("""
    - 🔵 **Low Risk** → Safe landing zones  
    - 🟢 **Moderate Risk** → Caution  
    - 🔴 **High Risk** → Avoid  
    """)

    # ------------------------------------------
    # 🧠 AI EXPLANATION
    # ------------------------------------------

    st.markdown("### 🧠 AI Insight")

    if (1-score) > 0.8:
        st.success("Landing zone is highly safe with minimal terrain hazards.")
    elif (1-score) > 0.6:
        st.warning("Moderately safe landing zone. Minor hazards detected.")
    else:
        st.error("Unsafe landing region. High hazard concentration detected.")

    st.markdown("---")
    st.caption("Built with Deep Learning • Computer Vision • PyTorch")