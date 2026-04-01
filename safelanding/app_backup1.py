# ==========================================
# 🚀 MARS LANDING AI (DEPLOYABLE)
# ==========================================

import os
import cv2
import torch
import gdown
import numpy as np
import streamlit as st
from torchvision import models
from torch.utils.data import Dataset
from PIL import Image

# ------------------------------------------
# DATASET (FOR TRAINING COMPATIBILITY)
# ------------------------------------------
class MarsDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        classes = sorted(os.listdir(data_path))

        for label, cls in enumerate(classes):
            path = os.path.join(data_path, cls)
            for img in os.listdir(path):
                self.image_paths.append(os.path.join(path, img))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]

# ------------------------------------------
# DOWNLOAD MODEL (GOOGLE DRIVE)
# ------------------------------------------
MODEL_PATH = "model.pth"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
    gdown.download(url, MODEL_PATH, quiet=False)

# ------------------------------------------
# LOAD MODEL
# ------------------------------------------
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 8)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ------------------------------------------
# RISK MAP
# ------------------------------------------
risk_map = {0:0.3,1:1.0,2:0.4,3:0.9,4:0.1,5:0.6,6:0.8,7:0.85}

# ------------------------------------------
# HAZARD MAP
# ------------------------------------------
def generate_hazard_map(model, image):
    h, w, _ = image.shape
    image = image[int(h*0.3):, :]
    image = cv2.resize(image, (256, 256))

    patch_size = 64
    stride = 32

    hazard_map = np.zeros((256,256))
    count_map = np.zeros((256,256))

    for i in range(0,256-patch_size,stride):
        for j in range(0,256-patch_size,stride):

            patch = image[i:i+patch_size,j:j+patch_size]
            patch = cv2.resize(patch,(224,224))/255.0

            mean = np.array([0.485,0.456,0.406])
            std = np.array([0.229,0.224,0.225])
            patch = (patch-mean)/std

            patch = np.transpose(patch,(2,0,1))
            patch = torch.tensor(patch,dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                pred = model(patch)
                probs = torch.softmax(pred,dim=1).numpy()[0]
                conf = np.max(probs)

                risk = sum(probs[k]*risk_map[k] for k in range(len(probs)))
                risk *= (1+(1-conf))

            hazard_map[i:i+patch_size,j:j+patch_size]+=risk
            count_map[i:i+patch_size,j:j+patch_size]+=1

    hazard_map/= (count_map+1e-6)
    hazard_map = (hazard_map-hazard_map.min())/(hazard_map.max()-hazard_map.min()+1e-6)
    hazard_map = cv2.GaussianBlur(hazard_map,(15,15),0)

    return hazard_map, image

# ------------------------------------------
# SAFE ZONE
# ------------------------------------------
def find_safe_zone(hazard_map):
    best = (0,0)
    best_score = 999

    for i in range(0,200):
        for j in range(0,200):
            win = hazard_map[i:i+40,j:j+40]
            score = np.mean(win)+0.1*np.std(win)

            if score < best_score:
                best_score = score
                best = (i,j)

    return best, best_score

# ------------------------------------------
# UI
# ------------------------------------------
st.set_page_config(page_title="Mars AI", layout="wide")

st.title("🚀 Mars Autonomous Landing AI")

file = st.file_uploader("Upload Image")

if file:
    bytes_data = np.asarray(bytearray(file.read()),dtype=np.uint8)
    img = cv2.imdecode(bytes_data,1)

    hazard_map, processed = generate_hazard_map(model,img)
    coord, score = find_safe_zone(hazard_map)

    heatmap = cv2.applyColorMap((hazard_map*255).astype(np.uint8),cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap,(processed.shape[1],processed.shape[0]))

    overlay = cv2.addWeighted(processed,0.6,heatmap,0.4,0)

    x,y = coord
    cv2.rectangle(overlay,(y,x),(y+40,x+40),(0,255,0),2)

    col1,col2 = st.columns(2)

    col1.image(cv2.cvtColor(processed,cv2.COLOR_BGR2RGB),caption="Input")
    col2.image(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB),caption="AI Heatmap")

    st.success(f"Best Landing Zone: {coord}")
    st.metric("Safety Score", f"{1-score:.2f}")