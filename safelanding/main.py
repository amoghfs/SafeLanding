# ==========================================
# 🚀 TRAINING SCRIPT (FINAL)
# ==========================================

import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models
from SafeLanding.safelanding.app_backup1 import MarsDataset  # uses same dataset

# ------------------------------------------
# SEED
# ------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ------------------------------------------
# TRANSFORMS (MUST MATCH APP)
# ------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

data_path = "Auburn_1"
dataset = MarsDataset(data_path, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ------------------------------------------
# MODEL (RESNET)
# ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights="DEFAULT")
model.fc = torch.nn.Linear(model.fc.in_features, 8)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# ------------------------------------------
# TRAIN LOOP
# ------------------------------------------
epochs = 15
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # VALIDATION
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, labels)
            val_loss += loss.item()

            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.2f}")
    print(f"Val Loss: {val_loss:.2f}")
    print(f"Accuracy: {acc:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "model.pth")
        print("✅ Saved best model")

print("🎉 Training Done")