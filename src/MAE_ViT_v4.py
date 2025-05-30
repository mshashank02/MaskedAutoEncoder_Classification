import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from timm.models.vision_transformer import vit_base_patch16_224
from torchvision.datasets.utils import download_and_extract_archive
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from PIL import Image
import glob
import urllib.request

# ========== 1. Download and Load Dataset ==========
DATA_ROOT = './flower_data'
DATA_URL = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
LABEL_URL = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
SETID_URL = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat'

if not os.path.exists(DATA_ROOT):
    os.makedirs(DATA_ROOT)
    download_and_extract_archive(DATA_URL, download_root=DATA_ROOT)
    os.system(f"wget -P {DATA_ROOT} {LABEL_URL}")
    os.system(f"wget -P {DATA_ROOT} {SETID_URL}")

class FlowerDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.images = sorted(glob.glob(os.path.join(root, 'jpg', '*.jpg')))
        labels = loadmat(os.path.join(root, 'imagelabels.mat'))['labels'][0]
        self.labels = [int(l) - 1 for l in labels]

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def __len__(self):
        return len(self.images)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset and official split
full_dataset = FlowerDataset(DATA_ROOT, transform)
setid = loadmat(os.path.join(DATA_ROOT, 'setid.mat'))
train_ids = setid['trnid'].squeeze() - 1
val_ids = setid['valid'].squeeze() - 1

train_dataset = Subset(full_dataset, train_ids.tolist())
val_dataset = Subset(full_dataset, val_ids.tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ========== 2. Load Pretrained Encoder and Freeze ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = vit_base_patch16_224(pretrained=False)
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False
encoder.to(device)

# ========== 3. Extract Features ==========
def extract_features(model, dataloader):
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            patches = model.patch_embed(imgs)
            latent = model.blocks(patches)
            pooled = model.norm(latent).mean(dim=1)
            features.append(pooled.cpu().numpy())
            labels.append(lbls.numpy())
    return np.vstack(features), np.concatenate(labels)

print("\n🔍 Extracting features with frozen encoder...")
X_train, y_train = extract_features(encoder, train_loader)
X_val, y_val = extract_features(encoder, val_loader)

# ========== 4. Train Linear and MLP Probes ==========
print("\n🎯 Training linear probe...")
clf_linear = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=42)
)
clf_linear.fit(X_train, y_train)
y_pred_linear = clf_linear.predict(X_val)
acc_linear = accuracy_score(y_val, y_pred_linear)

print("🎯 Training MLP probe...")
clf_mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(512, 256),
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    )
)
clf_mlp.fit(X_train, y_train)
y_pred_mlp = clf_mlp.predict(X_val)
acc_mlp = accuracy_score(y_val, y_pred_mlp)

# ========== 5. Save Results ==========
df = pd.DataFrame({
    "Probe": ["Linear", "MLP"],
    "Accuracy": [acc_linear, acc_mlp]
})
df.to_csv("frozen_encoder_probes.csv", index=False)

print(f"\n✅ Linear Probe Accuracy: {acc_linear * 100:.2f}%")
print(f"✅ MLP Probe Accuracy: {acc_mlp * 100:.2f}%")
print("📄 Results saved to 'frozen_encoder_probes.csv'")
