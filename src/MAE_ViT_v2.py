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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from timm.models.vision_transformer import vit_base_patch16_224
from torchvision.datasets.utils import download_and_extract_archive
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from PIL import Image
import glob

# ========== 1. Download and Load Dataset ==========
DATA_ROOT = './flower_data'
DATA_URL = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
LABEL_URL = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'

if not os.path.exists(DATA_ROOT):
    os.makedirs(DATA_ROOT)
    download_and_extract_archive(DATA_URL, download_root=DATA_ROOT)
    os.system(f"wget -P {DATA_ROOT} {LABEL_URL}")

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

# Use first 1020 images (10 per class × 102 classes)
full_dataset = FlowerDataset(DATA_ROOT, transform)
train_dataset = Subset(full_dataset, list(range(1020)))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ========== 2. Define MAE Model ==========
class MAE(nn.Module):
    def __init__(self, encoder, decoder_dim=512, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Linear(encoder.embed_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, 3 * 224 * 224)  # reconstruct full image
        )

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.encoder.patch_embed(x)
        num_patches = patches.shape[1]
        num_mask = int(self.mask_ratio * num_patches)

        noise = torch.rand(B, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :-num_mask]
        patches_keep = torch.gather(patches, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, patches.shape[2]))

        latent = self.encoder.blocks(patches_keep)
        latent = self.encoder.norm(latent)

        pred = self.decoder(latent.mean(dim=1))
        pred = pred.view(B, 3, 224, 224)  # reshape to full image
        return pred

# ========== 3. Pretrain MAE ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = vit_base_patch16_224(pretrained=False)
mae = MAE(encoder).to(device)

optimizer = optim.Adam(mae.parameters(), lr=1e-4)
criterion = nn.MSELoss()
train_losses = []

print("🚀 Starting MAE pretraining...")
for epoch in range(20):
    mae.train()
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        output = mae(imgs)
        loss = criterion(output, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_loss = total_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/20: Loss = {epoch_loss:.4f}")

# Save training loss plot
plt.figure()
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
plt.title("MAE Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("mae_loss_curve.png")
plt.close()

# ========== 4. Feature Extraction ==========
print("🔍 Extracting features...")
mae.eval()
features, labels = [], []
with torch.no_grad():
    for imgs, lbls in train_loader:
        imgs = imgs.to(device)
        patches = mae.encoder.patch_embed(imgs)
        encoded = mae.encoder.blocks(patches).mean(dim=1)
        features.append(encoded.cpu().numpy())
        labels.append(lbls.numpy())

features = np.vstack(features)
labels = np.concatenate(labels)

# ========== 5. Train Linear and MLP Classifiers ==========
print("🎯 Training probes...")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, stratify=labels, random_state=42)

# Linear Probe
clf_linear = LogisticRegression(max_iter=5000)
clf_linear.fit(X_train, y_train)
acc_linear = accuracy_score(y_test, clf_linear.predict(X_test))

# MLP Probe with dropout-like regularization and weight decay


clf_mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(256,),
        activation='relu',
        solver='adam',
        alpha=1e-4,                 # L2 regularization (weight decay)
        learning_rate_init=1e-3,
        max_iter=1000,
        random_state=42,
        early_stopping=True,       # monitor validation loss
        validation_fraction=0.2
    )
)
clf_mlp.fit(X_train, y_train)
acc_mlp = accuracy_score(y_test, clf_mlp.predict(X_test))

# Save probe results
df = pd.DataFrame({
    "Probe": ["Linear", "MLP"],
    "Accuracy": [acc_linear, acc_mlp]
})
df.to_csv("probe_results.csv", index=False)

print("✅ Done. All steps completed.")
print("📈 Loss curve saved to 'mae_loss_curve.png'")
print("📄 Probe accuracies saved to 'probe_results.csv'")
