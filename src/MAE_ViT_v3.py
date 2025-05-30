import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
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
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
])

# Load dataset and official split
full_dataset = FlowerDataset(DATA_ROOT, transform)
setid = loadmat(os.path.join(DATA_ROOT, 'setid.mat'))
train_ids = setid['trnid'].squeeze() - 1
val_ids = setid['valid'].squeeze() - 1

train_dataset = Subset(full_dataset, train_ids.tolist())
val_dataset = Subset(full_dataset, val_ids.tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ========== 2. Define MAE and Classifier Head ==========
class MAEWithClassifier(nn.Module):
    def __init__(self, encoder, num_classes=102):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        patches = self.encoder.patch_embed(x)
        latent = self.encoder.blocks(patches)
        latent = self.encoder.norm(latent)
        pooled = latent.mean(dim=1)
        return self.classifier(pooled)

# ========== 3. Fine-Tuning Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = vit_base_patch16_224(pretrained=False)
model = MAEWithClassifier(encoder).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

train_accuracies = []
val_accuracies = []
EPOCHS = 30

print("\n🚀 Starting Fine-tuning...")
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")

# Save accuracy plot
plt.figure()
plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Train Acc')
plt.plot(range(1, EPOCHS + 1), val_accuracies, label='Val Acc')
plt.title("Fine-Tuning Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.savefig("finetune_accuracy_curve.png")

# Save final results
df = pd.DataFrame({
    "Epoch": list(range(1, EPOCHS + 1)),
    "Train_Accuracy": train_accuracies,
    "Val_Accuracy": val_accuracies
})
df.to_csv("finetune_results.csv", index=False)

print("\n✅ Fine-tuning complete.")
print("📈 Accuracy curve saved to 'finetune_accuracy_curve.png'")
print("📄 Detailed log saved to 'finetune_results.csv'")
