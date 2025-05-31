import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from transformers import ViTImageProcessor, ViTMAEForPreTraining, ViTModel
from scipy.io import loadmat
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
DATA_ROOT = './flower_data'
BATCH_SIZE = 8
NUM_EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Definition
class FlowerDataset(Dataset):
    def __init__(self, root, processor):
        self.images = sorted(glob.glob(os.path.join(root, 'jpg', '*.jpg')))
        labels = loadmat(os.path.join(root, 'imagelabels.mat'))['labels'][0]
        self.labels = [int(l) - 1 for l in labels]
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed['pixel_values'].squeeze(0)
        return pixel_values, self.labels[idx]

# Load processor and dataset
processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")
dataset = FlowerDataset(DATA_ROOT, processor)

# Split indices
setid = loadmat(os.path.join(DATA_ROOT, 'setid.mat'))
train_ids = setid['trnid'].squeeze() - 1
val_ids = setid['valid'].squeeze() - 1
test_ids = setid['tstid'].squeeze() - 1

train_dataset = Subset(dataset, train_ids.tolist())
val_dataset = Subset(dataset, val_ids.tolist())
test_dataset = Subset(dataset, test_ids.tolist())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load full pretrained MAE model
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
model.to(DEVICE)
model.train()

# Optimizer and training loop
optimizer = optim.AdamW(model.parameters(), lr=LR)
losses = []

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for batch, _ in train_loader:
        batch = batch.to(DEVICE)
        outputs = model(pixel_values=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")

# Save model
save_path = "./vitmae_flower_pretrained"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
processor.save_pretrained(save_path)

# Save training loss curve
plt.plot(range(1, NUM_EPOCHS + 1), losses, marker='o')
plt.title("ViTMAE Pretraining Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("vitmae_flower_loss.png")

# Feature extraction using ViTModel (frozen encoder)
print("Extracting features...")
vit_encoder = ViTModel.from_pretrained("facebook/vit-mae-base")
vit_encoder.to(DEVICE)
vit_encoder.eval()

def extract_features(loader):
    all_feats, all_labels = [], []
    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(DEVICE)
            outputs = vit_encoder(pixel_values=batch)
            cls_feats = outputs.last_hidden_state[:, 0].cpu().numpy()
            all_feats.append(cls_feats)
            all_labels.append(labels.numpy())
    return np.vstack(all_feats), np.concatenate(all_labels)

X_train, y_train = extract_features(train_loader)
X_val, y_val = extract_features(val_loader)
X_test, y_test = extract_features(test_loader)

# Train Probes
print("Training probes...")

# Linear Probe
clf_linear = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)
clf_linear.fit(X_train, y_train)
acc_linear_val = accuracy_score(y_val, clf_linear.predict(X_val))
acc_linear_test = accuracy_score(y_test, clf_linear.predict(X_test))

# MLP Probe
clf_mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2
    )
)
clf_mlp.fit(X_train, y_train)
acc_mlp_val = accuracy_score(y_val, clf_mlp.predict(X_val))
acc_mlp_test = accuracy_score(y_test, clf_mlp.predict(X_test))

# Save results
results = pd.DataFrame({
    "Probe": ["Linear", "MLP"],
    "Validation Accuracy": [acc_linear_val, acc_mlp_val],
    "Test Accuracy": [acc_linear_test, acc_mlp_test]
})
results.to_csv("vitmae_probe_results.csv", index=False)
print("Results saved to vitmae_probe_results.csv")
