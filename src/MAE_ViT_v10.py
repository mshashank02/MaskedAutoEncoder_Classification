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

# Configuration
DATA_ROOT = './flower_data'
BATCH_SIZE = 8
NUM_EPOCHS = 200
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

# Split using setid.mat
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

# Step 1: Pretrain MAE model
mae_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
mae_model.to(DEVICE)
mae_model.train()

optimizer = optim.AdamW(mae_model.parameters(), lr=LR)
losses = []

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for batch, _ in train_loader:
        batch = batch.to(DEVICE)
        outputs = mae_model(pixel_values=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")

# Save training curve
plt.plot(range(1, NUM_EPOCHS + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Pretraining Loss")
plt.title("MAE Pretraining Loss on Flower102")
plt.grid(True)
plt.savefig("vitmae_flower_pretraining_loss.png")

# Save pretrained MAE model
mae_model.save_pretrained("./vitmae_flower_pretrained")
processor.save_pretrained("./vitmae_flower_pretrained")

# Step 2: Load ViT encoder and transfer weights
vit_encoder = ViTModel.from_pretrained("facebook/vit-mae-base")
vit_encoder.load_state_dict(mae_model.vit.state_dict(), strict=False)
vit_encoder.to(DEVICE)
vit_encoder.eval()

# Step 3: Feature Extraction
def extract_features(loader):
    features, labels = [], []
    with torch.no_grad():
        for batch, label in loader:
            batch = batch.to(DEVICE)
            outputs = vit_encoder(pixel_values=batch)
            cls_token = outputs.last_hidden_state[:, 0].cpu().numpy()
            features.append(cls_token)
            labels.append(label.numpy())
    return np.vstack(features), np.concatenate(labels)

X_train, y_train = extract_features(train_loader)
X_val, y_val = extract_features(val_loader)
X_test, y_test = extract_features(test_loader)

# Step 4: Probing
# Linear probe
clf_linear = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)
clf_linear.fit(X_train, y_train)
acc_val_linear = accuracy_score(y_val, clf_linear.predict(X_val))
acc_test_linear = accuracy_score(y_test, clf_linear.predict(X_test))

# MLP probe
clf_mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(512, 256), activation='relu', solver='adam',
                  max_iter=1000, early_stopping=True, validation_fraction=0.2)
)
clf_mlp.fit(X_train, y_train)
acc_val_mlp = accuracy_score(y_val, clf_mlp.predict(X_val))
acc_test_mlp = accuracy_score(y_test, clf_mlp.predict(X_test))

# Step 5: Save results
results = pd.DataFrame({
    "Probe": ["Linear", "MLP"],
    "Validation Accuracy": [acc_val_linear, acc_val_mlp],
    "Test Accuracy": [acc_test_linear, acc_test_mlp]
})
results.to_csv("vitmae_flower_probe_results_10.csv", index=False)
print(results)
