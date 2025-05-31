import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from transformers import (
    ViTImageProcessor, ViTMAEForPreTraining, ViTForImageClassification, ViTConfig
)
from scipy.io import loadmat
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

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

# Save pretrained MAE model
save_path = "./vitmae_flower_pretrained"
os.makedirs(save_path, exist_ok=True)
mae_model.save_pretrained(save_path)
processor.save_pretrained(save_path)

plt.plot(range(1, NUM_EPOCHS + 1), losses, marker='o')
plt.title("ViTMAE Pretraining Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("vitmae_flower_loss.png")

# Step 2: Transfer encoder to classification model
config = ViTConfig.from_pretrained("facebook/vit-mae-base")
config.num_labels = 102
clf_model = ViTForImageClassification(config)
clf_model.vit.load_state_dict(mae_model.vit.state_dict())

# Step 3: Freeze encoder
for param in clf_model.vit.parameters():
    param.requires_grad = False
clf_model.to(DEVICE)
clf_model.train()

# Step 4: Train classification head
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, clf_model.parameters()), lr=LR)
ce_loss = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    clf_model.train()
    total_loss = 0
    for batch, labels in train_loader:
        batch = batch.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = clf_model(pixel_values=batch)
        loss = ce_loss(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Classifier Epoch {epoch+1}/10 - Loss: {total_loss/len(train_loader):.4f}")

# Step 5: Evaluate
clf_model.eval()
def evaluate(loader):
    preds, truths = [], []
    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(DEVICE)
            outputs = clf_model(pixel_values=batch)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            truths.extend(labels.numpy())
    return accuracy_score(truths, preds)

val_acc = evaluate(val_loader)
test_acc = evaluate(test_loader)

print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Step 6: Save results to CSV
results_df = pd.DataFrame({
    "Metric": ["Validation Accuracy", "Test Accuracy"],
    "Score": [val_acc, test_acc]
})
results_df.to_csv("vitmae_final_results_8.csv", index=False)
print("Results saved to vitmae_final_results.csv")
