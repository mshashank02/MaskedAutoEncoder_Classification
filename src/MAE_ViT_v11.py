# FULL SCRIPT — FINETUNING + LINEAR/MLP PROBES
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from torchvision import transforms
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

# Config
DATA_ROOT = './flower_data'
BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor()
])
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset
class FlowerDataset(Dataset):
    def __init__(self, root, processor, transform=None):
        self.images = sorted(glob.glob(os.path.join(root, 'jpg', '*.jpg')))
        labels = loadmat(os.path.join(root, 'imagelabels.mat'))['labels'][0]
        self.labels = [int(l) - 1 for l in labels]
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = self.transform(image)
        image = self.processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
        return image, self.labels[idx]

# Load processor
processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")

# Datasets
dataset_train = FlowerDataset(DATA_ROOT, processor, transform=train_transform)
dataset_val = FlowerDataset(DATA_ROOT, processor, transform=val_test_transform)
dataset_test = FlowerDataset(DATA_ROOT, processor, transform=val_test_transform)

setid = loadmat(os.path.join(DATA_ROOT, 'setid.mat'))
train_ids = setid['trnid'].squeeze() - 1
val_ids = setid['valid'].squeeze() - 1
test_ids = setid['tstid'].squeeze() - 1

train_dataset = Subset(dataset_train, train_ids.tolist())
val_dataset = Subset(dataset_val, val_ids.tolist())
test_dataset = Subset(dataset_test, test_ids.tolist())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load ViT encoder
vit = ViTModel.from_pretrained("facebook/vit-mae-base", output_hidden_states=True)
vit.to(DEVICE)

# Freeze all but last block
for name, param in vit.named_parameters():
    param.requires_grad = False
for param in vit.encoder.layer[-1].parameters():
    param.requires_grad = True

# Classifier Head
class ViTClassifier(nn.Module):
    def __init__(self, encoder, num_classes=102):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(768 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        outputs = self.encoder(x)
        hs = outputs.hidden_states
        cls_tokens = [h[:, 0] for h in hs[-4:]]
        concat = torch.cat(cls_tokens, dim=1)
        return self.classifier(concat)

model = ViTClassifier(vit).to(DEVICE)

# Train loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
train_accs, val_accs = [], []

for epoch in range(NUM_EPOCHS):
    model.train()
    total, correct = 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), torch.tensor(labels).to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total
    train_accs.append(train_acc)

    # Validation
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), torch.tensor(labels).to(DEVICE)
            outputs = model(inputs)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# Test eval (finetuned model)
model.eval()
total, correct = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), torch.tensor(labels).to(DEVICE)
        outputs = model(inputs)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
test_acc_finetuned = correct / total
print(f"Final Test Accuracy (Finetuned): {test_acc_finetuned:.4f}")

# Accuracy plot
plt.plot(train_accs, label="Train")
plt.plot(val_accs, label="Validation")
plt.title("ViT Finetuning Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("vit_finetune_acc_plot.png")

# --- Linear & MLP Probe ---
def extract_frozen_features(loader):
    features, labels = [], []
    vit.eval()
    with torch.no_grad():
        for inputs, lbls in loader:
            inputs = inputs.to(DEVICE)
            hs = vit(inputs).hidden_states
            cls_tokens = [h[:, 0] for h in hs[-4:]]
            concat = torch.cat(cls_tokens, dim=1).cpu().numpy()
            features.append(concat)
            labels.append(np.array(lbls))
    return np.vstack(features), np.concatenate(labels)

print("Extracting frozen features...")
X_train, y_train = extract_frozen_features(train_loader)
X_val, y_val = extract_frozen_features(val_loader)
X_test, y_test = extract_frozen_features(test_loader)

# Linear probe
clf_linear = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
clf_linear.fit(X_train, y_train)
acc_val_linear = accuracy_score(y_val, clf_linear.predict(X_val))
acc_test_linear = accuracy_score(y_test, clf_linear.predict(X_test))

# MLP probe
clf_mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(512, 256), activation='relu',
                  max_iter=1000, early_stopping=True)
)
clf_mlp.fit(X_train, y_train)
acc_val_mlp = accuracy_score(y_val, clf_mlp.predict(X_val))
acc_test_mlp = accuracy_score(y_test, clf_mlp.predict(X_test))

# Save all results
results = pd.DataFrame({
    "Probe": ["Finetuned", "Linear", "MLP"],
    "Validation Accuracy": [val_accs[-1], acc_val_linear, acc_val_mlp],
    "Test Accuracy": [test_acc_finetuned, acc_test_linear, acc_test_mlp]
})
results.to_csv("vit_finetune_vs_probe_results.csv", index=False)
print("\n=== Final Results ===")
print(results)
