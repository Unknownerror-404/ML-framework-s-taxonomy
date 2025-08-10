import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import time
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import os
import random
import argparse

# CutMix functions
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size(0), device=x.device)
    y_a = y
    y_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

# Deterministic seeding
def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

# ---------------------------
# Args
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--data-dir", type=str, default="data")
args = parser.parse_args()

set_seed(args.seed)

# DataLoader RNG
generator = torch.Generator()
generator.manual_seed(args.seed)

# Transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Loaders
data_dir = args.data_dir
train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
val_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=val_test_transform)
test_dataset = datasets.ImageFolder(f"{data_dir}/test", transform=val_test_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, generator=generator
)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training with CutMix
epochs = args.epochs
start_time = time.perf_counter()
for epoch in range(epochs):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, targets_a, targets_b, lam = cutmix_data(imgs, labels)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} done.")

# Training time
train_time_s = time.perf_counter() - start_time
print(f"Training time: {train_time_s:.2f}s ({train_time_s/60:.2f} min)")

# Evaluation with extra metrics
model.eval()
y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        if len(train_dataset.classes) == 2:
            y_prob.extend(probs[:, 1].cpu().numpy())

# Accuracy
acc = accuracy_score(y_true, y_pred)

# AUC (binary only)
if len(train_dataset.classes) == 2:
    auc = roc_auc_score(y_true, y_prob)
else:
    auc = None

# Sensitivity & Specificity (binary only)
if len(train_dataset.classes) == 2:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
else:
    sensitivity = None
    specificity = None

print(f"Accuracy: {acc:.4f}")
if auc is not None:
    print(f"AUC: {auc:.4f}")
if sensitivity is not None:
    print(f"Sensitivity: {sensitivity:.4f}")
if specificity is not None:
    print(f"Specificity: {specificity:.4f}")