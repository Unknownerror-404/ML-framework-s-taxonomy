import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import os
import random
import time
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import argparse
from PIL import Image

# ---------------------------
# Deterministic seeding
# ---------------------------
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
# Corruption functions
# ---------------------------
def add_gaussian_noise_pil(img, std=0.1):
    arr = np.array(img).astype('float32') / 255.0
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    arr = (arr * 255).astype('uint8')
    return Image.fromarray(arr)

def save_jpeg_quality(img, path, quality=40):
    img.save(path, 'JPEG', quality=quality)

def create_corrupted_dataset(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for cls in os.listdir(src_dir):
        cls_src = os.path.join(src_dir, cls)
        cls_dst = os.path.join(dst_dir, cls)
        os.makedirs(cls_dst, exist_ok=True)
        for fname in os.listdir(cls_src):
            img = Image.open(os.path.join(cls_src, fname)).convert('RGB')
            # Gaussian noise
            noisy_img = add_gaussian_noise_pil(img, std=0.3)
            # JPEG compression after noise
            temp_path = os.path.join(cls_dst, fname)
            save_jpeg_quality(noisy_img, temp_path, quality=40)

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

# ---------------------------
# Data transforms
# ---------------------------
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.TrivialAugmentWide(),
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

# ---------------------------
# Dataset & loaders
# ---------------------------
data_dir = args.data_dir
train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
val_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=val_test_transform)
test_dataset = datasets.ImageFolder(f"{data_dir}/test", transform=val_test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                           shuffle=True, num_workers=0, generator=generator)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0, generator=generator)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=0, generator=generator)

# ---------------------------
# Model setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------------------------
# Training loop
# ---------------------------
epochs = args.epochs
start_time = time.perf_counter()
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Training time
train_time_s = time.perf_counter() - start_time
print(f"Training time: {train_time_s:.2f}s ({train_time_s/60:.2f} min)")

# ---------------------------
# Evaluation function
# ---------------------------
def evaluate_model(data_path):
    ds = datasets.ImageFolder(data_path, transform=val_test_transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0, generator=generator)
    y_true, y_pred, y_prob = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            if len(train_dataset.classes) == 2:
                y_prob.extend(probs[:, 1].cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(train_dataset.classes) == 2 else None
    if len(train_dataset.classes) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity = specificity = None
    return acc, auc, sensitivity, specificity

# ---------------------------
# Evaluate on clean & corrupted sets
# ---------------------------
corrupt_test_dir = f"{data_dir}/test_corrupt"
create_corrupted_dataset(f"{data_dir}/test", corrupt_test_dir)

# Evaluate
metrics_clean = evaluate_model(f"{data_dir}/test")
metrics_corrupt = evaluate_model(corrupt_test_dir)

print("\n=== Performance on Clean Test Set ===")
print(f"Accuracy: {metrics_clean[0]:.4f}")
if metrics_clean[1] is not None:
    print(f"AUC: {metrics_clean[1]:.4f}")
if metrics_clean[2] is not None:
    print(f"Sensitivity: {metrics_clean[2]:.4f}")
if metrics_clean[3] is not None:
    print(f"Specificity: {metrics_clean[3]:.4f}")

print("\n=== Performance on Corrupted Test Set ===")
print(f"Accuracy: {metrics_corrupt[0]:.4f}")
if metrics_corrupt[1] is not None:
    print(f"AUC: {metrics_corrupt[1]:.4f}")
if metrics_corrupt[2] is not None:
    print(f"Sensitivity: {metrics_corrupt[2]:.4f}")
if metrics_corrupt[3] is not None:
    print(f"Specificity: {metrics_corrupt[3]:.4f}")

print("\n=== Performance Drop (Clean - Corrupted) ===")
drops = [c - r if c is not None and r is not None else None
         for c, r in zip(metrics_clean, metrics_corrupt)]
labels = ["Accuracy", "AUC", "Sensitivity", "Specificity"]
for lbl, drop in zip(labels, drops):
    if drop is not None:
        print(f"{lbl} Drop: {drop:.4f}")