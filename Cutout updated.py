import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import time
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
import os
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import json

"""
cutout_training.py
------------------

This script trains a Exotropia_MobileNetV3 model on an image dataset using the Cutout
augmentation strategy. It supports multi-seed experiments, early stopping, and
outputs full evaluation metrics with confidence intervals.

Modified version:
    - Early stopping and checkpointing use validation ROC-AUC instead of accuracy
"""

# ---------------------------
# Cutout function
# ---------------------------
def cutout_data(x, mask_size=50):
    """Apply Cutout to a batch of images"""
    h, w = x.size(2), x.size(3)
    new_x = x.clone()
    for i in range(x.size(0)):
        top = np.random.randint(0, max(1, h - mask_size))
        left = np.random.randint(0, max(1, w - mask_size))
        new_x[i, :, top:top+mask_size, left:left+mask_size] = 0
    return new_x

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
# Bootstrap confidence intervals
# ---------------------------
def bootstrap_ci(metric_fn, y_true, y_score, n_bootstraps=1000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    stats = []
    y_true, y_score = np.array(y_true), np.array(y_score)
    for _ in range(n_bootstraps):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            stats.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    return np.percentile(stats, 100*alpha/2), np.percentile(stats, 100*(1-alpha/2))

# ---------------------------
# Args
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 35, 28, 14, 7, 53, 65, 78, 86, 98])
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--data-dir", type=str, default="./padded")
parser.add_argument("--mask-size", type=int, default=50)
parser.add_argument("--save_prefix", type=str, default="Exotropia_MobileNetV3_Cutout")
parser.add_argument("--patience", type=int, default=10)
args = parser.parse_args([])

# ---------------------------
# Multi-seed loop
# ---------------------------

for seed in args.seeds:
    print(f"\n===== Running with seed {seed} =====")
    set_seed(seed)

    generator = torch.Generator().manual_seed(seed)

    # ---------------------------
    # Transforms
    # ---------------------------
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ---------------------------
    # Datasets & Loaders
    # ---------------------------
    data_dir = args.data_dir
    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{data_dir}/validation", transform=val_test_transform)
    test_dataset = datasets.ImageFolder(f"{data_dir}/test", transform=val_test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=0, generator=generator)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0, generator=generator)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0, generator=generator)

    # ---------------------------
    # Model
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(train_dataset.classes))
    
    
    #model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    #num_features = model.classifier[3].in_features
    #model.classifier[3] = nn.Linear(num_features, len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ---------------------------
    # Training with Cutout + Early Stopping (AUC-based)
    # ---------------------------
    best_val_auc = 0.0
    patience_counter = 0
    best_model_wts = deepcopy(model.state_dict())
    history = {"train_loss": [], "val_auc": []}

    start_time = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = cutout_data(imgs, mask_size=args.mask_size)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        y_true_val, y_prob_val = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                y_true_val.extend(labels.cpu().numpy())
                if len(train_dataset.classes) == 2:
                    y_prob_val.extend(probs[:, 1].cpu().numpy())
                else:
                    y_prob_val.extend(probs.cpu().numpy())

        if len(train_dataset.classes) == 2:
            val_auc = roc_auc_score(y_true_val, y_prob_val)
        else:
            val_auc = roc_auc_score(y_true_val, np.array(y_prob_val),
                                    multi_class='ovr', average='macro')

        history["train_loss"].append(avg_loss)
        history["val_auc"].append(val_auc)

        if val_auc > best_val_auc:
            print(f"Epoch {epoch+1}/{args.epochs} "
                  f"Loss: {avg_loss:.4f} | "
                  f"Val AUC improved {best_val_auc:.4f} → {val_auc:.4f} | Saving model   ")
            best_val_auc = val_auc
            patience_counter = 0
            best_model_wts = deepcopy(model.state_dict())
            torch.save(best_model_wts, f"best_Exotropia_MobileNetV3_seed_CutoutV3{seed}.pth")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{args.epochs} "
                  f"Loss: {avg_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} (no improvement, patience {patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    train_time_s = time.perf_counter() - start_time
    print(f"Training time: {train_time_s:.2f}s ({train_time_s/60:.2f} min)")

    # ---------------------------
    # Plot learning curves
    # ---------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_auc"], label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Training Progress (Seed {seed})")
    plt.legend()
    plt.tight_layout()

    # ---------------------------
    # Test Evaluation
    # ---------------------------
    model.load_state_dict(torch.load(f"best_Exotropia_MobileNetV3_seed_CutoutV3{seed}.pth"))
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

    # ---------------------------
    # Metrics
    # ---------------------------
    acc = accuracy_score(y_true, y_pred)
    if len(train_dataset.classes) == 2:
        auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        auc_ci = bootstrap_ci(roc_auc_score, np.array(y_true), np.array(y_prob))
        pr_auc_ci = bootstrap_ci(average_precision_score, np.array(y_true), np.array(y_prob))
    else:
        auc = pr_auc = precision = recall = f1 = None
        sensitivity = specificity = ppv = npv = None
        auc_ci = pr_auc_ci = None

    print("\n===== Per-Class Results =====")
    cls_report = classification_report(
        y_true, y_pred, target_names=train_dataset.classes,
        digits=4, output_dict=True
    )
    print(classification_report(
        y_true, y_pred, target_names=train_dataset.classes, digits=4
    ))

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    results = {
        "seed": seed,
        "classifier_used": "Exotropia_MobileNetV3",
        "augmentation_performed": "padded224",
        "save_prefix": args.save_prefix,
        "train_time_seconds": train_time_s,
        "metrics": {
            "accuracy": float(acc),
            "auc": float(auc) if auc is not None else None,
            "auc_ci": [float(x) for x in auc_ci] if auc_ci is not None else None,
            "pr_auc": float(pr_auc) if pr_auc is not None else None,
            "pr_auc_ci": [float(x) for x in pr_auc_ci] if pr_auc_ci is not None else None,
            "precision": float(precision) if auc is not None else None,
            "recall": float(recall) if auc is not None else None,
            "f1": float(f1) if auc is not None else None,
            "specificity": float(specificity) if auc is not None else None,
            "npv": float(npv) if auc is not None else None,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": cm_norm.round(3).tolist(),
            "per_class_report": cls_report
        },
        "history": history
    }

    json_filename = f"Cutout_Exotropia_MobileNetV3_{seed}.json"
    with open(json_filename, "w") as jf:
        json.dump(results, jf, indent=4)

    np.save(f"y_true_seed{seed}.npy", np.array(y_true))
    np.save(f"y_pred_seed{seed}_{args.save_prefix}.npy", np.array(y_pred))
    if y_prob:
        np.save(f"y_prob_seed{seed}_{args.save_prefix}.npy", np.array(y_prob))

    # ---------------------------
    # Confusion Matrix
    # ---------------------------
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (raw):")
    print(cm)

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    print("\nConfusion Matrix (normalized per class):")
    print(np.round(cm_normalized, 3))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Seed {seed})")
    plt.tight_layout()