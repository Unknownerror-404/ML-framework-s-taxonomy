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
cutmix_training.py
------------------
This script trains a model on an image dataset using the CutMix
augmentation strategy. It supports multi-seed experiments, early stopping, and
outputs full evaluation metrics with confidence intervals.

Main features:
    - Applies CutMix augmentation during training
    - Deterministic seeding for reproducibility
    - Computes Accuracy, ROC-AUC, PR-AUC, Precision, Recall, F1, Sensitivity,
      Specificity, PPV, NPV
    - Produces bootstrap confidence intervals for AUC and PR-AUC
    - Saves best model weights, predictions, probabilities, and confusion matrices

Usage:
    python cutmix_training.py --data-dir ./data --epochs 50 --batch-size 16

Arguments:
    --data-dir     Path to dataset directory with 'train', 'val', and 'test' folders
    --epochs       Number of training epochs (default: 100)
    --batch-size   Batch size (default: 16)
    --seeds        List of random seeds (default: [42, 35, 28, 14, 53, 78, 86, 98])
    --patience     Early stopping patience (default: 10)
    --save_prefix  Prefix for saved outputs (default: "CutMix")

Outputs:
    - best_model_seed{seed}.pth
    - y_true_seed{seed}_CutMix.npy
    - y_pred_seed{seed}_CutMix.npy
    - y_prob_seed{seed}_CutMix.npy (binary classification only)
    - confusion_matrix_CutMix_seed{seed}.png
    - Printed metrics and runtime per seed
"""

# ---------------------------
# CutMix function
# ---------------------------
def rand_bbox(size, lam):
    """Generate random bounding box"""
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
    """Apply CutMix to a batch of images and return new data + mixed targets"""
    indices = torch.randperm(x.size(0))
    shuffled_x = x[indices]
    shuffled_y = y[indices]

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    x = x.clone()
    x[:, :, bby1:bby2, bbx1:bbx2] = shuffled_x[:, :, bby1:bby2, bbx1:bbx2]

    # adjust lambda based on the patch area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

    return x, y, shuffled_y, lam

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
parser.add_argument("--seeds", type=int, nargs="+", default=[98, 86, 78, 65, 53, 42, 35, 28, 14, 7])
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--data-dir", type=str, default="./padded")
parser.add_argument("--save_prefix", type=str, default="Exotropia_Efficient-B0_CutMix")
parser.add_argument("--patience", type=int, default=10)
args = parser.parse_args([]) 

# ---------------------------
# Multi-seed loop
# ---------------------------
seeds = args.seeds
for seed in seeds:
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
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ---------------------------
    # Training with CutMix
    # ---------------------------
    best_val_acc = 0.0
    patience_counter = 0
    best_model_wts = deepcopy(model.state_dict())
    history = {"train_loss": [], "val_auc": []}

    start_time = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # Apply CutMix
            imgs, y1, y2, lam = cutmix_data(imgs, labels, alpha=1.0)
            outputs = model(imgs)
            loss = lam * criterion(outputs, y1) + (1 - lam) * criterion(outputs, y2)
            loss = loss.mean()

            optimizer.zero_grad()
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
                if len(train_dataset.classes) == 2:
                    y_prob_val.extend(probs[:, 1].cpu().numpy())
                y_true_val.extend(labels.cpu().numpy())

        if len(train_dataset.classes) == 2:
            val_auc = roc_auc_score(y_true_val, y_prob_val)
        else:
            val_auc = accuracy_score(y_true_val, np.argmax(y_prob_val, axis=1))  

        history["train_loss"].append(avg_loss)
        history["val_auc"].append(val_auc)  

        if val_auc > best_val_acc:
            print(f"Epoch {epoch+1}/{args.epochs} "
                  f"Loss: {avg_loss:.4f} | "
                  f"Val AUC improved {best_val_acc:.4f} → {val_auc:.4f} | Saving model")
            best_val_acc = val_auc
            patience_counter = 0
            best_model_wts = deepcopy(model.state_dict())
            torch.save(best_model_wts, f"best_Exotropia_Efficient-B0_seedCutmix{seed}.pth")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{args.epochs} "
                  f"Loss: {avg_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} (no improvement, patience {patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break


    # ---------------------------
    # Test Evaluation
    # ---------------------------
    model.load_state_dict(torch.load(f"best_Exotropia_Efficient-B0_seedCutmix{seed}.pth"))
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
    # SAVE predictions & probs
    # ---------------------------
    print("\n===== Per-Class Results =====")
    cls_report = classification_report(
            y_true,
            y_pred,
            target_names=train_dataset.classes,
            digits=4,
            output_dict=True
        )
    print(classification_report(
            y_true,
            y_pred,
            target_names=train_dataset.classes,
            digits=4
        ))
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    np.save(f"y_true_seed{seed}_{args.save_prefix}.npy", np.array(y_true))
    np.save(f"y_pred_seed{seed}_{args.save_prefix}.npy", np.array(y_pred))
    if len(train_dataset.classes) == 2:
        np.save(f"y_prob_seed{seed}_{args.save_prefix}.npy", np.array(y_prob))
    
    train_time_s = time.perf_counter() - start_time
    print(f"\nTraining time: {train_time_s:.2f} seconds "
          f"({train_time_s/60:.2f} minutes)")
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
    results = {
        "seed": seed,
        "classifier_used": "Exotropia_Efficient-B0",
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
    #plt.savefig(f"confusion_matrix_{args.save_prefix}_seed{seed}.png")
    #plt.show()

    # ---------------------------
    # Training done: print time
    # ---------------------------

    json_filename = f"Cutmix_Exotropia_Efficient-B0_{seed}.json"
    with open(json_filename, "w") as jf:
        json.dump(results, jf, indent=4)