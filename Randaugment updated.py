import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
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
    stats = np.array(stats)
    return np.percentile(stats, 100*alpha/2), np.percentile(stats, 100*(1-alpha/2))

# ---------------------------
# Args
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=[98,86,78,65,53,42,35,28,14,7])
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--data-dir", type=str, default="./padded")
parser.add_argument("--save-prefix", type=str, default="Exotropia_MobileNetV2_RA")
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()

# ---------------------------
# Multi-seed loop
# ---------------------------
for seed in args.seeds:
    print(f"\n===== Running seed {seed} =====")
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # ---------------------------
    # Transforms (RandAugment)
    # ---------------------------
    train_transform = transforms.Compose([
        transforms.RandAugment(),
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
    train_dataset = datasets.ImageFolder(f"{args.data_dir}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{args.data_dir}/validation", transform=val_test_transform)
    test_dataset = datasets.ImageFolder(f"{args.data_dir}/test", transform=val_test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, generator=generator
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, generator=generator
    )

    # ---------------------------
    # Model
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---------------------------
    # Training loop with early stopping
    # ---------------------------
    best_metric = -float("inf")
    patience_counter = 0
    best_model_wts = deepcopy(model.state_dict())
    history = {"train_loss": [], "val_acc": [], "val_auc": []}

    start_time = time.perf_counter()

    for epoch in range(args.epochs):
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
        avg_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # ---- Validation ----
        model.eval()
        y_true_val, y_pred_val, y_prob_val = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())
                if len(train_dataset.classes) == 2:
                    y_prob_val.extend(probs[:, 1].cpu().numpy())

        val_acc = accuracy_score(y_true_val, y_pred_val)
        history["val_acc"].append(val_acc)

        # Use AUC for binary, else accuracy
        if len(train_dataset.classes) == 2:
            val_auc = roc_auc_score(y_true_val, y_prob_val)
            history["val_auc"].append(val_auc)
            metric_to_monitor = val_auc
        else:
            metric_to_monitor = val_acc
            val_auc = None

        # Early stopping
        if metric_to_monitor > best_metric:
            best_metric = metric_to_monitor
            patience_counter = 0
            best_model_wts = deepcopy(model.state_dict())
            torch.save(best_model_wts, f"model_RA_Exotropia_MobileNetV2{seed}.pth")
            print(f"Epoch {epoch+1}: Improvement! Metric={metric_to_monitor:.4f}, saving model.")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: No improvement. Patience {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}" +
              (f", Val AUC={val_auc:.4f}" if val_auc else ""))

    # Load best weights
    model.load_state_dict(best_model_wts)
    train_time_s = time.perf_counter() - start_time
    print(f"\nTraining completed in {train_time_s:.2f}s ({train_time_s/60:.2f} min)")

    # ---------------------------
    # Plot learning curves
    # ---------------------------
    plt.figure(figsize=(8,5))
    plt.plot(history["train_loss"], label="Train Loss")
    if len(train_dataset.classes) == 2 and history["val_auc"]:
        plt.plot(range(len(history["val_auc"])), history["val_auc"], label="Val AUC")
    else:
        plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Training Progress (Seed {seed})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"learning_curve_{args.save_prefix}_seed{seed}.png")
    plt.close()

    # ---------------------------
    # Test evaluation
    # ---------------------------
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

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    if len(train_dataset.classes) == 2:
        auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        auc_ci = bootstrap_ci(roc_auc_score, np.array(y_true), np.array(y_prob))
        pr_auc_ci = bootstrap_ci(average_precision_score, np.array(y_true), np.array(y_prob))
    else:
        auc = pr_auc = precision = recall = f1 = None
        sensitivity = specificity = ppv = npv = None
        auc_ci = pr_auc_ci = None

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # ---------------------------
    # Save results and confusion matrix
    # ---------------------------
    results = {
        "seed": seed,
        "classifier_used": "Exotropia_MobileNetV2",
        "augmentation_performed": "RandAugment",
        "save_prefix": args.save_prefix,
        "train_time_seconds": train_time_s,
        "metrics": {
            "seed" : seed,
            "accuracy": float(acc),
            "auc": float(auc) if auc else None,
            "auc_ci": [float(x) for x in auc_ci] if auc_ci else None,
            "pr_auc": float(pr_auc) if pr_auc else None,
            "pr_auc_ci": [float(x) for x in pr_auc_ci] if pr_auc_ci else None,
            "precision": float(precision) if auc else None,
            "recall": float(recall) if auc else None,
            "f1": float(f1) if auc else None,
            "specificity": float(specificity) if auc else None,
            "npv": float(npv) if auc else None,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": cm_norm.round(3).tolist(),
            "per_class_report": classification_report(y_true, y_pred, target_names=train_dataset.classes, digits=4, output_dict=True)
        },
        "history": history
    }

    json_filename = f"Randaugment_Exotropia_MobileNetV2_seed{seed}.json"
    with open(json_filename, "w") as jf:
        json.dump(results, jf, indent=4)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Seed {seed})")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{args.save_prefix}_seed{seed}.png")
    plt.close()

    np.save(f"y_true_seed{seed}.npy", np.array(y_true))
    np.save(f"y_pred_{args.save_prefix}_seed{seed}.npy", np.array(y_pred))
    if len(train_dataset.classes) == 2:
        np.save(f"y_prob_{args.save_prefix}_seed{seed}.npy", np.array(y_prob))
    torch.save(model.state_dict(), f"best_model_{args.save_prefix}_seed{seed}.pth")

    print(f"Seed {seed} done. Model and results saved.\n")
