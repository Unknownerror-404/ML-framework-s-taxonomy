import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from PIL import Image
import numpy as np
import time
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import os, random, argparse
import matplotlib.pyplot as plt
import json

"""
autoaugment_training.py
-----------------------

Trains EfficientNet-B0 using AutoAugment. Supports multiple seeds, early stopping,
and full evaluation metrics with confidence intervals.
"""

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
    return np.percentile(stats, 100 * alpha / 2), np.percentile(stats, 100 * (1 - alpha / 2))

# ---------------------------
# Args
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=[7, 14, 28, 35, 42, 53, 65, 78, 86, 98])
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--data-dir", type=str, default="./padded")
parser.add_argument("--save-prefix", type=str, default="Exotropia_Efficient-B0_AA")
parser.add_argument("--patience", type=int, default=10)
args = parser.parse_args()

# ---------------------------
# Loop over seeds
# ---------------------------
for seed in args.seeds:
    print(f"\n===== Running seed {seed} (AutoAugment) =====")
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # ---------------------------
    # Transforms
    # ---------------------------
    train_transform = transforms.Compose([
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
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
    # Datasets & loaders
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

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ---------------------------
    # Training loop
    # ---------------------------
    epochs = args.epochs
    history = {"loss": [], "val_acc": [], "val_auc": []}
    best_val_metric = -float("inf")
    patience_counter = 0
    best_model_wts = None

    start_time = time.perf_counter()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # ---- Train ----
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
        history["loss"].append(avg_loss)

        # ---- Validate ---- #
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

        # Metrics
        val_acc = accuracy_score(y_true_val, y_pred_val)
        history["val_acc"].append(val_acc)
        val_auc = roc_auc_score(y_true_val, y_prob_val) if len(train_dataset.classes) == 2 else None
        if val_auc is not None:
            history["val_auc"].append(val_auc)

        # Early stopping
        metric_to_monitor = val_auc if val_auc is not None else val_acc
        if metric_to_monitor > best_val_metric:
            best_val_metric = metric_to_monitor
            best_model_wts = model.state_dict()
            patience_counter = 0
            print(f"  Improvement detected: {metric_to_monitor:.4f}, saving model weights...")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("  Early stopping triggered.")
                break

        print(f"  Training Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}" + (f", Val AUC: {val_auc:.4f}" if val_auc else ""))

    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    train_time_s = time.perf_counter() - start_time
    print(f"\nTraining completed in {train_time_s:.2f}s ({train_time_s/60:.2f} min)")

    # ---------------------------
    # Plot training curves
    # ---------------------------
    plt.figure(figsize=(10, 4))
    if len(train_dataset.classes) == 2 and len(history["val_auc"]) > 0:
        plt.plot(range(1, len(history["val_auc"]) + 1), history["val_auc"], marker="o", label="Val AUC")
    else:
        plt.plot(range(1, len(history["val_acc"]) + 1), history["val_acc"], marker="o", label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"Training Progress (AutoAugment, seed {seed})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"learning_curve_{args.save_prefix}_seed{seed}.png")
    plt.close()

    # ---------------------------
    # Final Evaluation
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
    print("\nConfusion Matrix (Normalized):")
    print(np.round(cm_norm, 3))
    print("\n===== Final Test Results =====")
    print(f"Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"AUC: {auc:.4f} (95% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f})")
        print(f"PR-AUC: {pr_auc:.4f} (95% CI: {pr_auc_ci[0]:.4f} - {pr_auc_ci[1]:.4f})")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Specificity: {specificity:.4f}, NPV: {npv:.4f}")

    cls_report = classification_report(y_true, y_pred, target_names=train_dataset.classes, digits=4, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=train_dataset.classes, digits=4))

    # ---------------------------
    # Save results
    # ---------------------------
    results = {
        "seed": seed,
        "classifier_used": "Exotropia_Efficient-B0",
        "augmentation_performed": "AutoAugment",
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

    json_filename = f"autoaugment_Exotropia_Efficient-B0_{seed}.json"
    with open(json_filename, "w") as jf:
        json.dump(results, jf, indent=4)
    print(f"Results JSON saved at: {json_filename}")

    # Save predictions
    np.save(f"y_true_seed{seed}.npy", np.array(y_true))
    np.save(f"y_pred_{args.save_prefix}_seed{seed}.npy", np.array(y_pred))
    if len(train_dataset.classes) == 2:
        np.save(f"y_prob_{args.save_prefix}_seed{seed}.npy", np.array(y_prob))

    # Save model
    model_path = f"best_model_{args.save_prefix}_seed{seed}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Best model weights saved to {model_path}")