import numpy as np
from sklearn.metrics import roc_auc_score
from delong import delong_roc_test
import argparse

# ---------------------------
# Args
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model1", type=str, required=True, help="File for model1 probabilities (e.g., y_prob_autoaugment.npy)")
parser.add_argument("--model2", type=str, required=True, help="File for model2 probabilities (e.g., y_prob_randaugment.npy)")
parser.add_argument("--ytrue", type=str, default="y_true.npy", help="File for ground truth labels")
args = parser.parse_args()

# ---------------------------
# Load saved predictions
# ---------------------------
y_true = np.load(args.ytrue)
y_prob_model1 = np.load(args.model1)
y_prob_model2 = np.load(args.model2)

# ---------------------------
# Compute AUCs
# ---------------------------
auc1 = roc_auc_score(y_true, y_prob_model1)
auc2 = roc_auc_score(y_true, y_prob_model2)

# ---------------------------
# Run DeLong test
# ---------------------------
pvalue, auc1_delong, auc2_delong = delong_roc_test(
    y_true, y_prob_model1, y_prob_model2
)

# ---------------------------
# Print results
# ---------------------------
print("\n===== DeLong Test Results =====")
print(f"Model 1 AUC: {auc1:.4f}")
print(f"Model 2 AUC: {auc2:.4f}")
print(f"DeLong test p-value: {pvalue:.4e}")