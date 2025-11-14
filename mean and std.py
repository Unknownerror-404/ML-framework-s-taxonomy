import numpy as np

"""
metrics_summary.py
------------------

This script computes the mean and standard deviation across multiple seeds 
for a set of classification metrics. It is designed to summarize results 
from repeated experiments and report them in the "mean ± std" format.

Metrics supported:
    - Accuracy (Acc)
    - Area Under the ROC Curve (AUC)
    - Precision-Recall AUC (PR-AUC)
    - Precision
    - Recall
    - F1 Score
    - Specificity
    - Positive Predictive Value (PPV)
    - Negative Predictive Value (NPV)

Usage:
    Fill the lists with metric values collected across seeds (e.g., from saved 
    numpy arrays or experiment outputs). The script will compute mean ± standard 
    deviation for each metric.

Outputs:
    Prints each metric in the format: mean ± standard deviation
    Example:
        Acc:
        Reported as: 0.0602 ± 0.0224
"""


# Fill with your metric values, e.g., AUCs across seeds
acc_scores = [0.0619, 0.0218, 0.0599, 0.0846, 0.0729] #base
auc_scores = [0.0553, 0.0702, 0.0593, 0.0534, 0.0465] #AA
prauc_scores=[0.0522, 0.0845, 0.0688, 0.0422, 0.0513] #RA
precision =  [0.0843, 0.0524, 0.1496, 0.0880, 0.0330] #TA
recall =     [0.0246, 0.0535, 0.0459, 0.0665, 0.0999] #CutM
f1 =         [0.0496, 0.0666, 0.0630, 0.0226, 0.0520] #CutO
specificity= [0.0851, 0.0556, 0.0272, 0.0831, 0.0813] #Mixup
ppv =        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
npv =        [0.8182, 0.8333, 0.6667, 0.9091, 0.8333] 

if len(acc_scores) == 0:
    print("No scores provided.")
else:
    ddof = 1 if len(acc_scores) > 1 else 0
    meanacc = np.mean(acc_scores)
    stdacc = np.std(acc_scores, ddof=ddof)
    print("Acc:")
    print(f"Reported as: {meanacc:.4f} ± {stdacc:.4f}")

if len(auc_scores) == 0:
    print("No scores provided.")
else:
    ddof = 1 if len(auc_scores) > 1 else 0
    meanacc = np.mean(auc_scores)
    stdacc = np.std(auc_scores, ddof=ddof)
    print("Auc:")
    print(f"Reported as: {meanacc:.4f} ± {stdacc:.4f}")

if len(prauc_scores) == 0:
    print("No scores provided.")
else:
    ddof = 1 if len(prauc_scores) > 1 else 0
    meanacc = np.mean(prauc_scores)
    stdacc = np.std(prauc_scores, ddof=ddof)
    print("prauc:")
    print(f"Reported as: {meanacc:.4f} ± {stdacc:.4f}")

if len(precision) == 0:
    print("No scores provided.")
else:
    ddof = 1 if len(precision) > 1 else 0
    meanacc = np.mean(precision)
    stdacc = np.std(precision, ddof=ddof)
    print("Precision:")
    print(f"Reported as: {meanacc:.4f} ± {stdacc:.4f}")

if len(recall) == 0:
    print("No scores provided.")
else:
    ddof = 1 if len(recall) > 1 else 0
    meanacc = np.mean(recall)
    stdacc = np.std(recall, ddof=ddof)
    print("recall:")
    print(f"Reported as: {meanacc:.4f} ± {stdacc:.4f}")

if len(f1) == 0:
    print("No scores provided.")
else:
    ddof = 1 if len(f1) > 1 else 0
    meanacc = np.mean(f1)
    stdacc = np.std(f1, ddof=ddof)
    print("F1:")
    print(f"Reported as: {meanacc:.4f} ± {stdacc:.4f}")

if len(specificity) == 0:
    print("No scores provided.")
else:
    ddof = 1 if len(specificity) > 1 else 0
    meanacc = np.mean(specificity)
    stdacc = np.std(specificity, ddof=ddof)
    print("specificity:")
    print(f"Reported as: {meanacc:.4f} ± {stdacc:.4f}")

if len(ppv) == 0:
    print("No scores provided.")
else:
    ddof = 1 if len(ppv) > 1 else 0
    meanacc = np.mean(ppv)
    stdacc = np.std(ppv, ddof=ddof)
    print("PPV:")
    print(f"Reported as: {meanacc:.4f} ± {stdacc:.4f}")

if len(npv) == 0:
    print("No scores provided.")
else:
    ddof = 1 if len(npv) > 1 else 0
    meanacc = np.mean(npv)
    stdacc = np.std(npv, ddof=ddof)
    print("NPV:")
    print(f"Reported as: {meanacc:.4f} ± {stdacc:.4f}")