import numpy as np
import argparse
import itertools
from compare_models import delong_roc_test   # reuse from your old script
from sklearn.metrics import roc_auc_score

def main(args):
    # Load y_true once
    y_true = np.load(args.ytrue)

    # Load all model probabilities
    models = {}
    for name, path in zip(args.names, args.files):
        models[name] = np.load(path)

    # Compute AUCs
    aucs = {name: roc_auc_score(y_true, y_prob) for name, y_prob in models.items()}

    print("\n===== AUCs for All Models =====")
    for name, auc in aucs.items():
        print(f"{name}: {auc:.4f}")

    # Run pairwise DeLong tests
    names = list(models.keys())
    print("\n===== Pairwise DeLong Test p-values =====")
    for (name1, name2) in itertools.combinations(names, 2):
        pvalue, auc_pair = delong_roc_test(y_true, models[name1], models[name2])
        print(f"{name1} vs {name2}: p = {pvalue:.4e} "
              f"(AUCs: {auc_pair[0]:.4f} vs {auc_pair[1]:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ytrue", type=str, required=True, help="Path to y_true.npy")
    parser.add_argument("--files", nargs="+", required=True, help="List of y_prob .npy files")
    parser.add_argument("--names", nargs="+", required=True, help="List of model names in same order as files")
    args = parser.parse_args()

    assert len(args.files) == len(args.names), "files and names must match in length"
    main(args)