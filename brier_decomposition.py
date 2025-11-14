#import argparse
import numpy as np
from sklearn.metrics import brier_score_loss
import json
"""
brier_score_eval.py
-------------------

This script computes the Brier score for probabilistic probictions in both
binary and multiclass classification tasks. For binary classification, it also
provides a decomposition of the Brier score into reliability, resolution, and
uncertainty components.

Main features:
    - Computes Brier score for binary and multiclass probictions
    - For binary classification, provides reliability, resolution, and uncertainty
      decomposition
    - Supports probictions stored as .npy files

Usage:
    python brier_score_eval.py --model ./y_prob.npy --y_true ./y_true.npy

Arguments:
    --model     Path to probicted probabilities (.npy file)
    --y_true    Path to true labels (.npy file)

Outputs:
    - Prints Brier score to stdout
    - For binary classification, also prints Reliability, Resolution, and Uncertainty
"""

def brier_score_multiclass(y_true, y_prob):
    n_classes = y_prob.shape[1]
    y_onehot = np.zeros_like(y_prob)
    y_onehot[np.arange(len(y_true)), y_true] = 1
    return np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1))


def brier_decomposition(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Overall climatology
    p_bar = np.mean(y_true)

    # Bin probictions
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.minimum(bin_ids, n_bins - 1)  # clamp to valid range

    reliability = 0.0
    resolution = 0.0
    counts_total = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        y_bin = y_true[mask]
        p_bin = y_prob[mask]
        n_bin = len(y_bin)

        # Average forecast probability in this bin
        p_hat = np.mean(p_bin)
        # Empirical event frequency
        o_hat = np.mean(y_bin)

        reliability += n_bin * (p_hat - o_hat) ** 2
        resolution += n_bin * (o_hat - p_bar) ** 2

    reliability /= counts_total
    resolution /= counts_total
    uncertainty = p_bar * (1 - p_bar)

    brier = brier_score_loss(y_true, y_prob)
    return brier, reliability, resolution, uncertainty


def main():
    models = [

        #"./y_prob_MobileNetV2_baseline_98.npy",
        #"./y_prob_MobileNet-V2_AA_seed98.npy",
        #"./y_prob_MobileNetV2_RA_seed98.npy",
        #"./y_prob_MobileNet_V2_TA_seed98.npy",
        #"./y_prob_seed98_MobileNetV2_Cutout.npy",
        #"./y_prob_seed98_MobileNetV2_CutMix.npy",
        #"./y_prob_seed98_MobileNetV2-Mixup.npy",

        #"./y_prob_MobileNetV3_baseline_98.npy",
        #"./y_prob_MobileNet-V3_AA_seed98.npy",
        #"./y_prob_MobileNetV3_RA_seed98.npy",
        #"./y_prob_MobileNet_V3_TA_seed98.npy",
        #"./y_prob_seed98_MobileNetV3_Cutout.npy",
        #"./y_prob_seed98_MobileNetV3_CutMix.npy",
        #"./y_prob_seed98_MobileNetV3-Mixup.npy",

        #"./y_prob_Efficient-B0_baseline_98.npy",
        #"./y_prob_Efficient-B0_AA_seed98.npy",
        #"./y_prob_Efficient-B0_RA_seed98.npy",
        #"./y_prob_Efficient-B0_TA_seed98.npy",
        #"./y_prob_seed98_Efficient-B0_Cutout.npy",
        #"./y_prob_seed98_Efficient-B0_CutMix.npy",
        #"./y_prob_seed98_Efficient-B0-Mixup.npy",

        #"./y_prob_Exotropia_Efficient-B0_baseline_98.npy",
        #"./y_prob_Exotropia_Efficient-B0_AA_seed98.npy",
        #"./y_prob_Exotropia_Efficient-B0_RA_seed98.npy",
        #"./y_prob_Exotropia_Efficient-B0_TA_seed98.npy",
        #"./y_prob_seed98_Exotropia_Efficient-B0_Cutout.npy",
        #"./y_prob_seed98_Exotropia_Efficient-B0_CutMix.npy",
        #"./y_prob_seed98_Exotropia_Efficient-B0-Mixup.npy",

        #"./y_prob_Exotropia_MobileNetV2_baseline_98.npy",
        #"./y_prob_Exotropia_MobileNet-V2_AA_seed98.npy",
        #"./y_prob_Exotropia_MobileNetV2_RA_seed98.npy",
        #"./y_prob_Exotropia_MobilenetV2_TA_seed98.npy",
        #"./y_prob_seed98_Exotropia_MobileNetV2_Cutout.npy",
        #"./y_prob_seed98_Exotropia_MobileNetV2_CutMix.npy",
        #"./y_prob_seed98_Exotropia_MobileNetV2-Mixup.npy",
        
        "./y_prob_Exotropia_MobileNetV3_baseline_98.npy",
        "./y_prob_Exotropia_MobileNet-V3_AA_seed98.npy",
        "./y_prob_Exotropia_MobileNetV3_RA_seed98.npy",
        "./y_prob_Exotropia_MobileNetV3_TA_seed98.npy",
        "./y_prob_seed98_Exotropia_MobileNetV3_Cutout.npy",
        "./y_prob_seed98_Exotropia_MobileNetV3_CutMix.npy",
        "./y_prob_seed98_Exotropia_MobileNetV3-Mixup.npy",

    ]
    y_true_ = "./y_true_seed98.npy"
    all_results = []

    for model in models:
        # Load arrays
        y_prob = np.load(model)
        y_true = np.load(y_true_)

        # Default values in case of multiclass
        rel = res = unc = None

        # If y_prob is binary
        if y_prob.ndim == 1 or y_prob.shape[1] == 1:
            if y_prob.ndim > 1:
                y_prob = y_prob.ravel()
            brier, rel, res, unc = brier_decomposition(y_true, y_prob)
            print(f"Brier Score (binary): {brier:.4f}")
            print(f"  Reliability: {rel:.4f}")
            print(f"  Resolution : {res:.4f}")
            print(f"  Uncertainty: {unc:.4f}")
        elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
            # Binary case with two-class softmax: take prob of positive class
            y_prob = y_prob[:, 1]
            brier, rel, res, unc = brier_decomposition(y_true, y_prob)
            print(f"Brier Score (binary, 2-col softmax): {brier:.4f}")
            print(f"  Reliability: {rel:.4f}")
            print(f"  Resolution : {res:.4f}")
            print(f"  Uncertainty: {unc:.4f}")
        else:
            # Multiclass case (no decomposition here)
            brier = brier_score_multiclass(y_true, y_prob)
            print(f"Brier Score (multiclass): {brier:.4f}")

        results = {
            "model": model,
            "brier_score": float(brier),
            "reliability": float(rel) if rel is not None else None,
           "resolution": float(res) if res is not None else None,
            "uncertainty": float(unc) if unc is not None else None
        }
        all_results.append(results)

    # Save once after the loop
    with open("Exotropia_brier_decomposition_seed98_MobileNetV3.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()