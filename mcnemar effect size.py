import numpy as np
from scipy.stats import binom
import math
import json


def mcnemar(y_pred_A, y_pred_B, y_true):
    # Build correctness arrays
    correct_A = (y_pred_A == y_true)
    correct_B = (y_pred_B == y_true)

    # Contingency counts
    n01 = np.sum((correct_A == 1) & (correct_B == 0))  # A correct, B wrong
    n10 = np.sum((correct_A == 0) & (correct_B == 1))  # A wrong, B correct
    n = n01 + n10  # Total discordant pairs
    k = min(n01, n10)
    N = len(y_true)

    # Exact binomial test (two-sided)
    p_exact = 2 * binom.cdf(k, n, 0.5)
    p_exact = min(p_exact, 1.0)

    # Mid-p adjustment
    p_mid = p_exact - 0.5 * binom.pmf(k, n, 0.5)
    p_mid = max(p_mid, 0.0)

    # Effect sizes
    if n01 == 0 and n10 == 0:
        OR, CI_low, CI_high = None, None, None
    elif n01 == 0:
        OR, CI_low, CI_high = float('inf'), None, None
    elif n10 == 0:
        OR, CI_low, CI_high = 0.0, None, None
    else:
        OR = n10 / n01
        SE = math.sqrt(1/n01 + 1/n10)
        CI_low = math.exp(math.log(OR) - 1.96 * SE)
        CI_high = math.exp(math.log(OR) + 1.96 * SE)

    # Proportion difference (delta)
    delta = (n10 - n01) / N

    # Cohen’s g (effect size for paired proportions)
    g = abs(n10 - n01) / N

    return {
        "n01": int(n01),
        "n10": int(n10),
        "discordant_total": int(n),
        "p_exact": float(p_exact),
        "p_mid": float(p_mid),
        "odds_ratio": OR,
        "ci_low": CI_low,
        "ci_high": CI_high,
        "delta": float(delta),
        "cohens_g": float(g),
    }

if __name__ == "__main__":
    files = [
        
        #"./y_pred_MobileNetV2_baseline_98.npy",
        #"./y_pred_MobileNet-V2_AA_seed98.npy",
        #"./y_pred_MobileNetV2_RA_seed98.npy",
        #"./y_pred_MobileNet_V2_TA_seed98.npy",
        #"./y_pred_seed98_MobileNetV2_Cutout.npy",
        #"./y_pred_seed98_MobileNetV2_CutMix.npy",
        #"./y_pred_seed98_MobileNetV2-Mixup.npy",

        #"./y_pred_MobileNetV3_baseline_98.npy",
        #"./y_pred_MobileNet-V3_AA_seed98.npy",
        #"./y_pred_MobileNetV3_RA_seed98.npy",
        #"./y_pred_MobileNet_V3_TA_seed98.npy",
        #"./y_pred_seed98_MobileNetV3_Cutout.npy",
        #"./y_pred_seed98_MobileNetV3_CutMix.npy",
        #"./y_pred_seed98_MobileNetV3-Mixup.npy",

        #"./y_pred_Efficient-B0_baseline_98.npy",
        #"./y_pred_Efficient-B0_AA_seed98.npy",
        #"./y_pred_Efficient-B0_RA_seed98.npy",
        #"./y_pred_Efficient-B0_TA_seed98.npy",
        #"./y_pred_seed98_Efficient-B0_Cutout.npy",
        #"./y_pred_seed98_Efficient-B0_CutMix.npy",
        #"./y_pred_seed98_Efficient-B0-Mixup.npy",

        #"./y_pred_Exotropia_Efficient-B0_baseline_98.npy",
        #"./y_pred_Exotropia_Efficient-B0_AA_seed98.npy",
        #"./y_pred_Exotropia_Efficient-B0_RA_seed98.npy",
        #"./y_pred_Exotropia_Efficient-B0_TA_seed98.npy",
        #"./y_pred_seed98_Exotropia_Efficient-B0_Cutout.npy",
        #"./y_pred_seed98_Exotropia_Efficient-B0_CutMix.npy",
        #"./y_pred_seed98_Exotropia_Efficient-B0-Mixup.npy",

        #"./y_pred_Exotropia_MobileNetV2_baseline_98.npy",
        #"./y_pred_Exotropia_MobileNet-V2_AA_seed98.npy",
        #"./y_pred_Exotropia_MobileNetV2_RA_seed98.npy",
        #"./y_pred_Exotropia_MobilenetV2_TA_seed98.npy",
        #"./y_pred_seed98_Exotropia_MobileNetV2_Cutout.npy",
        #"./y_pred_seed98_Exotropia_MobileNetV2_CutMix.npy",
        #"./y_pred_seed98_Exotropia_MobileNetV2-Mixup.npy",
        
        #"./y_pred_Exotropia_MobileNetV3_baseline_98.npy",
        #"./y_pred_Exotropia_MobileNet-V3_AA_seed98.npy",
        #"./y_pred_Exotropia_MobileNetV3_RA_seed98.npy",
        #"./y_pred_Exotropia_MobileNetV3_TA_seed98.npy",
        #"./y_pred_seed98_Exotropia_MobileNetV3_Cutout.npy",
        #"./y_pred_seed98_Exotropia_MobileNetV3_CutMix.npy",
        #"./y_pred_seed98_Exotropia_MobileNetV3-Mixup.npy",
    ]

    y_true = np.load("./y_true_seed98.npy")
    results = {}

    for i in files:
        for j in files:
            if i == j:
                continue
            y_pred1 = np.load(i)
            y_pred2 = np.load(j)
            comparison_key = f"{i}|{j}"
            results[comparison_key] = mcnemar(y_pred1, y_pred2, y_true)
    with open("Exotropia_Mcnemar_MobileNetV3_98.json", "w") as f:
        json.dump(results, f, indent=4)