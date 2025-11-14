import numpy as np
from scipy import stats
import json
from rounding_delong import rounding_delong as rd

# ============================================================
# Core DeLong implementation (fast version)
# ============================================================

def compute_midrank(x):
    """Computes midranks for ties (used in DeLong)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    Fast implementation of DeLong's algorithm for two correlated ROC AUCs.
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    tx = np.array([compute_midrank(x) for x in positive_examples])
    ty = np.array([compute_midrank(x) for x in negative_examples])
    tz = np.array([compute_midrank(x) for x in predictions_sorted_transposed])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01, rowvar=False)
    sy = np.cov(v10, rowvar=False)

    if np.ndim(sx) == 0:
        sx = np.array([[sx]])
    if np.ndim(sy) == 0:
        sy = np.array([[sy]])

    delongcov = sx / m + sy / n
    return aucs, delongcov

# ============================================================
# DeLong test wrapper (deterministic bootstrap fallback)
# ============================================================

def delong_roc_test(y_true, y_pred1, y_pred2, n_bootstrap=2000, random_state=63):
    """
    Compare two ROC AUCs using DeLong's test with a deterministic bootstrap fallback.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred1, y_pred2 : array-like of shape (n_samples,)
        Prediction scores for model 1 and 2.
    n_bootstrap : int, default=2000
        Number of bootstrap replicates for fallback if variance <= 0.
    random_state : int, default=63
        Fixed seed for reproducibility.

    Returns
    -------
    pvalue : float
        Two-sided p-value for H0: AUC1 == AUC2.
    """

    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)

    assert len(y_true) == len(y_pred1) == len(y_pred2), "Mismatched lengths"
    assert set(np.unique(y_true)) <= {0, 1}, "Only binary labels allowed"

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)
    if m == 0 or n == 0:
        raise ValueError("Need both positive and negative samples")

    # Arrange with positives first, negatives second
    order = np.concatenate([pos_idx, neg_idx])
    preds = np.vstack([y_pred1, y_pred2])[:, order]

    aucs, cov = fastDeLong(preds, m)

    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]

    # fallback if variance invalid
    if var <= 0 or np.isclose(diff, 0):
        rng = np.random.default_rng(seed=random_state)
        diffs = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            order_bs = np.concatenate([np.where(y_true[idx] == 1)[0],
                                       np.where(y_true[idx] == 0)[0]])
            preds_bs = np.vstack([y_pred1[idx], y_pred2[idx]])[:, order_bs]
            try:
                aucs_bs, _ = fastDeLong(preds_bs, (y_true[idx] == 1).sum())
                diffs.append(aucs_bs[0] - aucs_bs[1])
            except Exception:
                continue
        if len(diffs) == 0:
            return 1.0
        diffs = np.array(diffs)
        p_boot = (np.sum(np.abs(diffs) >= np.abs(diff)) + 1) / (len(diffs) + 1)
        return p_boot

    z = np.abs(diff) / np.sqrt(var)
    pvalue = 2 * (1 - stats.norm.cdf(z))
    return pvalue

if __name__ == "__main__":
    #rng = np.random.default_rng(86)
    #y_true = np.array([0]*50 + [1]*50)

    #y_pred1 = np.concatenate([rng.normal(0.2, 0.1, 50), rng.normal(0.86, 0.1, 50)])

    #y_pred2 = np.concatenate([rng.normal(0.86, 0.1, 50), rng.normal(0.86, 0.1, 50)])
    #parser = argparse.ArgumentParser(description="Calculate Brier score")
    #parser.add_argument("--model1", type=str, required=True,
                        #help="Path to predicted probabilities (.npy file)")
    #parser.add_argument("--y_true", type=str, required=True,
                        #help="Path to true labels (.npy file)")
    #parser.add_argument("--model2", type=str, required=True,
                        #help="Path to predicted probabilities (.npy file)")
    #args = parser.parse_args()

    # Load arrays

    # list of probabilities
    files = [

        #"./y_prob_MobileNetV2_baseline_7.npy",
        #"./y_prob_MobileNet-V2_AA_seed7.npy",
        #"./y_prob_MobileNetV2_RA_seed7.npy",
        #"./y_prob_MobileNet_V2_TA_seed7.npy",
        #"./y_prob_seed7_MobileNetV2_Cutout.npy",
        #"./y_prob_seed7_MobileNetV2_CutMix.npy",
        #"./y_prob_seed7_MobileNetV2-Mixup.npy",

        #"./y_prob_MobileNetV3_baseline_7.npy",
        #"./y_prob_MobileNet-V3_AA_seed7.npy",
        #"./y_prob_MobileNetV3_RA_seed7.npy",
        #"./y_prob_MobileNet_V3_TA_seed7.npy",
        #"./y_prob_seed7_MobileNetV3_Cutout.npy",
        #"./y_prob_seed7_MobileNetV3_CutMix.npy",
        #"./y_prob_seed7_MobileNetV3-Mixup.npy",

        #"./y_prob_Efficient-B0_baseline_7.npy",
        #"./y_prob_Efficient-B0_AA_seed7.npy",
        #"./y_prob_Efficient-B0_RA_seed7.npy",
        #"./y_prob_Efficient-B0_TA_seed7.npy",
        #"./y_prob_seed7_Efficient-B0_Cutout.npy",
        #"./y_prob_seed7_Efficient-B0_CutMix.npy",
        #"./y_prob_seed7_Efficient-B0-Mixup.npy",

        #"./y_prob_Exotropia_Efficient-B0_baseline_7.npy",
        #"./y_prob_Exotropia_Efficient-B0_AA_seed7.npy",
        #"./y_prob_Exotropia_Efficient-B0_RA_seed7.npy",
        #"./y_prob_Exotropia_Efficient-B0_TA_seed7.npy",
        #"./y_prob_seed7_Exotropia_Efficient-B0_Cutout.npy",
        #"./y_prob_seed7_Exotropia_Efficient-B0_CutMix.npy",
        #"./y_prob_seed7_Exotropia_Efficient-B0-Mixup.npy",

        #"./y_prob_Exotropia_MobileNetV2_baseline_7.npy",
        #"./y_prob_Exotropia_MobileNet-V2_AA_seed7.npy",
        #"./y_prob_Exotropia_MobileNetV2_RA_seed7.npy",
        #"./y_prob_Exotropia_MobilenetV2_TA_seed7.npy",
        #"./y_prob_seed7_Exotropia_MobileNetV2_Cutout.npy",
        #"./y_prob_seed7_Exotropia_MobileNetV2_CutMix.npy",
        #"./y_prob_seed7_Exotropia_MobileNetV2-Mixup.npy",
        
        #"./y_prob_Exotropia_MobileNetV3_baseline_7.npy",
        #"./y_prob_Exotropia_MobileNet-V3_AA_seed7.npy",
        #"./y_prob_Exotropia_MobileNetV3_RA_seed7.npy",
        #"./y_prob_Exotropia_MobileNetV3_TA_seed7.npy",
        #"./y_prob_seed7_Exotropia_MobileNetV3_Cutout.npy",
        #"./y_prob_seed7_Exotropia_MobileNetV3_CutMix.npy",
        #"./y_prob_seed7_Exotropia_MobileNetV3-Mixup.npy",

    ]

    y_true = np.load("./y_true_seed7.npy")
    results = {}
    for i in range(len(files)):
        for j in range(len(files)):
            y_pred1 = np.load(files[i])
            y_pred2 = np.load(files[j])
            pval = delong_roc_test(y_true, y_pred1, y_pred2, random_state=63)
            results[f"{files[i]}|{files[j]}"] = pval

    with open("z_temp.json", "w") as f:
        json.dump(results, f, indent=4)

    rd("./z_temp.json", "./Exotropia_Delong_results_Efficient-B0_seed7.json")