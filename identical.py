import numpy as np
import hashlib
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

# --- load predictions ---
cutmix_preds = np.load("./y_prob_MobileNet_V2_TA_seed7.npy")
mixup_preds = np.load("./y_prob_seed7_MobileNetV2_Cutout.npy")

# --- 1. Check exact equality ---
print("Arrays identical? ", np.array_equal(cutmix_preds, mixup_preds))

# --- 2. Compare file checksums (catch accidental overwrites) ---
def file_checksum(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

print("CutMix checksum:", file_checksum("./y_prob_seed86_Exotropia_MobileNetV3_Cutout.npy"))
print("Mixup checksum:", file_checksum("./y_prob_seed86_Exotropia_MobileNetV3_CutMix.npy"))

# --- 3. Correlation between predictions ---
pearson_corr, _ = pearsonr(cutmix_preds.ravel(), mixup_preds.ravel())
spearman_corr, _ = spearmanr(cutmix_preds.ravel(), mixup_preds.ravel())
print("Pearson correlation:", pearson_corr)
print("Spearman correlation:", spearman_corr)

# --- 4. Difference statistics ---
diff = np.abs(cutmix_preds - mixup_preds)
print("Max abs diff:", diff.max())
print("Mean abs diff:", diff.mean())

# --- 5. Optional: AUC comparison (if you also have y_true labels) ---
# y_true = np.load("./y_true.npy")   # uncomment if you saved ground-truth labels
# print("CutMix AUC:", roc_auc_score(y_true, cutmix_preds))
# print("Mixup AUC:", roc_auc_score(y_true, mixup_preds))
