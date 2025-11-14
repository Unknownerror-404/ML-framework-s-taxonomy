import numpy as np
from scipy.stats import binom

# Load predictions and ground truth
y_true = np.load("y_true.npy")
y_pred_A = np.load("y_pred_baseline53.npy")
y_pred_B = np.load("y_pred_seed53_Cutout.npy")

# Build correctness arrays
correct_A = (y_pred_A == y_true)
correct_B = (y_pred_B == y_true)

# Contingency counts
n01 = np.sum((correct_A == 1) & (correct_B == 0))  # A correct, B wrong
n10 = np.sum((correct_A == 0) & (correct_B == 1))  # A wrong, B correct

# Total discordant pairs
n = n01 + n10
k = min(n01, n10)

# Exact binomial test (two-sided)
p_exact = 2 * binom.cdf(k, n, 0.5)
p_exact = min(p_exact, 1.0)

# Mid-p adjustment
p_mid = p_exact - 0.5 * binom.pmf(k, n, 0.5)
p_mid = max(p_mid, 0.0)

print(f"Discordant pairs: n01={n01}, n10={n10}, total={n}")
print(f"Exact p-value: {p_exact:.6f}")
print(f"Mid-p p-value: {p_mid:.6f}")