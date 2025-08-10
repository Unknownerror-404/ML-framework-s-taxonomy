import numpy as np

# Fill with your metric values, e.g., AUCs across seeds
scores = []

if len(scores) == 0:
    print("No scores provided.")
else:
    ddof = 1 if len(scores) > 1 else 0
    mean = np.mean(scores)
    std = np.std(scores, ddof=ddof)
    print(f"Mean: {mean:.4f}")
    print(f"Standard Deviation: {std:.4f}")
    print(f"Reported as: {mean:.4f} ± {std:.4f}")