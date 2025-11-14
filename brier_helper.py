import numpy as np
import pandas as pd
from scipy import stats

#---Brier scores---#
results = {
    "AutoAugment": [0.2500, 0.2000, 0.1500, 0.1500, 0.2000, 0.1000, 0.2000, 0.3000, 0.0500, 0.0500],
    "TrivialAugment": [0.2500, 0.0500, 0.2500, 0.1500, 0.0500, 0.0500, 0.1000, 0.1000, 0.1000, 0.1000],
    "RandAugment": [0.2000, 0.2500, 0.1500, 0.1000, 0.1000, 0.1500, 0.2000, 0.4000, 0.1000, 0.1000],
    "CutMix": [0.2000, 0.0500, 0.1000, 0.1500, 0.1500, 0.0500, 0.4000, 0.1500, 0.1000, 0.1500],
    "Mixup": [0.1000, 0.0500, 0.2000, 0.1500, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
    "Cutout": [0.1500, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.3000, 0.0500, 0.1000, 0.1500],
    "Baseline": [0.1500, 0.1000, 0.2500, 0.2000, 0.2000, 0.2500, 0.1000, 0.1500, 0.2500, 0.0500]
}

rows = []
for method, scores in results.items():
    arr = np.array(scores)
    mean = arr.mean()
    std = arr.std(ddof=1)
    sem = std / np.sqrt(len(arr))
    # 95% CI using t-distribution (safer for small n)
    ci = stats.t.interval(0.95, len(arr)-1, loc=mean, scale=sem)
    rows.append([method, mean, std, sem, ci])

df = pd.DataFrame(rows, columns=["Method", "Mean", "Std", "SEM", "95% CI"])
print(df)