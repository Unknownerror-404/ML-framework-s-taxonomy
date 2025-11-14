import json
import numpy as np

models = [
    #"./TA_MobileNet-V3_7.json",
    #"./TA_MobileNet-V3_14.json",
    #"./TA_MobileNet-V3_28.json",
    #"./TA_MobileNet-V3_35.json",
    #"./TA_MobileNet-V3_42.json",
    #"./TA_MobileNet-V3_53.json",
    #"./TA_MobileNet-V3_65.json",
    #"./TA_MobileNet-V3_78.json",
    #"./TA_MobileNet-V3_86.json",
    #"./TA_MobileNet-V3_98.json",
    
    #"./baseline_MobileNet_V2_7.json",
    #"./baseline_MobileNet_V2_14.json",
    #"./baseline_MobileNet_V2_28.json",
    #"./baseline_MobileNet_V2_35.json",
    #"./baseline_MobileNet_V2_42.json",
    #"./baseline_MobileNet_V2_53.json",
    #"./baseline_MobileNet_V2_65.json",
    #"./baseline_MobileNet_V2_78.json",
    #"./baseline_MobileNet_V2_86.json",
    #"./baseline_MobileNet_V2_98.json"

    #"./autoaugment_MobileNet_-23_7.json",
    #"./autoaugment_MobileNet_-23_14.json",
    #"./autoaugment_MobileNet_-23_28.json",
    #"./autoaugment_MobileNet_-23_35.json",
    #"./autoaugment_MobileNet_-23_42.json",
    #"./autoaugment_MobileNet_-23_53.json",
    #"./autoaugment_MobileNet_-23_65.json",
    #"./autoaugment_MobileNet_-23_78.json",
    #"./autoaugment_MobileNet_-23_86.json",
    #"./autoaugment_MobileNet_-23_98.json",

    #"./Randaugment_MobileNetV3_seed7.json",
    #"./Randaugment_MobileNetV3_seed14.json",
    #"./Randaugment_MobileNetV3_seed28.json",
    #"./Randaugment_MobileNetV3_seed35.json",
    #"./Randaugment_MobileNetV3_seed42.json",
    #"./Randaugment_MobileNetV3_seed53.json",
    #"./Randaugment_MobileNetV3_seed65.json",
    #"./Randaugment_MobileNetV3_seed78.json",
    #"./Randaugment_MobileNetV3_seed86.json",
    #"./Randaugment_MobileNetV3_seed98.json",

    #"./Mixup_MobileNet_V2_7.json",
    #"./Mixup_MobileNet_V2_14.json",
    #"./Mixup_MobileNet_V2_28.json",
    #"./Mixup_MobileNet_V2_35.json",
    #"./Mixup_MobileNet_V2_42.json",
    #"./Mixup_MobileNet_V2_53.json",
    #"./Mixup_MobileNet_V2_65.json",
    #"./Mixup_MobileNet_V2_78.json",
    #"./Mixup_MobileNet_V2_86.json",
    #"./Mixup_MobileNet_V2_98.json",

    "./TA_Mobilenet_V2_7.json",
    "./TA_Mobilenet_V2_14.json",
    "./TA_Mobilenet_V2_28.json",
    "./TA_Mobilenet_V2_35.json",
    "./TA_Mobilenet_V2_42.json",
    "./TA_Mobilenet_V2_53.json",
    "./TA_Mobilenet_V2_65.json",
    "./TA_Mobilenet_V2_78.json",
    "./TA_Mobilenet_V2_86.json",
    "./TA_Mobilenet_V2_98.json",

]

all_accuracy = []
all_auc = []
all_precision = []
all_pr_auc = []
all_recall = []
all_f1 = []
all_specificity = []
all_npv = []

# Load metrics from each model
for model in models:
    with open(model, "r") as f:
        data = json.load(f)

    all_accuracy.append(data["metrics"]["accuracy"])
    all_auc.append(data["metrics"]["auc"])
    all_pr_auc.append(data["metrics"]["pr_auc"])
    all_precision.append(data["metrics"]["precision"])
    all_recall.append(data["metrics"]["recall"])
    all_f1.append(data["metrics"]["f1"])
    all_specificity.append(data["metrics"]["specificity"])
    all_npv.append(data["metrics"]["npv"])

metrics_dict = {
    "accuracy": all_accuracy,
    "auc": all_auc,
    "pr_auc": all_pr_auc,
    "precision": all_precision,
    "recall": all_recall,
    "f1": all_f1,
    "specificity": all_specificity,
    "npv": all_npv,
}

avgs = {}
print("Summary across seeds:")
for name, values in metrics_dict.items():
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)

    print(f"{name:12s} mean={mean:.4f}, std={std:.4f}, min={min_val:.4f}, max={max_val:.4f}")

    avgs[name] = {
        "mean": float(mean),
        "std": float(std),
        "min": float(min_val),
        "max": float(max_val),
    }

with open("avg_TA_Mobilenet_V2.json", "w") as f1:
    json.dump(avgs, f1, indent=4)
