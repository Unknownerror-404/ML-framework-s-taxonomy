# Grad-CAM Visualization Scripts

This directory contains Grad-CAM (Gradient-weighted Class Activation Mapping) visualization scripts used to qualitatively inspect spatial attention behaviour across different augmentation strategies and lightweight convolutional neural network architectures.

These visualizations were used to analyze how augmentation policies influence feature localization, attention consistency, and model interpretability during binary strabismus classification.

---

# Supported Architectures

The Grad-CAM scripts support the following pretrained architectures:

- MobileNet-V2
- MobileNet-V3
- EfficientNet-B0

All models utilize pretrained ImageNet weights from torchvision and can optionally load experiment-specific trained checkpoints.

---

# Supported Augmentation Policies

Grad-CAM visualizations were generated for the following augmentation strategies:

| Augmentation | Purpose |
|---|---|
| Baseline | Standard preprocessing without augmentation |
| AutoAugment | Learned policy-based augmentation |
| RandAugment | Randomized augmentation operations |
| TrivialAugment | Lightweight tuning-free augmentation |
| CutMix | Patch-based image mixing |
| Cutout | Random spatial masking |
| Mixup | Linear image interpolation |

---

# Directory Structure

```text
visualization/
└── gradcam/
    ├── efficientnetb0/
    ├── mobilenetv2/
    ├── mobilenetv3/
```

---

# Purpose of the Scripts

The Grad-CAM scripts were developed to:

- visualize spatial attention maps,
- inspect learned feature localization,
- evaluate augmentation-induced attention behaviour,
- analyze qualitative robustness,
- and support interpretability analysis.

These scripts complement the quantitative benchmarking results reported in the manuscript.

---

# General Workflow

Each Grad-CAM script typically performs the following operations:

1. Load pretrained or fine-tuned model
2. Load target image
3. Apply preprocessing and augmentation
4. Register forward and backward hooks
5. Generate Grad-CAM activation maps
6. Resize heatmaps
7. Overlay attention maps onto original images
8. Save visualization outputs

---

# Grad-CAM Methodology

Grad-CAM generates class-specific activation maps using:

- feature activations from the target convolutional layer,
- gradients flowing into that layer during backpropagation.

The resulting heatmap highlights image regions contributing most strongly to model predictions.

---

# Augmentation-Specific Visualization Behaviour

## AutoAugment

Scripts using AutoAugment apply learned augmentation policies before Grad-CAM generation.

Purpose:
- inspect robustness under policy-based transformations,
- evaluate feature localization consistency after augmentation.

Example:
```text
gradcam_efficientnetb0_autoaugment.py
```

---

## RandAugment

Applies randomized augmentation operations before attention visualization.

Purpose:
- evaluate stability under stochastic transformations.

---

## TrivialAugment

Applies lightweight random augmentation operations.

Purpose:
- analyze simplified augmentation-induced attention shifts.

---

## CutMix

Combines image patches from two separate images before Grad-CAM analysis.

Purpose:
- inspect fragmented spatial attention,
- evaluate localization robustness under mixed-image supervision.

Example:
```text
gradcam_mobilenetv2_cutmix.py
```

The script:
- randomly selects a second image,
- applies CutMix patch replacement,
- generates Grad-CAM maps on the mixed sample,
- and visualizes resulting attention distributions.

---

## Cutout

Masks random spatial regions prior to Grad-CAM generation.

Purpose:
- evaluate model reliance on partial visual cues.

---

## Mixup

Linearly interpolates image tensors and labels before visualization.

Purpose:
- inspect smoothed feature attention and distributed spatial representations.

---

# Output Visualizations

Scripts generally produce three outputs:

| Output | Description |
|---|---|
| Original Image | Input image after preprocessing |
| Grad-CAM Heatmap | Attention intensity visualization |
| Overlay Image | Combined heatmap and original image |

Saved outputs are typically exported as PNG images.

Example:
```text
grad_cam_result_V2_Cutmix_40.png
```

---

# Target Layers

Grad-CAM visualizations are generated using architecture-specific convolutional layers.

Example target layers:

| Architecture | Target Layer |
|---|---|
| EfficientNet-B0 | features.6 |
| MobileNet-V2 | features.18 |
| MobileNet-V3 | architecture-specific final feature layer |

These layers were selected to capture high-level semantic representations.

---

# Reproducibility Notes

Visualization scripts:
- use deterministic preprocessing where applicable,
- maintain consistent image resizing,
- and preserve normalization settings across experiments.

Some augmentation policies (e.g., CutMix, RandAugment) introduce stochastic behaviour through randomized transformations.

---

# Scientific Relevance

These visualization scripts support:
- qualitative interpretability analysis,
- augmentation robustness inspection,
- feature localization evaluation,
- and augmentation-policy comparison.

The Grad-CAM outputs provide supplementary evidence alongside:
- ROC-AUC,
- PR-AUC,
- calibration analysis,
- and statistical significance testing.

---

# Notes

These scripts are intended for:
- research visualization,
- qualitative analysis,
- and interpretability support.

They are not optimized for real-time inference or production deployment.