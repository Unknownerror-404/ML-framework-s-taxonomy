import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os

# =========================================================
# 1. Load Pretrained EfficientNet-B0 (ImageNet-1K)
# =========================================================
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
model.eval()

TARGET_LAYER = "features.6"

# =========================================================
# 2. Base Transform 
# =========================================================
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image_tensor(path):
    img = Image.open(path).convert("RGB")
    return img, base_transform(img)

# =========================================================
# 3. CUTMIX function for ONE visualization
# =========================================================
def apply_cutmix(t1, t2, alpha=1.0):
    """
    Apply CutMix to two image tensors of shape [3, H, W]
    """
    _, H, W = t1.shape

    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1. - lam)

    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed = t1.clone()
    mixed[:, y1:y2, x1:x2] = t2[:, y1:y2, x1:x2]

    return mixed.unsqueeze(0), lam


# =========================================================
# 4. Grad-CAM Class
# =========================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, m in self.model.named_modules():
            if name == self.target_layer:
                m.register_forward_hook(forward_hook)
                m.register_full_backward_hook(backward_hook)

    def generate(self, x, target_class=None):
        out = self.model(x)

        if target_class is None:
            target_class = out.argmax(dim=1).item()

        self.model.zero_grad()
        out[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# =========================================================
# 5. Load Two Images for CutMix
# =========================================================
import os
import random

img_path1 = r"ML-framework-s-taxonomy/padded/padded_output_class3_train_exotropia/40.jpg"
filename1 = os.path.basename(img_path1)

folder = os.path.dirname(img_path1)

# list full paths
files = [
    os.path.join(folder, f)
    for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".png"))
]

# remove matching file by filename
files = [f for f in files if os.path.basename(f) != filename1]

if len(files) == 0:
    raise ValueError("No other images found for CutMix!")

# pick second image
img_path2 = random.choice(files)

# now load tensors
orig1, t1 = load_image_tensor(img_path1)
orig2, t2 = load_image_tensor(img_path2)

# apply CutMix
mixed_tensor, lam = apply_cutmix(t1, t2)

print(f"CutMix applied with lambda = {lam:.3f}")

# =========================================================
# 6. Run Grad-CAM
# =========================================================
gradcam = GradCAM(model, TARGET_LAYER)
cam = gradcam.generate(mixed_tensor)
cam_resized = cv2.resize(cam, (224, 224))


# =========================================================
# 7. Overlay Heatmap
# =========================================================
orig_resized = cv2.resize(np.array(orig1), (224, 224)) / 255.0
heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
heatmap = heatmap.astype(np.float32) / 255

overlay = 0.5 * orig_resized + 0.5 * heatmap
overlay = overlay / overlay.max()


# =========================================================
# 8. Save Output
# =========================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(orig_resized)

plt.subplot(1, 3, 2)
plt.title("Grad-CAM Heatmap")
plt.imshow(cam_resized, cmap="jet")

plt.subplot(1, 3, 3)
plt.title("Grad-CAM Overlay (CutMix)")
plt.imshow(overlay)

plt.tight_layout()
plt.savefig("./grad_cam_result_Cutmix_40.png")
plt.close()