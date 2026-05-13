import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import cv2


# =========================================================
# 1. Load Pretrained EfficientNet-B0 (ImageNet-1K)
# =========================================================
weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
model = mobilenet_v3_large(weights=weights)
model.eval()

TARGET_LAYER = "features.16"


# =========================================================
# 2. Mixup Function
# =========================================================
def apply_mixup(x1, x2, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    mixed = lam * x1 + (1 - lam) * x2
    return mixed, lam


# =========================================================
# 3. Transform (No Augment — only Normalize)
# =========================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_image_tensor(path):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0)
    return img, t


# =========================================================
# 4. GRAD-CAM CLASS
# =========================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)

    def generate(self, x):
        output = self.model(x)
        cls = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, cls].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# =========================================================
# 5. Pick 2 Images for Mixup
# =========================================================
img_path1 = r"ML-framework-s-taxonomy/padded/padded_output_class3_train_exotropia/40.jpg"
filename1 = os.path.basename(img_path1)
folder = os.path.dirname(img_path1)

files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
files = [os.path.join(folder, f) for f in files if f != filename1]

img_path2 = random.choice(files)
print("Mixup images:")
print(" • Image A:", img_path1)
print(" • Image B:", img_path2)

orig1, t1 = load_image_tensor(img_path1)
orig2, t2 = load_image_tensor(img_path2)

mixed_tensor, lam = apply_mixup(t1, t2)
print(f"λ (mixup ratio): {lam:.3f}")


# =========================================================
# 6. Grad-CAM on the Mixed Image
# =========================================================
gradcam = GradCAM(model, TARGET_LAYER)
cam = gradcam.generate(mixed_tensor)

cam_resized = cv2.resize(cam, (224, 224))
heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255


# Mixed image for visualization (unnormalized)
mixed_vis = mixed_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
mixed_vis = (mixed_vis - mixed_vis.min()) / (mixed_vis.max() - mixed_vis.min() + 1e-8)

overlay = 0.5 * mixed_vis + 0.5 * heatmap
overlay /= overlay.max()


# =========================================================
# 7. Save Results
# =========================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.title("Image A")
plt.imshow(orig1.resize((224, 224)))

plt.subplot(1, 4, 2)
plt.title("Image B")
plt.imshow(orig2.resize((224, 224)))

plt.subplot(1, 4, 3)
plt.title(f"Mixup λ={lam:.2f}")
plt.imshow(mixed_vis)

plt.subplot(1, 4, 4)
plt.title("Grad-CAM Overlay")
plt.imshow(overlay)

plt.tight_layout()
plt.savefig("./grad_cam_result_V3_Mixup_40.png")
plt.close()

#results were from fig.1 --> fig.40 and fig.2 --> fig. 85 with mixup ratio 0.850