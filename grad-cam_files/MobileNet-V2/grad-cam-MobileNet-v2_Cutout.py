import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from torchvision.transforms import RandomErasing
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


# =========================================================
# 1. Load Pretrained EfficientNet-B0 (ImageNet-1K)
# =========================================================
weights = MobileNet_V2_Weights.IMAGENET1K_V1
model = mobilenet_v2(weights=weights)
model.eval()

TARGET_LAYER = "features.18"

# =========================================================
# 2. CUTOUT (RandomErasing) + Normalization
# =========================================================

MASK_SIZE = 50                
IMG_RES = 224

cutout_scale = ( (MASK_SIZE*MASK_SIZE) / (IMG_RES*IMG_RES) )

transform = transforms.Compose([
    transforms.ToTensor(),
    # -----------------------
    # CUTOUT here
    # -----------------------
    RandomErasing(
        p=1.0,
        scale=(cutout_scale, cutout_scale),
        ratio=(1.0, 1.0),
        value=0
    ),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])


def load_image(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return img, tensor


# =========================================================
# 3. GRAD-CAM CLASS
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

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)

    def generate(self, x, target_class=None):
        out = self.model(x)

        if target_class is None:
            target_class = out.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = out[0, target_class]
        class_score.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# =========================================================
# 4. Run Grad-CAM on One Image
# =========================================================
img_path = r"ML-framework-s-taxonomy/padded/padded_output_class3_train_exotropia/40.jpg"
orig_img, input_tensor = load_image(img_path)

gradcam = GradCAM(model, TARGET_LAYER)
cam = gradcam.generate(input_tensor)

cam_resized = cv2.resize(cam, (224, 224))


# =========================================================
# 5. Overlay Heatmap
# =========================================================
heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255

orig_resized = cv2.resize(np.array(orig_img), (224, 224)) / 255.0
overlay = heatmap * 0.5 + orig_resized * 0.5
overlay = overlay / overlay.max()


# =========================================================
# 6. Save Results
# =========================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(orig_resized)

plt.subplot(1, 3, 2)
plt.title("Grad-CAM Heatmap")
plt.imshow(cam_resized, cmap="jet")

plt.subplot(1, 3, 3)
plt.title("Grad-CAM Overlay")
plt.imshow(overlay)

plt.tight_layout()
plt.savefig("./grad_cam_result_Cutout_40.png")
plt.close()