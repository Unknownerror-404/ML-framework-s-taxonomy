import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torchvision.transforms as transforms
from torchvision.transforms import TrivialAugmentWide
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# -------------------------
# 1. Load Pretrained Model
# -------------------------
weights = MobileNet_V2_Weights.IMAGENET1K_V1
model = mobilenet_v2(weights=weights)
model.eval()

# EfficientNet-B0's final conv layer name:
TARGET_LAYER = "features.18" 

# -------------------------
# 2. Preprocessing + AutoAugment
# -------------------------
transform = transforms.Compose([
    TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    return img, tensor

# -------------------------
# 3. Grad-CAM
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# -------------------------
# 4. Run Grad-CAM
# -------------------------
#for i in range(40):
#i = 40
img_path = r"ML-framework-s-taxonomy/padded/padded_output_class3_train_exotropia/40.jpg"
orig_img, input_tensor = load_image(img_path)

grad_cam = GradCAM(model, TARGET_LAYER)
cam = grad_cam.generate(input_tensor)

cam_resized = cv2.resize(cam, (224, 224))

    # -------------------------
    # 5. Overlay Heatmap
    # -------------------------
heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
orig_img_resized = cv2.resize(np.array(orig_img), (224, 224)) / 255.0
overlay = 0.5 * heatmap + 0.5 * orig_img_resized
overlay = overlay / overlay.max()

    # -------------------------
    # 6. Save Output
    # -------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(orig_img_resized)

plt.subplot(1, 3, 2)
plt.title("Grad-CAM Heatmap")
plt.imshow(cam_resized, cmap="jet")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(overlay)

plt.savefig("./grad_cam_result_TA_40.png")
plt.close()