import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Large_Weights
from PIL import Image
import numpy as np

# --- Configuration ---
input_folder = 'C:/Users/HP/Desktop/CODE-Green/ML taxonomy/ML-framework-s-taxonomy/padded/train/padded_output_class3'  # Make sure this folder contains images
output_folder = './gradcam_results_on_mobilenet_v3/class3'
os.makedirs(output_folder, exist_ok=True)

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --- Load MobileNetV3-Large model ---
model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
target_layer_name = 'features.15'  # Last convolutional block
model = model.to(device)
model.eval()

# --- Image transforms ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Grad-CAM Implementation ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(self._forward_hook)
                module.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        grad_cam_map = torch.sum(weights * self.activations, dim=1)[0]
        grad_cam_map = F.relu(grad_cam_map)
        grad_cam_map = grad_cam_map / (grad_cam_map.max() + 1e-8)
        return grad_cam_map.cpu().numpy()

# --- Overlay Grad-CAM ---
def overlay_gradcam(img_path, gradcam_map, output_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Failed to read {img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(gradcam_map, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# --- Process images ---
gradcam = GradCAM(model, target_layer_name)

# List images in input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    print(f"No images found in {input_folder}. Please check the folder path and image extensions.")
else:
    print(f"Found {len(image_files)} images. Processing...")

for img_name in image_files:
    img_path = os.path.join(input_folder, img_name)
    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Failed to open {img_name}: {e}")
        continue

    print(f"Loaded {img_name}, size: {image.size}")

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Get predicted class
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
    print(f"Predicted class for {img_name}: {pred_class}")

    # Generate Grad-CAM
    cam = gradcam.generate(input_tensor, class_idx=pred_class)

    # Save overlay
    output_path = os.path.join(output_folder, f'gradcam_{img_name}')
    overlay_gradcam(img_path, cam, output_path)
    print(f"Saved Grad-CAM for {img_name} -> {output_path}")

print("Processing complete.")