import os
from PIL import Image, ImageOps
import torchvision.transforms as transforms

def resize_with_padding(img, target_size=224):
    """
    Resize image keeping aspect ratio, then pad to square.
    """
    old_size = img.size  # (width, height)
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)

    # Pad to square
    delta_w = target_size - new_size[0]
    delta_h = target_size - new_size[1]
    padding = (delta_w // 2, delta_h // 2,
               delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    img = ImageOps.expand(img, padding, fill=0)  # black padding
    return img

# Transform pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Input images loop
for i in range(100, 105):
    if i in [29]:
        continue
    else:
        img_path = f"./data/train_base_2/{i}.jpg"
        img = Image.open(img_path).convert("RGB")

        # Apply resize+padding ONCE
        padded_img = resize_with_padding(img, 224)

        # Save the padded image
        output_dir = "./padded/validation/padded_class3"
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        padded_img.save(output_path)
        print(f"Saved processed image at {output_path}")

        # Convert to tensor for model input
        tensor = transform(padded_img).unsqueeze(0)