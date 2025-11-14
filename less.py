import os

# Set the folder path
folder_path = "./padded/train/padded_output_class1"

# List of image file extensions to count
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

# Count images
image_count = sum(1 for file in os.listdir(folder_path) if file.lower().endswith(image_extensions))

print(f"Number of images in the folder: {image_count}")