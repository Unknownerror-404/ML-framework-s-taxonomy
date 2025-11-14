import os
import shutil
import numpy as np
from torchvision import datasets

def stratified_split_and_save(dataset_dir, output_dir, fractions):
    """
    Splits ImageFolder dataset into stratified subsets (per class) and saves them.

    Args:
        dataset_dir: Path to original dataset (e.g., data/train).
        output_dir: Where to save the splits.
        fractions: List of fractions (e.g., [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).
    """
    dataset = datasets.ImageFolder(dataset_dir)
    class_to_idx = dataset.class_to_idx

    # Collect file paths per class
    class_files = {cls: [] for cls in class_to_idx}
    for filepath, label in dataset.samples:
        cls_name = dataset.classes[label]
        class_files[cls_name].append(filepath)

    # For each class, shuffle and split
    for cls_name, files in class_files.items():
        np.random.shuffle(files)
        start = 0
        for i, frac in enumerate(fractions):
            n = int(len(files) * frac)
            subset_files = files[start:start+n]
            start += n

            # Make subset folder
            subset_dir = os.path.join(output_dir, f"split_{int(frac*100)}", cls_name)
            os.makedirs(subset_dir, exist_ok=True)

            # Copy files
            for f in subset_files:
                shutil.copy(f, subset_dir)

    print(f"Saved splits into: {output_dir}")


# Example usage:
fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
if __name__ == "__main__":
    stratified_split_and_save("./data/train", "data/splits", fractions)