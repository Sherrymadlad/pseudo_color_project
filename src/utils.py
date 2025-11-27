import os
import numpy as np
from PIL import Image
from skimage import color
import torch

# Verify dataset structure
def verify_dataset_pairs(root_dir):
    required = ["train_black", "train_color", "test_black", "test_color"]
    for r in required:
        if not os.path.exists(os.path.join(root_dir, r)):
            print(f"Missing {r}")
            return False
    return True

# Convert LAB to RGB numpy
def lab_to_rgb_np(L, a, b):
    lab = np.stack([L, a, b], axis=-1).astype(np.float32)
    rgb = color.lab2rgb(lab)
    rgb = np.clip(rgb*255,0,255).astype(np.uint8)
    return rgb

# Convert grayscale PIL → normalized L tensor
def rgb_to_lab_tensor(img):
    if img.mode != "L":
        img = img.convert("L")
    arr = np.array(img).astype(np.float32)/255.0
    return torch.from_numpy(arr[None,...])
