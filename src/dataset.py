import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from src.utils import rgb_to_lab_tensor
import torchvision.transforms as T

class ImagePairDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=(224, 224), hist_bins=32):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.hist_bins = hist_bins

        self.black_dir = os.path.join(root_dir, f"{split}_black")
        self.color_dir = os.path.join(root_dir, f"{split}_color")

        self.files = sorted(os.listdir(self.black_dir))

        self.transform = T.Resize(img_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        black_path = os.path.join(self.black_dir, self.files[idx])
        color_path = os.path.join(self.color_dir, self.files[idx])

        black_img = Image.open(black_path).convert("L")
        color_img = Image.open(color_path).convert("RGB")

        black_img = self.transform(black_img)
        color_img = self.transform(color_img)

        L = rgb_to_lab_tensor(black_img)      # 1xHxW
        lab = self.rgb_to_lab_tensor_full(color_img)  # 2xHxW

        hist = torch.zeros(self.hist_bins)  # placeholder

        return L.float(), lab.float(), hist.float()

    @staticmethod
    def rgb_to_lab_tensor_full(img):
        import numpy as np
        from skimage import color
        arr = np.array(img).astype(np.float32)/255.0
        lab = color.rgb2lab(arr)
        L = lab[:,:,0][None,...]/100.0
        ab = lab[:,:,1:3].transpose(2,0,1)/127.0
        return torch.from_numpy(ab)
