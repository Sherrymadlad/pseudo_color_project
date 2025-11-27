# evaluate.py
import os
import torch
from torch.utils.data import DataLoader
from src.dataset import ImagePairDataset
from src.model import UNet
from src.utils import lab_to_rgb_np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tqdm import tqdm
import numpy as np
from PIL import Image

def evaluate_model(model_path="models/colorization_best.pth", data_dir="data", img_size=(224,224), batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImagePairDataset(data_dir, split="test", img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    psnr_list = []
    ssim_list = []

    os.makedirs("test_results", exist_ok=True)

    for i, (L, ab_true, _) in enumerate(tqdm(loader)):
        L = L.to(device)
        with torch.no_grad():
            ab_pred = model(L).cpu().numpy()
        L_np = (L.cpu().numpy()[:,0]*100)
        for j in range(L_np.shape[0]):
            pred_rgb = lab_to_rgb_np(L_np[j], ab_pred[j,0]*127, ab_pred[j,1]*127)
            true_rgb = lab_to_rgb_np(L_np[j], ab_true[j,0].numpy()*127, ab_true[j,1].numpy()*127)
            
            psnr_list.append(psnr(true_rgb, pred_rgb))
            ssim_list.append(ssim(true_rgb, pred_rgb, multichannel=True))

            # Save side-by-side image
            combined = np.hstack([true_rgb, pred_rgb])
            Image.fromarray(combined).save(f"test_results/result_{i*batch_size+j}.png")
    
    print(f"Average PSNR: {np.mean(psnr_list):.4f}")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")

if __name__=="__main__":
    evaluate_model()
