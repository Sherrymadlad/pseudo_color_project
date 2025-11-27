import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import ImagePairDataset
from src.model import UNet
from src.utils import verify_dataset_pairs, lab_to_rgb_np
import argparse
from tqdm import tqdm
import numpy as np

# Tiny perceptual net
class TinyVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
    def forward(self,x):
        return self.features(x)

def lab_tensors_to_rgb_numpy(L_tensor, ab_tensor):
    L = L_tensor.detach().cpu().numpy()
    ab = ab_tensor.detach().cpu().numpy()
    B = L.shape[0]
    outs = []
    for i in range(B):
        Li = (L[i,0]*100).astype(np.float32)
        ai = ab[i,0]*127
        bi = ab[i,1]*127
        rgb = lab_to_rgb_np(Li, ai, bi)
        outs.append(rgb)
    return np.stack(outs, axis=0)

def compute_perceptual_loss(pred_ab, target_ab, L_tensor, perceptual_net, device):
    with torch.no_grad():
        rgb_pred_np = lab_tensors_to_rgb_numpy(L_tensor, pred_ab)
        rgb_targ_np = lab_tensors_to_rgb_numpy(L_tensor, target_ab)
    pred_tensor = torch.from_numpy(rgb_pred_np/255.0).permute(0,3,1,2).float().to(device)
    targ_tensor = torch.from_numpy(rgb_targ_np/255.0).permute(0,3,1,2).float().to(device)
    feat_pred = perceptual_net(pred_tensor)
    feat_targ = perceptual_net(targ_tensor)
    return nn.functional.l1_loss(feat_pred, feat_targ)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)
    if not verify_dataset_pairs(args.data):
        raise SystemExit("Dataset verification failed")
    dataset = ImagePairDataset(args.data, split="train", img_size=(args.img_w, args.img_h))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = UNet().to(device)
    perceptual = TinyVGG().to(device)
    perceptual.eval()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    l1 = nn.L1Loss()
    best_loss = 1e9
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0
        pbar = tqdm(loader)
        for L, ab, hist in pbar:
            L, ab = L.to(device), ab.to(device)
            ab_pred = model(L)
            loss_l1 = l1(ab_pred, ab)
            loss_perc = compute_perceptual_loss(ab_pred, ab, L, perceptual, device)
            loss = loss_l1 + 0.2*loss_perc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss/(pbar.n+1)})
        avg = running_loss/len(loader)
        scheduler.step(avg)
        print(f"Epoch {epoch+1} avg loss: {avg:.6f}")
        ckpt = {"epoch": epoch+1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": avg}
        torch.save(ckpt, f"{args.save_dir}/ckpt_epoch_{epoch+1}.pth")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), f"{args.save_dir}/colorization_best.pth")
            print("Saved best model")
    torch.save(model.state_dict(), f"{args.save_dir}/colorization_final.pth")
    print("Training complete. Best loss:", best_loss)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--save_dir", default="models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_w", type=int, default=224)
    parser.add_argument("--img_h", type=int, default=224)
    args = parser.parse_args()
    train(args)
