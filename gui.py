# gui.py
import sys, os
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QScrollArea, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from PIL import Image
import numpy as np
from skimage import color
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import torch.nn.functional as F
from src.utils import *
from src.convolution import convolute

l_cent, l_norm, ab_norm = 50, 100, 110

# Convolution Helpers
def normalize_l(in_l): return (in_l - l_cent) / l_norm
def unnormalize_ab(in_ab): return in_ab * ab_norm

def preprocess_convolution(img_rgb, HW=(256,256)):
    from PIL import Image
    img_rs = np.array(Image.fromarray(img_rgb).resize((HW[1], HW[0])))
    lab_orig = color.rgb2lab(img_rgb)
    lab_rs = color.rgb2lab(img_rs)
    tens_l_orig = torch.Tensor(lab_orig[:,:,0])[None,None,...]
    tens_l_rs = torch.Tensor(lab_rs[:,:,0])[None,None,...]
    return tens_l_orig, tens_l_rs

def postprocess_convolution(tens_orig_l, out_ab):
    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]
    if HW != HW_orig:
        out_ab = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
    out_lab = torch.cat((tens_orig_l, out_ab), dim=1)
    return color.lab2rgb(out_lab[0].cpu().numpy().transpose(1,2,0))

# GUI
class ColorizationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pseudo-Colorization GUI")
        self.showMaximized()

        self.colorizer_convolution = convolute().eval()

        self.input_imgs = []
        self.gt_imgs = []

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.setLayout(layout)

        # Dark mode
        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: #f0f0f0; }
            QPushButton { background-color: #444; color: #f0f0f0; padding: 6px; border-radius: 4px; }
            QPushButton:hover { background-color: #666; }
            QLabel { border: 1px solid #555; background-color: #1e1e1e; }
        """)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Images")
        self.btn_load.clicked.connect(self.load_images)
        btn_layout.addWidget(self.btn_load)
        layout.addLayout(btn_layout)

        # Scrollable area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setSpacing(10)
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll_area)

    def add_headers(self):
        header_frame = QFrame()
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)
        for text in ["Grayscale Input", "Ground Truth", "Colorized Output", "Metrics"]:
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFont(QFont("Arial", 11, QFont.Bold))
            lbl.setStyleSheet("color: #ffdd57;")
            lbl.setFixedHeight(30)
            header_layout.addWidget(lbl)
        header_frame.setLayout(header_layout)
        self.scroll_layout.addWidget(header_frame)

    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg)")
        if not paths:
            return
        self.input_imgs = [Image.open(p).convert("L").resize((224,224)) for p in paths]
        self.gt_imgs = []
        for p in paths:
            color_path = p.replace("_black", "_color")
            if os.path.exists(color_path):
                self.gt_imgs.append(Image.open(color_path).convert("RGB").resize((224,224)))
            else:
                self.gt_imgs.append(None)
        self.colorize_and_display()

    def colorize_and_display(self):
        # Clear scroll area
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.add_headers()

        psnr_list, ssim_list = [], []

        for i, input_img in enumerate(self.input_imgs):
            # Convert input_img to RGB numpy if it's grayscale PIL
            if isinstance(input_img, Image.Image):
                img_rgb = np.array(input_img.convert("RGB"))
                L_np = np.array(input_img).astype(np.float32)
            else:
                img_rgb = input_img
                L_np = input_img

            # Convolution preprocessing
            tens_orig_l, tens_rs_l = preprocess_convolution(img_rgb)
            with torch.no_grad():
                out_ab = self.colorizer_convolution(tens_rs_l)
            pred_rgb = postprocess_convolution(tens_orig_l, out_ab.cpu())
            pred_rgb = (pred_rgb*255).clip(0,255).astype(np.uint8)

            # Ground truth
            gt_img = self.gt_imgs[i]
            if isinstance(gt_img, Image.Image):
                gt_rgb = np.array(gt_img)
            else:
                gt_rgb = gt_img

            # Metrics
            if gt_rgb is not None:
                psnr_val = psnr(gt_rgb, pred_rgb)
                ssim_val = ssim(gt_rgb, pred_rgb, channel_axis=2, win_size=7)
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
            else:
                psnr_val, ssim_val = None, None

            self.add_result_row(L_np, gt_rgb, pred_rgb, psnr_val, ssim_val)

        # Overall metrics
        if psnr_list:
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            overall_lbl = QLabel(f"Average PSNR: {avg_psnr:.2f} | Average SSIM: {avg_ssim:.4f}")
            overall_lbl.setFont(QFont("Arial", 12, QFont.Bold))
            overall_lbl.setStyleSheet("color: #ffdd57; margin-top: 10px;")
            self.scroll_layout.addWidget(overall_lbl)

    def add_result_row(self, L_np, gt_rgb, pred_rgb, psnr_val=None, ssim_val=None):
        row_frame = QFrame()
        row_frame.setFrameShape(QFrame.StyledPanel)
        row_layout = QHBoxLayout()
        row_layout.setSpacing(10)

        preview_size = 256

        # Input
        if isinstance(L_np, Image.Image):
            img_input = np.array(L_np)
        else:
            img_input = (L_np/np.max(L_np)*255).astype(np.uint8)  # normalize to 0-255
        qimg_input = QImage(img_input.tobytes(), img_input.shape[1], img_input.shape[0], QImage.Format_Grayscale8)
        lbl_input = QLabel()
        lbl_input.setPixmap(QPixmap.fromImage(qimg_input).scaled(preview_size, preview_size, Qt.KeepAspectRatio))
        row_layout.addWidget(lbl_input)

        # Ground truth
        if gt_rgb is not None:
            qimg_gt = QImage(gt_rgb.tobytes(), gt_rgb.shape[1], gt_rgb.shape[0], QImage.Format_RGB888)
            lbl_gt = QLabel()
            lbl_gt.setPixmap(QPixmap.fromImage(qimg_gt).scaled(preview_size, preview_size, Qt.KeepAspectRatio))
        else:
            lbl_gt = QLabel("N/A")
            lbl_gt.setAlignment(Qt.AlignCenter)
        row_layout.addWidget(lbl_gt)

        # Predicted
        qimg_pred = QImage(pred_rgb.tobytes(), pred_rgb.shape[1], pred_rgb.shape[0], QImage.Format_RGB888)
        lbl_pred = QLabel()
        lbl_pred.setPixmap(QPixmap.fromImage(qimg_pred).scaled(preview_size, preview_size, Qt.KeepAspectRatio))
        row_layout.addWidget(lbl_pred)

        # Metrics
        metrics_text = f"PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f}" if psnr_val is not None else "PSNR: N/A | SSIM: N/A"
        lbl_metrics = QLabel(metrics_text)
        lbl_metrics.setAlignment(Qt.AlignCenter)
        lbl_metrics.setStyleSheet("color: #ffdd57;")
        row_layout.addWidget(lbl_metrics)

        row_frame.setLayout(row_layout)
        self.scroll_layout.addWidget(row_frame)


if __name__=="__main__":
    app = QApplication(sys.argv)
    win = ColorizationGUI()
    win.show()
    sys.exit(app.exec_())
