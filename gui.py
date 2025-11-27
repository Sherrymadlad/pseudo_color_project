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
from src.model import UNet
from src.dataset import ImagePairDataset
from src.utils import rgb_to_lab_tensor, lab_to_rgb_np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from torch.utils.data import DataLoader

class ColorizationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pseudo-Colorization GUI")
        self.setGeometry(100, 50, 1300, 700)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = UNet().to(self.device)
        model_path = os.path.join("models", "ckpt_epoch6.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

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

        self.btn_load_default = QPushButton("Load Default Test Dataset (50 images)")
        self.btn_load_default.clicked.connect(self.load_default_dataset)
        btn_layout.addWidget(self.btn_load_default)

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

    def load_default_dataset(self):
        data_dir = "data"
        dataset = ImagePairDataset(data_dir, split="test", img_size=(224,224))
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.input_imgs, self.gt_imgs = [], []
        for i, (L, ab, _) in enumerate(loader):
            if i >= 50:  # Limit to first 50
                break
            L_np = L[0,0].numpy()*100
            ab_np = ab[0].numpy()
            gt_rgb = lab_to_rgb_np(L_np, ab_np[0]*127, ab_np[1]*127)
            self.input_imgs.append(L_np)
            self.gt_imgs.append(gt_rgb)
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
            # Prepare input tensor
            if isinstance(input_img, Image.Image):
                L_tensor = rgb_to_lab_tensor(input_img).unsqueeze(0).to(self.device)
                L_np = np.array(input_img).astype(np.float32)/255*100
            else:
                L_np = input_img
                L_tensor = torch.from_numpy(L_np/100)[None,None,...].float().to(self.device)

            # Predict
            with torch.no_grad():
                ab_pred = self.model(L_tensor).cpu().numpy()[0]
            pred_rgb = lab_to_rgb_np(L_np, ab_pred[0]*127, ab_pred[1]*127)

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

        # Input
        img_input = (L_np/100*255).astype(np.uint8) if not isinstance(L_np, Image.Image) else np.array(L_np)
        qimg_input = QImage(img_input.tobytes(), img_input.shape[1], img_input.shape[0], QImage.Format_Grayscale8)
        lbl_input = QLabel()
        lbl_input.setPixmap(QPixmap.fromImage(qimg_input).scaled(128,128,Qt.KeepAspectRatio))
        row_layout.addWidget(lbl_input)

        # Ground truth
        if gt_rgb is not None:
            qimg_gt = QImage(gt_rgb.tobytes(), gt_rgb.shape[1], gt_rgb.shape[0], QImage.Format_RGB888)
            lbl_gt = QLabel()
            lbl_gt.setPixmap(QPixmap.fromImage(qimg_gt).scaled(128,128,Qt.KeepAspectRatio))
        else:
            lbl_gt = QLabel("N/A")
            lbl_gt.setAlignment(Qt.AlignCenter)
        row_layout.addWidget(lbl_gt)

        # Predicted
        qimg_pred = QImage(pred_rgb.tobytes(), pred_rgb.shape[1], pred_rgb.shape[0], QImage.Format_RGB888)
        lbl_pred = QLabel()
        lbl_pred.setPixmap(QPixmap.fromImage(qimg_pred).scaled(128,128,Qt.KeepAspectRatio))
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
