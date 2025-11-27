Pseudo-Colorization Project

Overview:
This project implements a pseudo-colorization pipeline that converts grayscale images into realistic colored images using deep learning. It uses a UNet-based neural network to predict the 'ab' color channels from the grayscale 'L' channel.

A PyQt5 GUI is included for visualization, allowing automatic colorization of images, evaluation with PSNR and SSIM, and viewing multiple images in a scrollable interface.

Features:
- UNet-based colorization model predicting 'ab' channels from grayscale.
- Automatic batch colorization with GUI.
- Scrollable view showing grayscale input, ground truth, and colorized output side by side.
- Metrics display: PSNR and SSIM, per image and averaged.
- Load default test dataset or custom images.
- Automatic colorization upon image load without pressing buttons.
- Preview first 50 test images for simplicity.
- High-quality resizing and display of results.

Requirements:
- torch>=2.0.0
- torchvision>=0.15.0
- numpy>=1.24.0
- Pillow>=10.0.0
- scikit-image>=0.21.0
- tqdm>=4.65.0
- PyQt5>=5.15.9

Installation:
1. Clone the repository:
   git clone https://github.com/Sherrymadlad/pseudo-colorization.git
   cd pseudo-colorization

2. Install dependencies:
   pip install -r requirements.txt

Running the GUI:
- python gui.py
- Click "Load Default Test Dataset" to automatically load and colorize the first 50 test images.
- Scrollable interface shows:
  - Grayscale Input
  - Ground Truth
  - Colorized Output
- Metrics PSNR and SSIM displayed per image and averaged across batch.
- GUI automatically colorizes images when loaded, no extra buttons required.

Training:
- python train.py --data data --save_dir models --epochs 50 --batch_size 8 --lr 1e-4 --img_w 224 --img_h 224
- Uses L1 + perceptual loss.
- Checkpoints saved in models/.
- Training on ~5,000 images with 50 epochs gives good results.

Inspirations & References:
1. Automatic Colorization with Deep Convolutional Networks – Zhang et al., ECCV 2016
2. Perceptual Losses for Real-Time Style Transfer and Super-Resolution – Johnson et al., ECCV 2016
3. UNet Architecture – Ronneberger et al., MICCAI 2015

Project-specific contributions:
- PyQt5 GUI for scrollable batch visualization.
- Automatic colorization on load.
- Dark mode UI with modern design.
- Display of grayscale input, ground truth, and output in separate columns.
- PSNR/SSIM computation per image and averaged.
- Loading only first 50 test images for faster preview.
