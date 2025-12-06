Pseudo-Colorization Project

Project Overview
  This project implements an advanced grayscale-to-color image colorizer using a UNet-based neural network with enhanced convolutional refinement layers. The model predicts the 'ab' channels in LAB color space from the grayscale 'L' channel, producing more visually accurate and detailed results compared to standard colorization approaches.
  A PyQt5 GUI is included for visualization, allowing automatic colorization of images, evaluation with PSNR and SSIM, and viewing multiple images in a scrollable interface.

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
- Click "Load Image"
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


Pseudo-Colorization Project Documentation

1. Key contributions:
    - Novel UNet + convolution refinement architecture for improved color fidelity.
    - Automatic batch colorization via an intuitive GUI.
    - Scrollable display of multiple images with per-image metrics.
    - Real-time colorization triggered upon image load.
    - Quantitative evaluation using PSNR and SSIM metrics.

2. Dataset
  - The dataset structure is as follows:
    data/
    ├── train_black/
    ├── train_color/
    ├── test_black/
    └── test_color/
  - Each grayscale image has a corresponding color image.
  - Images are resized to 224x224 for training and GUI display.
  - GUI preview supports up to 50 test images for quick visualization.

3. Model Architecture
  3.1 UNet Backbone
    - Standard encoder-decoder with skip connections.
    - Input: 1-channel grayscale image (L).
    - Output: 2-channel color prediction (ab).
    - Preserves spatial and structural details via skip connections.
    - Convolutional blocks followed by ReLU activation.

  3.2 Convolutional Refinement Module (Novelty)
    - Adds an extra convolutional refinement block after the UNet output.
    - Improves edge details and color consistency.
    - Helps the network learn subtle texture and shading information that standard UNet may miss.
    - Lightweight and fast to compute, making it compatible with real-time GUI inference.

  3.3 Tiny Perceptual Network
    - Used for perceptual loss computation.
    - Focuses on high-level visual features beyond pixel-wise differences.
    - Improves the realism and visual quality of the predicted colors.

4. Loss Functions
  - L1 Loss: Measures pixel-wise difference between predicted and ground truth 'ab' channels.
  - Perceptual Loss: L1 difference computed on features extracted from the Tiny Perceptual Network.
  - Total Loss:
      loss = L1 + 0.2 * perceptual_loss
      This combination ensures both pixel-level accuracy and perceptual fidelity.

5. GUI Features
  - Dark mode with a modern color scheme.
  - Scrollable display for batch previews.
  - Three-column layout per row:
  - Grayscale Input
  - Ground Truth
  - Colorized Output
  - Automatic colorization when images are loaded.
  - Load default test dataset or custom images.
  - Displays per-image and overall PSNR/SSIM metrics.
  - Efficient preview: only the first 50 images are displayed for performance.

  5.1 UX Enhancements
    - Automatic layout adjustments for multiple images.
    - Metrics displayed clearly beside each row.
    - First image preview immediately upon loading.
    - Hover effects and consistent styling for readability.

6. Evaluation Metrics
  - PSNR (Peak Signal-to-Noise Ratio): Measures pixel-level fidelity between predicted and ground truth images.
  - SSIM (Structural Similarity Index): Measures perceptual similarity and visual quality.
  Metrics are computed for each image and averaged across the batch.

7. Project Inspirations
  - Automatic Colorization with Deep Convolutional Networks – Zhang et al., ECCV 2016
  - Perceptual Losses for Real-Time Style Transfer and Super-Resolution – Johnson et al., ECCV 2016
  - UNet for Image-to-Image Tasks – Ronneberger et al., MICCAI 2015

  Unique Contributions:
    - Novel UNet + convolution refinement module for improved color accuracy.
    - Scrollable batch GUI for efficient visualization.
    - Automatic colorization with combined PSNR/SSIM evaluation.
    - Limiting preview to first 50 images for fast GUI responsiveness.
    - Clear three-column layout for easy comparison between input, ground truth, and output.

8. Future Improvements
  - Video colorization support using temporal consistency.
  - VGG-based perceptual loss for even more realistic colorization.
  - Adaptive histogram and color hints for small datasets.
  - Real-time high-resolution inference optimization.