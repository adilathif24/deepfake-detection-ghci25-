# GHCI 25 Hackathon Submission: Real-Time Deepfake Detection System

Official implementation of Real-Time Deepfake Detection system for GHCI 25 Hackathon.

## Overview

**Real-Time Deepfake Detection System**

Advanced AI-powered solution for detecting synthesized fake images and videos in real-time. This project leverages deep learning techniques to identify deepfakes with high accuracy across multiple image sources and social media platforms.

**Key Capabilities:**
- Real-time detection of manipulated images and videos
- Patch-level analysis for precise identification of generation artifacts
- Efficient model architecture suitable for edge devices
- Evaluation on diverse datasets to ensure robustness
- Focus on practical real-world deepfakes from social media sources

## Features

- Real-time detection capabilities
- Support for image and video inputs
- Lightweight and efficient model architecture
- High accuracy on benchmark datasets
- Evaluation on diverse social media platforms
- Customizable detection thresholds

## Setup

1. **Clone this repository**

```bash
git clone https://github.com/adilathif24/deepfake-detection-ghci25-
cd deepfake-detection-ghci25-
```

2. **Create a virtual environment, activate it and install requirements:**

```bash
virtualenv -p /usr/bin/python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Datasets

### Supported Datasets

- **Synthetic Datasets**: Standard benchmarks for evaluating detection performance
- **Real-World Datasets**: Deepfakes sourced from social media platforms
- **Custom Datasets**: Support for custom image collections

### Dataset Preparation

Organize your dataset in the following structure:

```
Your_Dataset
├── train
│   ├── 0_real
│   └── 1_fake
├── val
│   ├── 0_real
│   └── 1_fake
└── test
    ├── 0_real
    └── 1_fake
```

Where '0_real' contains authentic images and '1_fake' contains deepfake images.

## Evaluation

To evaluate the model on your dataset:

```bash
python test.py --dataroot {PATH_TO_TEST_SET} --model_path {PATH_TO_CHECKPOINT.pth}
```

### Parameters:
- `--dataroot`: Path to the test dataset directory
- `--model_path`: Path to the pre-trained model checkpoint
- `--output`: Output directory for results (optional)

## Training

### Train a detection model:

```bash
python train.py --name deepfake_detector --dataroot {PATH_TO_DATASET} --checkpoints_dir ./checkpoints --batch_size 32 --lr 0.0002
```

### Training Parameters:
- `--name`: Name of the experiment
- `--dataroot`: Path to training dataset
- `--checkpoints_dir`: Directory to save checkpoints
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.0002)
- `--epochs`: Number of training epochs
- `--gpu_ids`: GPU IDs to use for training

## Project Structure

```
.
├── data/                      # Dataset directory
├── checkpoints/              # Model checkpoints
├── weights/                  # Pre-trained weights
├── train.py                  # Training script
├── test.py                   # Evaluation script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── LICENSE                   # License information
```

## Requirements

Key dependencies:
- Python 3.7+
- PyTorch 1.9+
- OpenCV
- NumPy
- scikit-learn

See `requirements.txt` for complete list.

## Acknowledgment

This project draws inspiration from recent advances in deepfake detection research and follows best practices from the community.

## Citation

If you find this project useful for your research, please consider citing our hackathon submission.

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or suggestions about this project, please reach out through the GitHub repository.

---

**GHCI 25 Hackathon Submission** | Real-Time Deepfake Detection
