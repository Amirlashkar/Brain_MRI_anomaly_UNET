# Brain MRI Anomaly Detection Using Simplified U-Net

This repository contains the source code for a deep learning project designed to detect anomalies in brain MRI scans using a lightweight, 2D U-Net-inspired model. The model performs binary classification of MRI sequences (T1, T2, T2 FLAIR) through unsupervised learning.

## Features
- Simplified U-Net architecture optimized for speed and efficiency.
- Dual-model training strategy for noise handling and anomaly scoring.
- Custom loss function combining L1, SSIM, and patch-based metrics.
- Preprocessing pipeline tailored for MRI data, including slice filtering, resizing, and cropping.

## Project Structure
```plaintext
├── plots/                # Prediction samples
├── constants.py          # Adjustable factors which affect preprocessing and training progress
├── models.py             # different tested model architectures
├── functions.py          # Needed functions for preprocess and train and inferences
├── utils.py              # Loss function, dataset, etc.
├── trainer.py            # Runs training
└── README.md             # Project overview (this file)
```
## Usage

Decide if you want to only fit or get inferences of already trained model or both and change last lines of `trainer.py` due to your need.

## License

This project is licensed under the [MIT License](LICENSE), which allows for personal, academic, or commercial use with the conditions outlined in the license.

## Medium Article

You can read more about this project in detail on [Medium](https://medium.com/@amireza1422/brain-mri-sequences-t1-t2-t2-flair-anomaly-detection-with-2d-u-net-8ceaf8712124).

Please contact me for access to the model weights.
