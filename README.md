**Classifying Body-Focused Repetitive Behaviors using Multi-Sensor Wrist-Wearable Data**
This repository contains our full pipeline for the Child Mind Institute – Detecting Behavioral Repetitive Movements competition.
We build a deep-learning model that classifies BFRB-like vs non-BFRB-like gestures using IMU, thermopile, and time-of-flight (TOF) data collected from the Helios wrist-worn device.

 1. Problem Overview

Body-Focused Repetitive Behaviors (BFRBs) include actions such as:
hair pulling
skin picking
nail biting

These behaviors can cause physical harm and are common in anxiety-related and OCD-related disorders.

The Helios device records:
IMU (motion + rotation)
Thermopiles (temperature/heat)
Time-of-Flight sensors (proximity)

The task is to classify a sensor sequence into one of 18 possible gestures (8 BFRB-like + 10 non-BFRB-like).

2. Repository Structure
├── train.py              # Full training code (cross-validation, augmentation)
├── model.py              # Model architecture: FastFeatureExtractor + Efficient Blocks
├── inference.py          # Prediction pipeline + Kaggle inference server
├── utils.py              # Preprocessing, filtering, padding utilities
├── README.md
└── saved_models/
      ├── model_fold0.pth
      ├── model_fold1.pth
      ├── ...
      └── scaler.pkl


(Structure will auto-match your real repo once uploaded.)

3. Key Features of Our Approach
✔ Multi-modal Sensor Fusion

We combine:

IMU features (acceleration + rotation)
TOF proximity sensors
Thermopile heat sensors

✔ Advanced Deep Learning Architecture

Our model includes:
FastFeatureExtractor CNN for IMU streams

Efficient Conv1D Blocks with:
depthwise separable convolutions
dropout regularization
BatchNorm
Squeeze-and-Excitation (SE) blocks
Multi-head Temporal Attention
Adaptive pooling + fully connected classifier

✔ Strong Regularization

Label smoothing

Dropout up to 0.4

Early stopping

Smaller batch size for generalization

✔ 5-Fold Stratified Cross-Validation

We train five models that are ensembled at inference time.

✔ Heavy Data Augmentation

To increase robustness:

Gaussian noise

Random magnitude scaling

Time shifting

Random TOF dropout

Butterworth low-pass filtering

🛠 4. Installation
pip install torch polars pandas numpy scipy scikit-learn joblib


On Kaggle, all required packages are preinstalled.

📦 5. Dataset Description

We use the official competition dataset:

train.csv

test.csv

test_demographics.csv

Each row corresponds to a moment in time.
Each sequence corresponds to a whole gesture.

Sensors included:

Sensor Type	Features
IMU	acc_x, acc_y, acc_z, rot_x, rot_y, rot_z
TOF	tof_0 … tof_4
Thermopile	thm_0 … thm_4

The dataset includes metadata: gesture name, subject ID, orientation, and timestamps.

🧹 6. Preprocessing Pipeline

Group by sequence_id

Fill missing values

Low-pass Butterworth filter for IMU

Standardize features with StandardScaler

Pad sequences to length = 100

Convert to tensor format

 7. Model Training

To train using cross-validation:

python train.py


Training features:

Optimizer: AdamW

LR: 1.5e-3 with OneCycleLR

Loss: CrossEntropy + Label Smoothing

EPOCHS = 100

PATIENCE = 20

AMP mixed precision for speed

This produces:

model_fold0.pth
model_fold1.pth
...
model_fold4.pth
scaler.pkl
feature_cols.npy
classes.npy

🔮 8. Inference

To predict one sequence:

from inference import predict
result = predict(sequence_df, demographics_df)


Behind the scenes:

Preprocessing identical to training

Ensemble of 5 models

Average softmax probability

Return gesture label (string)

9. Kaggle Integration

Our repository supports:

CMIInferenceServer for competition submissions

Local gateway for testing

To run locally:

python inference.py

 10. Results


 Leaderboard score: 0.72
