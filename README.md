# Classifying Body-Focused Repetitive Behaviors (BFRBs) Using Multi-Sensor Wrist-Wearable Data

This repository contains the full pipeline developed for the *Child Mind Institute – Detecting Behavioral Repetitive Movements* competition.

We build a deep-learning system to classify **BFRB-like vs non-BFRB-like gestures** using multi-modal sensor data (IMU, Time-of-Flight, Thermopiles) collected from the Helios wrist-worn device.

---

## 1. Problem Overview

Body-Focused Repetitive Behaviors (BFRBs) include:

- Hair pulling  
- Skin picking  
- Nail biting  

These behaviors often occur in anxiety-related and OCD-related disorders and may cause physical harm.

### Sensors in the Helios Device

- IMU (motion + rotation)  
- Thermopiles (temperature/heat)  
- Time-of-Flight (TOF) sensors (proximity)

**Goal:** Classify each sensor sequence into **one of 18 gestures**  
(8 BFRB-like + 10 non-BFRB-like).

---

## 2. Repository Structure
                   ┌─────────────────────────┐
                   │     Raw Sensor Data     │
                   │ (IMU, TOF, Thermopile)  │
                   └─────────────┬───────────┘
                                 │
                                 ▼
                   ┌─────────────────────────┐
                   │    Preprocessing        │
                   │  - Filtering            │
                   │  - Scaling              │
                   │  - Padding (len=100)    │
                   └─────────────┬───────────┘
                                 │
                                 ▼
             ┌─────────────────────────────────────────┐
             │             Neural Network              │
             │ ┌──────────────┬─────────────────────┐  │
             │ │ FastFeature  │ Efficient Conv1D    │  │
             │ │ Extractor    │ Blocks + SE + Attn  |  │
             │ └──────────────┴─────────────────────┘  │
             │                Classifier               │
             └───────────────────┬─────────────────────┘
                                 │
                                 ▼
                   ┌─────────────────────────┐
                   │      Predictions        │
                   │  18-class gesture label │
                   └─────────────────────────┘



## 3. Key Features of Our Approach

### Multi-Modal Sensor Fusion
We combine features from:
- IMU accelerometer + gyroscope  
- TOF proximity sensors  
- Thermopile heat sensors  

### Advanced Deep Learning Architecture
Model components include:

- **FastFeatureExtractor CNN** for IMU  
- Efficient Conv1D blocks with:
  - Depthwise separable convolutions  
  - Pointwise convolutions  
  - BatchNorm  
  - Dropout  
  - Squeeze-and-Excitation (SE)  
- Multi-head temporal attention  
- Adaptive pooling  
- Fully connected classifier  

### Regularization Techniques
- Label smoothing  
- Dropout (up to 0.4)  
- Early stopping  
- Small batch size for stronger generalization  

### 5-Fold Stratified Cross-Validation
Five independently trained models are ensembled for inference.

### Heavy Data Augmentation
- Gaussian noise  
- Random magnitude scaling  
- Time shifting  
- TOF dropout  
- Low-pass Butterworth filtering  

---

## 4. Installation

```bash
pip install torch polars pandas numpy scipy scikit-learn joblib

```
## 5.  Dataset Description

The dataset provided in the competition contains synchronized multi-sensor streams collected from the Helios wrist-worn device.
It includes both raw sensor measurements and metadata required for sequence-level classification.
Sensors included:
```
dataset/
│
├── train.csv
│   ├── sequence_id
│   ├── timestamp
│   ├── acc_x, acc_y, acc_z
│   ├── rot_x, rot_y, rot_z
│   ├── tof_0 … tof_4
│   ├── thm_0 … thm_4
│   └── gesture (label)
│
├── test.csv
│   ├── same sensor features (no label)
│
└── test_demographics.csv
    ├── sequence_id
    ├── age
    ├── gender
    └── handedness
```
## 6. Preprocessing Pipeline

- Group by sequence_id

- Handle missing values

- Apply Butterworth low-pass filter

- Standardize features (StandardScaler)

- Pad sequences to length 100

- Convert to tensors
## 7. Model Training

To train using cross-validation:

```
python train.py
```

Training features:

- Optimizer: AdamW

- LR: 1.5e-3 with OneCycleLR

- Loss: CrossEntropy + Label Smoothing

- EPOCHS = 100

- PATIENCE = 20

- AMP mixed precision for speed

This produces:
```
model_fold0.pth
model_fold1.pth
...
model_fold4.pth
scaler.pkl
feature_cols.npy
classes.npy
```
## 8. Inference

To predict one sequence:
```
from inference import predict
result = predict(sequence_df, demographics_df)

```
Behind the scenes:

- Preprocessing identical to training

- Ensemble of 5 models

- Average softmax probability

- Return gesture label (string)

## 9. Kaggle Integration

Our repository supports:

- CMIInferenceServer for competition submissions

- Local gateway for testing

To run locally:
```
python inference.py
```
## 10. Results


 Leaderboard score: 0.72
