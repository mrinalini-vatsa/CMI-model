import os, json, joblib, numpy as np, pandas as pd
import random, math, gc
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.signal import butter, filtfilt
import polars as pl

# ================================
# Configuration - IMPROVED
# ================================
TRAIN = True
RAW_DIR = Path("../input/cmi-detect-behavior-with-sensor-data")
PRETRAINED_DIR = Path("/kaggle/input/cmi-models")
EXPORT_DIR = Path("./")

# Optimized hyperparameters
BATCH_SIZE = 96  # Smaller batch for better generalization
PAD_LEN = 100
LR = 1.5e-3
WEIGHT_DECAY = 5e-4  # Increased from 1e-4
EPOCHS = 100  # Increased with better early stopping
FOLDS = 5  # Back to 5 for better ensemble
PATIENCE = 20  # More patience
RANDOM_STATE = 42

USE_AMP = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | PyTorch: {torch.__version__}")

# Global variables
models = []
data_scaler = None
feature_cols = []
classes = []
imu_cols = []
tof_cols = []

# ================================
# Feature Engineering
# ================================
class FastFeatureExtractor(nn.Module):
    def __init__(self, imu_dim=7):
        super().__init__()
        self.imu_dim = imu_dim
        self.conv_acc = nn.Conv1d(3, 12, kernel_size=5, padding=2, groups=3)
        
    def forward(self, imu):
        B, C, T = imu.shape
        
        acc = imu[:, :3]
        rot = imu[:, 3:] if C > 3 else torch.zeros(B, 1, T, device=imu.device)
        
        acc_mag = torch.norm(acc, dim=1, keepdim=True)
        rot_mag = torch.norm(rot, dim=1, keepdim=True) if rot.size(1) > 0 else torch.zeros(B, 1, T, device=imu.device)
        
        acc_feat = self.conv_acc(acc)
        
        output = torch.cat([acc, rot, acc_mag, rot_mag, acc_feat], dim=1)
        return output

# ================================
# Model Architecture - IMPROVED
# ================================
class EfficientBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, dropout=0.2):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, 
                                   stride=stride, padding=kernel_size//2, 
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)  # Added dropout
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.dropout(x)
        return self.act(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class ImprovedBFRBModel(nn.Module):
    """Improved model with better regularization"""
    def __init__(self, n_classes, imu_dim=7, tof_dim=25):
        super().__init__()
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim
        
        self.feat_extractor = FastFeatureExtractor(imu_dim=imu_dim)
        feat_dim = 3 + (imu_dim - 3) + 1 + 1 + 12
        
        # Smaller network with more regularization
        self.imu_conv1 = EfficientBlock(feat_dim, 96, kernel_size=7, dropout=0.25)
        self.imu_pool1 = nn.MaxPool1d(2)
        self.imu_se1 = SqueezeExcitation(96)
        
        self.imu_conv2 = EfficientBlock(96, 192, kernel_size=5, dropout=0.3)
        self.imu_pool2 = nn.MaxPool1d(2)
        self.imu_se2 = SqueezeExcitation(192)
        
        # TOF pathway
        self.tof_conv1 = nn.Sequential(
            nn.Conv1d(tof_dim, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.MaxPool1d(2)
        )
        
        self.tof_conv2 = nn.Sequential(
            nn.Conv1d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.MaxPool1d(2)
        )
        
        fusion_dim = 192 + 96
        
        # Multi-head attention with dropout
        self.temporal_attn = nn.MultiheadAttention(
            fusion_dim, num_heads=8, dropout=0.2, batch_first=True
        )
        
        # Classification head with heavier dropout
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(192, n_classes)
        )
        
    def forward(self, x):
        imu = x[:, :, :self.imu_dim].transpose(1, 2)
        tof = x[:, :, self.imu_dim:].transpose(1, 2)
        
        # IMU features
        imu_feat = self.feat_extractor(imu)
        x1 = self.imu_conv1(imu_feat)
        x1 = self.imu_pool1(x1)
        x1 = self.imu_se1(x1)
        
        x1 = self.imu_conv2(x1)
        x1 = self.imu_pool2(x1)
        x1 = self.imu_se2(x1)
        
        # TOF features
        x2 = self.tof_conv1(tof)
        x2 = self.tof_conv2(x2)
        
        # Fusion
        x = torch.cat([x1, x2], dim=1)
        
        # Temporal attention
        x_t = x.transpose(1, 2)
        x_t, _ = self.temporal_attn(x_t, x_t, x_t)
        x = x_t.transpose(1, 2)
        
        return self.classifier(x)

# ================================
# Data Processing
# ================================
def butter_lowpass_filter(data, cutoff=5, fs=50, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

class BFRBDataset(Dataset):
    def __init__(self, sequences, labels, mode='train', imu_dim=7):
        self.sequences = sequences
        self.labels = labels
        self.mode = mode
        self.imu_dim = imu_dim
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = self.sequences[idx].copy()
        y = self.labels[idx]
        
        # More aggressive augmentation
        if self.mode == 'train':
            # Gaussian noise
            if random.random() > 0.3:
                noise = np.random.randn(*x[:, :self.imu_dim].shape) * 0.02
                x[:, :self.imu_dim] = x[:, :self.imu_dim] + noise
            
            # Random scaling
            if random.random() > 0.4:
                scale = np.random.uniform(0.9, 1.1)
                x[:, :self.imu_dim] = x[:, :self.imu_dim] * scale
            
            # Time shift
            if random.random() > 0.5:
                shift = np.random.randint(-5, 6)
                x = np.roll(x, shift, axis=0)
            
            # Random sensor dropout (simulate missing TOF data)
            if random.random() > 0.6:
                x[:, self.imu_dim:] *= np.random.uniform(0, 1)
        
        return torch.FloatTensor(x), torch.LongTensor([y])[0]

def preprocess_data(df, feature_cols, scaler=None, fit_scaler=False):
    df_filled = df[feature_cols].select_dtypes(include=[np.number]).ffill().bfill().fillna(0)
    actual_feature_cols = df_filled.columns.tolist()
    
    imu_cols = [c for c in actual_feature_cols if c.startswith(('acc_', 'rot_'))]
    for col in imu_cols:
        if col in df_filled.columns and len(df_filled) > 10:
            try:
                df_filled[col] = butter_lowpass_filter(df_filled[[col]].values)
            except:
                pass
    
    if fit_scaler:
        scaler = StandardScaler()
        data = scaler.fit_transform(df_filled.values)
    else:
        data = scaler.transform(df_filled.values)
    
    return data.astype(np.float32), scaler

def pad_sequence(seq, maxlen, pad_value=0):
    if len(seq) >= maxlen:
        return seq[:maxlen]
    else:
        pad_width = ((0, maxlen - len(seq)), (0, 0))
        return np.pad(seq, pad_width, constant_values=pad_value)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ================================
# Training
# ================================
if TRAIN:
    set_seed(RANDOM_STATE)
    print("Loading training data...")
    
    df = pd.read_csv(RAW_DIR / "train.csv")
    
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['gesture'])
    classes = le.classes_
    np.save(EXPORT_DIR / "classes.npy", classes)
    
    meta_cols = {'gesture', 'label', 'sequence_type', 'behavior', 
                 'orientation', 'row_id', 'subject', 'sequence_id', 
                 'sequence_counter', 'phase'}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = numeric_cols
    
    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols = [c for c in feature_cols if c.startswith(('thm_', 'tof_'))]
    
    print(f"Features: {len(imu_cols)} IMU + {len(tof_cols)} TOF/THM = {len(feature_cols)} total")
    np.save(EXPORT_DIR / "feature_cols.npy", feature_cols)
    
    print("Preprocessing sequences...")
    sequences = []
    labels = []
    seq_ids = []
    
    data_scaler = None
    for seq_id, group in df.groupby('sequence_id'):
        data, data_scaler = preprocess_data(
            group, feature_cols, data_scaler, 
            fit_scaler=(data_scaler is None)
        )
        sequences.append(pad_sequence(data, PAD_LEN))
        labels.append(group['label'].iloc[0])
        seq_ids.append(seq_id)
    
    joblib.dump(data_scaler, EXPORT_DIR / "scaler.pkl")
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    print(f"Data shape: {sequences.shape}")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{FOLDS}")
        print(f"{'='*50}")
        
        train_dataset = BFRBDataset(
            sequences[train_idx], labels[train_idx], 
            mode='train', imu_dim=len(imu_cols)
        )
        val_dataset = BFRBDataset(
            sequences[val_idx], labels[val_idx], mode='val'
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE*2, 
            shuffle=False, num_workers=2, pin_memory=True
        )
        
        model = ImprovedBFRBModel(
            n_classes=len(le.classes_),
            imu_dim=len(imu_cols),
            tof_dim=len(tof_cols)
        ).to(device)
        
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR, epochs=EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.15, anneal_strategy='cos'
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        scaler = GradScaler() if USE_AMP else None
        
        best_acc = 0
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                
                if USE_AMP:
                    with autocast():
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()
            
            train_acc = 100. * train_correct / train_total
            
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | "
                      f"Train: {train_acc:.2f}% | "
                      f"Val: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save({
                    'model_state': model.state_dict(),
                    'imu_dim': len(imu_cols),
                    'tof_dim': len(tof_cols),
                    'n_classes': len(le.classes_)
                }, EXPORT_DIR / f"model_fold{fold}.pth")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Fold {fold+1} Best Val Acc: {best_acc:.2f}%")
        
        checkpoint = torch.load(EXPORT_DIR / f"model_fold{fold}.pth", map_location=device)
        best_model = ImprovedBFRBModel(
            n_classes=checkpoint['n_classes'],
            imu_dim=checkpoint['imu_dim'],
            tof_dim=checkpoint['tof_dim']
        ).to(device)
        best_model.load_state_dict(checkpoint['model_state'])
        best_model.eval()
        models.append(best_model)
        
        del optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"\n✓ Training complete! {len(models)} models ready.")

else:
    print("Loading pretrained models...")
    feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
    data_scaler = joblib.load(PRETRAINED_DIR / "scaler.pkl")
    classes = np.load(PRETRAINED_DIR / "classes.npy", allow_pickle=True)
    
    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols = [c for c in feature_cols if c.startswith(('thm_', 'tof_'))]
    
    for fold in range(FOLDS):
        checkpoint = torch.load(PRETRAINED_DIR / f"model_fold{fold}.pth", map_location=device)
        model = ImprovedBFRBModel(
            n_classes=checkpoint['n_classes'],
            imu_dim=checkpoint['imu_dim'],
            tof_dim=checkpoint['tof_dim']
        ).to(device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        models.append(model)
    
    print(f"✓ Loaded {len(models)} models")

# ================================
# Inference
# ================================
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    global models, data_scaler, feature_cols, classes
    
    if len(models) == 0:
        raise RuntimeError("No models loaded!")
    
    df = sequence.to_pandas()
    data, _ = preprocess_data(df, feature_cols, data_scaler, fit_scaler=False)
    data = pad_sequence(data, PAD_LEN)
    x = torch.FloatTensor(data).unsqueeze(0).to(device)
    
    with torch.no_grad():
        probs = []
        for model in models:
            output = model(x)
            prob = F.softmax(output, dim=1)
            probs.append(prob)
        
        avg_prob = torch.stack(probs).mean(0)
        pred_idx = avg_prob.argmax(1).item()
    
    return str(classes[pred_idx])

# Kaggle interface
import kaggle_evaluation.cmi_inference_server

print(f"\n{'='*60}")
print(f"INFERENCE READY")
print(f"Models: {len(models)} | Features: {len(feature_cols)} | Classes: {len(classes)}")
print(f"{'='*60}\n")

inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            RAW_DIR / 'test.csv',
            RAW_DIR / 'test_demographics.csv',
        )
    )
