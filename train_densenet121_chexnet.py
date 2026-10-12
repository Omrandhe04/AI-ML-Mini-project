"""
================================================================================
DenseNet-121 (CheXNet) Training & Grad-CAM Pipeline for Chest X-Ray Analysis
Academic Reference & Training Script for Google Colab / Kaggle GPUs
================================================================================
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

# 14 NIH Thoracic Pathologies
CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# ------------------------------------------------------------------------------
# 1. DATASET DEFINITION & PATIENT LEAKAGE PREVENTION
# ------------------------------------------------------------------------------
class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        img_path = os.path.join(self.image_dir, img_name)
        
        # Open as RGB (3-channel copy for DenseNet)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        labels = torch.tensor(row[CLASSES].values.astype(np.float32))
        return image, labels

def split_by_patient(df, test_size=0.20, val_size=0.10, random_state=42):
    """
    CRITICAL: Splits dataset by 'Patient ID' to ensure no images from the same
    patient exist in both training and test sets (Preventing Data Leakage).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss.split(df, groups=df['Patient ID']))
    
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]
    
    # Split train and val
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['Patient ID']))
    
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]
    
    print(f"Dataset split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    return train_df, val_df, test_df

# ------------------------------------------------------------------------------
# 2. MODEL DEFINITION: DENSENET-121
# ------------------------------------------------------------------------------
def build_chexnet_model(num_classes=14, pretrained=True):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
    
    # Replace the classification head
    num_ftrs = model.classifier.in_features # 1024
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, num_classes)
        # Note: Do not add Sigmoid here when training with BCEWithLogitsLoss
    )
    return model

# ------------------------------------------------------------------------------
# 3. GRAD-CAM EXPLAINABILITY ENGINE
# ------------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register PyTorch hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class_idx):
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor) # Logits shape [1, 14]
        target_score = output[0, target_class_idx]
        
        # Backward pass
        target_score.backward()
        
        # Global Average Pooling of gradients (weights alpha)
        # self.gradients: [1, C, H, W]
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True) # [1, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # [1, 1, H, W]
        
        # ReLU to eliminate negative contributions
        cam = torch.relu(cam)
        
        # Normalize between 0 and 1
        cam = cam - torch.min(cam)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
            
        return cam.squeeze().cpu().numpy()

# ------------------------------------------------------------------------------
# 4. TRAINING LOOP WITH WEIGHTED BCE LOSS
# ------------------------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate_auroc(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.numpy())
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    aurocs = {}
    for i, cls in enumerate(CLASSES):
        try:
            score = roc_auc_score(all_targets[:, i], all_preds[:, i])
            aurocs[cls] = score
        except ValueError:
            aurocs[cls] = np.nan
            
    mean_auroc = np.nanmean(list(aurocs.values()))
    return aurocs, mean_auroc

if __name__ == '__main__':
    print("CheXNet DenseNet-121 Training Template initialized.")
    print("Ready to deploy on Kaggle / Google Colab GPU environment.")
