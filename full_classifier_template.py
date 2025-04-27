import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score

# ========================
# 1. Configuración inicial
# ========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Hiperparámetros
CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 20,
    'num_classes': 10  # Cambiar según tu problema
}

# ========================
# 2. Definición del Dataset
# ========================
class CustomDataset(Dataset):
    def __init__(self, features, labels=None, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.transform:
            x = self.transform(x)
        
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        return x

# =========