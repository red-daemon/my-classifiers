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

# ========================
# 3. Arquitectura del Modelo
# ========================
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

# ========================
# 4. Funciones de Entrenamiento
# ========================
def train_model(model, train_loader, val_loader, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)  # Ajusta LR basado en val_accuracy

    best_accuracy = 0
    for epoch in range(config['epochs']):
        # Entrenamiento
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer