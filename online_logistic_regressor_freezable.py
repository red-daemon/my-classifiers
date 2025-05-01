import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Configuración de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parámetros configurables
INITIAL_TRAIN_SIZE = 0.3
ONLINE_TRAIN_SIZE = 0.6
LEARNING_RATE = 0.01
EPOCHS_OFFLINE = 20
RANDOM_SEED = 69
FREEZE_RATIO = 0.3  # Porcentaje de pesos a congelar

# Modelo de regresión logística con capacidad de congelar pesos
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.freeze_mask = None
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    def set_freeze_mask(self, freeze_mask):
        """Establece máscara para congelar pesos específicos"""
        self.freeze_mask = freeze_mask.to(device)
        
    def apply_freeze_mask(self):
        """Aplica la máscara de congelación a los gradientes"""
        if self.freeze_mask is not None:
            # Congelar pesos seleccionados
            self.linear.weight.grad[:, self.freeze_mask] = 0.0

# Cargar y preparar datos (igual que antes)
def load_data(filename):
    df = pd.read_csv(filename)
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_SEED
    )
    
    X_offline, X_online, y_offline, y_online = train_test_split(
        X_temp, y_temp, test_size=ONLINE_TRAIN_SIZE/(1-0.1), random_state=RANDOM_SEED
    )
    
    scaler = StandardScaler()
    X_offline = scaler.fit_transform(X_offline)
    X_online = scaler.transform(X_online)
    X_test = scaler.transform(X_test)
    
    return (X_offline, y_offline), (X_online, y_online), (X_test, y_test)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].to(device), self.y[idx].to(device)

# Entrenamiento offline (igual que antes)
def train_offline(model, criterion, optimizer, train_loader, epochs):
    model.train()
    losses = []
    accuracies = []
    