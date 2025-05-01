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

# Cargar y 