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
INITIAL_TRAIN_SIZE = 0.01  # 30% para entrenamiento offline
ONLINE_TRAIN_SIZE = 0.8   # 60% para entrenamiento online (el 10% restante será para test)
LEARNING_RATE = 0.01
EPOCHS_OFFLINE = 20       # Épocas para entrenamiento offline
RANDOM_SEED = 42

# Cargar y preparar datos
def load_data(filename):
    df = pd.read_csv(filename)
    X = df.drop('Outcome', axis=1).values  # Asume que la columna objetivo se llama 'target'
    y = df['Outcome'].values
    
    # Primera división: separar 10% para test final
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_SEED
    )
    
    # Segunda división: separar entrenamiento inicial (offline) y online
    X_offline, X_online, y_offline, y_online = train_test_split(
        X_temp, y_temp, test_size=ONLINE_TRAIN_SIZE/(1-0.1), random_state=RANDOM_SEED
    )
    
    # Estandarizar características
    scaler = StandardScaler()
    X_offline = scaler.fit_transform(X_offline)
    X_online = scaler.transform(X_online)
    X_test = scaler.transform(X_test)
    
    retu