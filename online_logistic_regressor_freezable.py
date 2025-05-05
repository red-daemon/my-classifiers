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
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    return losses, accuracies

# Entrenamiento online modificado para aplicar máscara de congelación
def train_online(model, criterion, optimizer, online_loader, test_loader):
    model.train()
    losses = []
    accuracies = []
    test_accuracies = []
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(online_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Aplicar máscara de congelación antes del paso de optimización
        model.apply_freeze_mask()
        
        optimizer.step()
        
        losses.append(loss.item())
        predicted = (outputs >= 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        accuracy = correct / total
        accuracies.append(accuracy)
        
        if i % 100 == 0:
            test_acc = evaluate(model, test_loader)
            test_accuracies.append(test_acc)
            
            ax1.clear()
            ax2.clear()
            
            ax1.plot(losses, label='Pérdida online')
            ax1.set_title('Pérdida durante entrenamiento online')
            ax1.legend()
            
            ax2.plot(accuracies, label='Precisión online')
            ax2.plot(np.linspace(0, len(accuracies)-1, len(test_accuracies)), 
                     test_accuracies, label='Precisión test', color='red')
            ax2.set_title('Precisión durante entrenamiento online')
            ax2.legend()
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
    
    plt.ioff()
    plt.show()
    return losses, accuracies, test_accuracies

# Evaluación (igual que antes)
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

# Función principal modificada
def main(filename):
    (X_offline, y_offline), (X_online, y_online), (X_test, y_test) = load_data(filename)
    
    offline_dataset = Dataset(X_offline, y_offline)
    online_dataset = Dataset(X_online, y_online)
    test_dataset = Dataset(X_test, y_test)
    
    offline_loader = torch.utils.data.DataLoader(offline_dataset, batch_size=32, shuffle=True)
    online_loader = torch.utils.data.DataLoader(online_dataset, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    input_dim = X_offline.shape[1]
    model = LogisticRegression(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # 1. Entrenamiento offline
    print("=== Entrenamiento Offline ===")
    train_offline(model, criterion, optimizer, offline_loader, EPOCHS_OFFLINE)
    
    # 2. Configurar máscara de congelación
    print("\n=== Configurando máscara de congelación ===")
    freeze_mask = torch.zeros(input_dim, dtype=torch.bool)
    num_freeze = int(input_dim * FREEZE_RATIO)
    freeze_mask[:num_freeze] = True  # Congelar primeros features
    model.set_freeze_mask(freeze_mask)
    print(f"Congelando {num_freeze}/{input_dim} pesos de características")
    
    # 3. Entrenamiento online con pesos congelados
    print("\n=== Entrenamiento Online con Pesos Congelados ===")
    train_online(model, criterion, optimizer, online_loader, tes