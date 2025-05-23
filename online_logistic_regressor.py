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
    
    return (X_offline, y_offline), (X_online, y_online), (X_test, y_test)

# Convertir a tensores de PyTorch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # Añadir dimensión para BCELoss
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].to(device), self.y[idx].to(device)

# Modelo de regresión logística
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Entrenamiento offline
def train_offline(model, criterion, optimizer, train_loader, epochs):
    model.train()
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Corregido: ya no necesita squeeze()
            
            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Estadísticas
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

# Entrenamiento online con visualización
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
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Estadísticas
        losses.append(loss.item())
        predicted = (outputs >= 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        accuracy = correct / total
        accuracies.append(accuracy)
        
        # Evaluación periódica en test
        if i % 100 == 0:
            test_acc = evaluate(model, test_loader)
            test_accuracies.append(test_acc)
            
            # Actualizar gráficos
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

# Evaluación del modelo
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
    
    accuracy = correct / total
    return accuracy

# Función principal
def main(filename):
    # Cargar datos
    (X_offline, y_offline), (X_online, y_online), (X_test, y_test) = load_data(filename)
    
    # Crear DataLoaders
    offline_dataset = Dataset(X_offline, y_offline)
    online_dataset = Dataset(X_online, y_online)
    test_dataset = Dataset(X_test, y_test)
    
    offline_loader = torch.utils.data.DataLoader(offline_dataset, batch_size=32, shuffle=True)
    online_loader = torch.utils.data.DataLoader(online_dataset, batch_size=1, shuffle=True)  # Batch size 1 para online
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    # Inicializar modelo
    input_dim = X_offline.shape[1]
    model = LogisticRegression(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # 1. Entrenamiento offline
    print("=== Entrenamiento Offline ===")
    offline_losses, offline_accuracies = train_offline(
        model, criterion, optimizer, offline_loader, EPOCHS_OFFLINE
    )
    
    # 2. Entrenamiento online
    print("\n=== Entrenamiento Online ===")
    online_losses, online_accuracies, test_accuracies = train_online(
        model, criterion, optimizer, online_loader, test_loader
    )
    
    # 3. Evaluación final
    final_accuracy = evaluate(model, test_loader)
    print(f"\nPrecisión final en test: {final_accuracy:.4f}")

if __name__ == '__main__':
    datafile = "diabetes.csv"
    main(datafile)