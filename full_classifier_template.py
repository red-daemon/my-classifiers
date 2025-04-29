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
            optimizer.step()
            
            train_loss += loss.item()

        # Validación
        val_accuracy, val_loss = evaluate_model(model, val_loader, criterion)
        scheduler.step(val_accuracy)  # Ajusta learning rate
        
        # Guardar mejor modelo
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pt')

        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.2%}")

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, total_loss/len(data_loader)

# ========================
# 5. Pipeline Principal
# ========================
def main():
    # 1. Cargar datos (reemplazar con tus datos)
    # Ejemplo con datos dummy
    X_train = np.random.rand(1000, 20)  # 1000 muestras, 20 características
    y_train = np.random.randint(0, CONFIG['num_classes'], 1000)
    X_val = np.random.rand(200, 20)
    y_val = np.random.randint(0, CONFIG['num_classes'], 200)

    # 2. Crear DataLoaders
    train_dataset = CustomDataset(X_train.astype(np.float32), y_train.astype(np.long))
    val_dataset = CustomDataset(X_val.astype(np.float32), y_val.astype(np.long))
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])

    # 3. Inicializar modelo
    model = Classifier(input_dim=20, num_classes=CONFIG['num_classes']).to(DEVICE)
    print(model)

    # 4. Entrenar
    train_model(model, train_loader, val_loader, CONFIG)

    # 5. Evaluar (opcional)
    # Cargar mejor modelo
    model.load_state_dict(torch.load('best_model.pt'))
    test_accuracy, _ = evaluate_model(model, val_loader, nn.CrossEntropyLoss())
    print(f"\nFinal Test Accuracy: {test_accuracy:.2%}")

if __name__ == '__main__':
    main()