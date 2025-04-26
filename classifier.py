import torch
import torch.nn as nn
from torch import optim

num_epochs = 100

# 1. Datos 
X = torch.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).float()

# 2. Modelo simple (1 capa oculta)
class Clasificador(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 1)

    
    def forward(self, x):
        # Activación
        return torch.sigmoid(self.linear1(x))

# 3. Inicialización
modelo = Clasificador()
criterio = nn.BCELoss()
optimizador = optim.SGD(modelo.parameters(), lr=0.1)

# 4. Entrenamiento
for epoch in range(num_epochs):
    # Forward pass
    # outputs: pasar los datos al modelo
    outputs = modelo(X)
    # loss: calcular la perdida entre los outputs y las etiquetas
    loss = criterio(outputs.squeeze(), y)
    
    # Backward pass
    optimizador.zero_grad()
    loss.backward()
    optimizador.step()
    
    if epoch % num_epochs//10 == 0:
        preds = (outputs > 0.5).float()
        accuracy = (preds.squeeze() == y).float().mean()
        print(f"Época: {epoch}, Loss: {loss.item():.4f}, Precisión: {accuracy:.2f}")

# 5. Prueba 
with torch.no_grad():
    test_output = modelo(X)
    predicted = (test_output > 0.5).float()
    final_accuracy = (predicted.squeeze() == y).float().mean()
    print(f"Precisión final: {final_accuracy:.2f}")
