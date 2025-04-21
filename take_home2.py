from sklearn.linear_model import SGDClassifier
import numpy as np

# Inicialización del modelo (log_loss = regresión logística)
# learning_rate='constant' mantiene la tasa de aprendizaje fija
modelo = SGDClassifier(loss='log_loss', 
                       learning_rate='constant', 
                       eta0=0.1,  # Tasa de aprendizaje
                       random_state=69,
                       warm_start=True)

# Simulamos el flujo de datos en tiempo real
datos = [(30, 0), (35, 0), (45, 0), (55, 1), (60, 1)]

print("=== Proceso de Aprendizaje Online ===")
for i, (x, y) in enumerate(datos, 1):
    # Preparamos el dato (scikit-learn espera matrices 2D)
    x_reshape = np.array([[x]])
    
    # 1. Hacemos predicción ANTES de actualizar
    if i > 1:  # Solo después del primer dato
        prob = modelo.predict_proba(x_reshape)[0]
        print(f"\nDato {i}: Edad={x} | Predicción: No-diabetes={prob[0]:.2f}, Diabetes={prob[1]:.2f}")
    else:
        print("\nDato 1: Primer dato recibido - sin predicción")
    
    # 2. Actualización online (partial_fit)
    if i == 1:
        modelo.partial_fit(x_reshape, [y], classes=[0, 1])
    else:
        modelo.partial_fit(x_reshape, [y])
    
    # 3. Mostramos los pesos actuales
    print(f"   Modelo actual: β₀ (intercepto) = {modelo.intercept_[0]:.2f}, "
          f"β₁ (coeficiente) = {modelo.coef_[0][0]:.4f}")
    
nuevo_dato = np.array([[55]])
print(f"Probabilidad para edad 50: {modelo.predict_proba(nuevo_dato)[0]}")

# Coeficientes del modelo
print(f"Intercepto (β₀): {modelo.intercept_}")
print(f"Coeficiente (β₁): {modelo.coef_}")

# Punto de decisión (donde probabilidad = 50%)
punto_decision = -modelo.intercept_ / modelo.coef_
print(f"Eda