import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.title("Regresión con Dataset Real (Diabetes)")

# Cargar dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name="progresión")

st.subheader("Vista del dataset")
st.dataframe(X.head())

# Selección de variable
feature = st.selectbox("Selecciona una variable para la regresión:", X.columns)

X_feature = X[[feature]]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X_feature, y, test_size=0.2, random_state=42
)

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Evaluación
y_pred = modelo.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.subheader("Evaluación del modelo")
st.write(f"R²: {r2:.3f}")
st.write(f"Error cuadrático medio: {mse:.2f}")

# Ingreso de nuevo dato
st.subheader("Predicción con nuevo dato")
nuevo_valor = st.number_input(
    f"Ingresa un valor para {feature}:", value=float(X_feature.mean())
)

prediccion = modelo.predict([[nuevo_valor]])
st.success(f"Predicción de progresión: {prediccion[0]:.2f}")

# Gráfica
st.subheader("Visualización")
x_line = np.linspace(X_feature.min(), X_feature.max(), 100)
y_line = modelo.predict(x_line)

fig, ax = plt.subplots()
ax.scatter(X_feature, y, alpha=0.5, label="Datos reales")
ax.plot(x_line, y_line, color="red", label="Regresión")
ax.scatter(nuevo_valor, prediccion, color="green", label="Nuevo dato")
ax.set_xlabel(feature)
ax.set_ylabel("Progresión")
ax.legend()

st.pyplot(fig)
