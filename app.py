import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Regresión Lineal - California Housing", layout="wide")
st.title("Regresión Lineal - California Housing (Streamlit)")
st.caption(
    "Presentación interactiva: datos, target, shapes, modelo, predicciones y visualizaciones"
)


# ---- Helpers & cache ----
@st.cache_data(show_spinner=False)
def load_data():
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    feature_names = list(X.columns)
    target_name = y.name if hasattr(y, "name") and y.name else "target"
    return X, y, feature_names, target_name


@st.cache_resource(show_spinner=False)
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# ---- Escalado de variables ----
@st.cache_resource(show_spinner=False)
def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ---- 1) Data & 2) Target ----
X, y, feature_names, target_name = load_data()

st.header("1) Data (features)")
st.write(f"Shape de X: {X.shape}")
st.dataframe(X.head(20), width='stretch')

st.header("2) Target")
st.write(f"Shape de y: {y.shape}")
st.dataframe(pd.DataFrame({target_name: y}).head(20), width='stretch')


# ---- 3) Train/Test split shapes ----
test_size = st.sidebar.slider(
    "Proporción de test", min_value=0.1, max_value=0.4, value=0.2, step=0.05
)
random_state = st.sidebar.number_input("Random state", min_value=0, value=0, step=1)


# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)

# Escalar variables
scaler = fit_scaler(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.header("3) Shapes de la división de datos")

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("X_train (scaled)", f"{X_train_scaled.shape}")
col_b.metric("X_test (scaled)", f"{X_test_scaled.shape}")
col_c.metric("y_train", f"{y_train.shape}")
col_d.metric("y_test", f"{y_test.shape}")


# ---- 4) Modelo ----
model = train_linear_regression(X_train_scaled, y_train)
st.header("4) Modelo")
st.write(model)

# Métrica rápida
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
st.metric("R² en test", f"{r2:.4f}")


# ---- 5) Predicciones ----
st.header("5) Predicciones")
df_result = pd.DataFrame({"Real": y_test.to_numpy(), "Predicho": y_pred})
st.dataframe(df_result.head(150), width='stretch')


# ---- Visualizaciones: dispersión + líneas de ajuste ----
st.subheader("Resultados visuales")
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred, alpha=0.5, label="Predicciones")

# Línea identidad y = x
min_val = float(min(y_test.min(), y_pred.min()))
max_val = float(max(y_test.max(), y_pred.max()))
lims = [min_val, max_val]
ax.plot(lims, lims, "r--", label="Línea identidad y = x")

# Línea de mejor ajuste (y_pred ~ y_test)
coef = np.polyfit(y_test, y_pred, 1)
line = np.poly1d(coef)
xs = np.linspace(lims[0], lims[1], 200)
ax.plot(xs, line(xs), color="green", label=f"Ajuste: y={coef[0]:.2f}x+{coef[1]:.2f}")

ax.set_xlabel("Valores reales (y_test)")
ax.set_ylabel("Predicciones (y_pred)")
ax.set_title("Dispersión y líneas de referencia")
ax.legend()
st.pyplot(fig, width='stretch')


# ---- Exportar predicciones ----
st.subheader("Exportar valores predichos y reales")
# Guardar a disco (útil si se corre localmente)
df_result.to_csv("predicciones.csv", index=False)

# Botón de descarga en la app
csv_bytes = df_result.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar predicciones CSV",
    data=csv_bytes,
    file_name="predicciones.csv",
    mime="text/csv",
)

st.caption(
    "Entregable: modelo entrenado, visualización de ajuste y archivo con predicciones (predicciones.csv)"
)
