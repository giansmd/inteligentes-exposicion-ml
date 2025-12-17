"""
Regresión Polinómica - California Housing
Aplicación Streamlit para predicción de precios de viviendas
usando regresión polinómica.

Autor: Persona 3 - Equipo de Algoritmos de Regresión
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Regresión Polinómica - California Housing",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏠 Regresión Polinómica - California Housing")
st.markdown("Predicción del precio medio de viviendas usando características polinómicas")
st.markdown("---")

# ============================================
# FUNCIONES CON CACHÉ
# ============================================
@st.cache_data
def load_data():
    """Carga el dataset California Housing."""
    data = fetch_california_housing(as_frame=True)
    return data.data, data.target, data.feature_names, data.target_names[0]

@st.cache_resource
def train_polynomial_model(_X_train, _y_train, degree):
    """Entrena el modelo de regresión polinómica."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(_X_train)
    model = LinearRegression()
    model.fit(X_train_poly, _y_train)
    return model, poly

# ============================================
# CARGA DE DATOS
# ============================================
X, y, feature_names, target_name = load_data()

# ============================================
# BARRA LATERAL - CONFIGURACIÓN
# ============================================
st.sidebar.header("⚙️ Configuración del Modelo")
st.sidebar.markdown("---")

test_size = st.sidebar.slider(
    "Porcentaje de datos para test (%)", 
    min_value=10, 
    max_value=50, 
    value=20, 
    step=5
) / 100

random_state = st.sidebar.number_input(
    "Semilla aleatoria", 
    min_value=0, 
    max_value=1000, 
    value=42, 
    step=1
)

degree = st.sidebar.slider(
    "Grado del polinomio", 
    min_value=1, 
    max_value=4, 
    value=2, 
    step=1,
    help="Grados más altos capturan relaciones más complejas pero pueden causar sobreajuste"
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Nota sobre el grado:**
- Grado 1 = Regresión Lineal
- Grado 2 = Términos cuadráticos
- Grado 3+ = Mayor complejidad
""")

# ============================================
# VISTA PREVIA DE LOS DATOS
# ============================================
st.subheader("📊 Vista Previa de los Datos")

col1, col2 = st.columns(2)

with col1:
    st.write("**Variables de entrada (X)**")
    st.dataframe(X.head(20), use_container_width=True)
    st.write(f"📐 Dimensiones: `{X.shape[0]}` filas × `{X.shape[1]}` columnas")

with col2:
    st.write(f"**Variable objetivo ({target_name})**")
    st.dataframe(pd.DataFrame({target_name: y}).head(20), use_container_width=True)
    st.write(f"📐 Dimensiones: `{y.shape[0]}` valores")

st.markdown("---")

# ============================================
# DIVISIÓN TRAIN / TEST
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

st.subheader("📂 División de Datos")
col1, col2, col3 = st.columns(3)
col1.metric("Total de muestras", f"{len(X):,}")
col2.metric("Muestras de entrenamiento", f"{len(X_train):,}")
col3.metric("Muestras de prueba", f"{len(X_test):,}")

st.markdown("---")

# ============================================
# TRANSFORMACIÓN POLINÓMICA
# ============================================
st.subheader("🔄 Transformación Polinómica")

# Crear transformador y aplicar
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

col1, col2, col3 = st.columns(3)
col1.metric("Grado seleccionado", degree)
col2.metric("Características originales", X_train.shape[1])
col3.metric("Características transformadas", X_train_poly.shape[1])

# Mostrar ejemplo de nombres de características generadas
with st.expander("🔍 Ver nombres de características generadas"):
    feature_names_poly = poly.get_feature_names_out(feature_names)
    st.write(f"Se generaron **{len(feature_names_poly)}** características:")
    # Mostrar solo las primeras 20 para no saturar
    st.write(list(feature_names_poly[:20]))
    if len(feature_names_poly) > 20:
        st.write(f"... y {len(feature_names_poly) - 20} más")

st.markdown("---")

# ============================================
# ENTRENAMIENTO DEL MODELO
# ============================================
st.subheader("🎯 Entrenamiento del Modelo")

with st.spinner("Entrenando modelo de regresión polinómica..."):
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

st.success("✅ Modelo entrenado exitosamente")

# ============================================
# PREDICCIONES
# ============================================
y_pred_train = model.predict(X_train_poly)
y_pred_test = model.predict(X_test_poly)

# ============================================
# MÉTRICAS DE EVALUACIÓN
# ============================================
st.subheader("📈 Métricas de Evaluación")

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Conjunto de Entrenamiento:**")
    subcol1, subcol2, subcol3 = st.columns(3)
    subcol1.metric("R²", f"{r2_train:.4f}")
    subcol2.metric("MSE", f"{mse_train:.4f}")
    subcol3.metric("RMSE", f"{rmse_train:.4f}")

with col2:
    st.markdown("**Conjunto de Prueba:**")
    subcol1, subcol2, subcol3 = st.columns(3)
    subcol1.metric("R²", f"{r2_test:.4f}")
    subcol2.metric("MSE", f"{mse_test:.4f}")
    subcol3.metric("RMSE", f"{rmse_test:.4f}")

# Detección de sobreajuste
st.markdown("---")
diff_r2 = r2_train - r2_test

if diff_r2 > 0.15:
    st.error(f"""
    ⚠️ **Sobreajuste detectado**
    
    La diferencia entre R² de entrenamiento ({r2_train:.4f}) y R² de prueba ({r2_test:.4f}) 
    es de {diff_r2:.4f}, lo cual indica que el modelo está memorizando los datos de 
    entrenamiento en lugar de generalizar. Considera reducir el grado del polinomio.
    """)
elif diff_r2 > 0.1:
    st.warning(f"""
    ⚠️ **Posible sobreajuste**
    
    La diferencia entre R² de entrenamiento y prueba es de {diff_r2:.4f}. 
    Monitorea el rendimiento con diferentes grados de polinomio.
    """)
elif r2_test < 0.5:
    st.warning("""
    ⚠️ **Bajo poder predictivo**
    
    El modelo tiene un R² menor a 0.5 en el conjunto de prueba, 
    lo que indica que explica menos del 50% de la variabilidad de los datos.
    """)
else:
    st.success(f"""
    ✅ **El modelo generaliza correctamente**
    
    R² en entrenamiento: {r2_train:.4f}
    R² en prueba: {r2_test:.4f}
    Diferencia: {diff_r2:.4f}
    """)

st.markdown("---")

# ============================================
# TABLA DE PREDICCIONES
# ============================================
st.subheader("📋 Predicciones vs Valores Reales")

df_result = pd.DataFrame({
    "Real": y_test.values,
    "Predicho": y_pred_test,
    "Error": y_test.values - y_pred_test,
    "Error Absoluto": np.abs(y_test.values - y_pred_test),
    "Error Porcentual (%)": np.abs((y_test.values - y_pred_test) / y_test.values) * 100
})

# Estadísticas de error
col1, col2, col3 = st.columns(3)
col1.metric("Error Medio Absoluto", f"{df_result['Error Absoluto'].mean():.4f}")
col2.metric("Error Máximo", f"{df_result['Error Absoluto'].max():.4f}")
col3.metric("Error Mínimo", f"{df_result['Error Absoluto'].min():.4f}")

st.dataframe(df_result.head(20).style.format({
    "Real": "{:.4f}",
    "Predicho": "{:.4f}",
    "Error": "{:.4f}",
    "Error Absoluto": "{:.4f}",
    "Error Porcentual (%)": "{:.2f}%"
}), use_container_width=True)

st.markdown("---")

# ============================================
# VISUALIZACIONES
# ============================================
st.subheader("📊 Visualizaciones")

tab1, tab2, tab3 = st.tabs(["Real vs Predicho", "Distribución de Errores", "Comparación de Grados"])

# Tab 1: Gráfico Real vs Predicho
with tab1:
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    scatter = ax1.scatter(y_test, y_pred_test, alpha=0.5, c=df_result["Error Absoluto"], 
                         cmap='coolwarm', edgecolors='k', linewidth=0.3)
    
    # Línea de predicción perfecta
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
             label='Predicción perfecta (y=x)')
    
    # Línea de tendencia
    z = np.polyfit(y_test, y_pred_test, 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_test.min(), y_test.max(), 100)
    ax1.plot(x_line, p(x_line), 'g-', lw=2, 
             label=f'Tendencia: y = {z[0]:.3f}x + {z[1]:.3f}')
    
    ax1.set_xlabel("Valor Real", fontsize=12)
    ax1.set_ylabel("Valor Predicho", fontsize=12)
    ax1.set_title(f"Regresión Polinómica (Grado {degree}): Real vs Predicho\nR² = {r2_test:.4f}", 
                  fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Error Absoluto')
    
    plt.tight_layout()
    st.pyplot(fig1)

# Tab 2: Distribución de Errores
with tab2:
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma de errores
    axes[0].hist(df_result["Error"], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Error = 0')
    axes[0].axvline(x=df_result["Error"].mean(), color='g', linestyle='-', linewidth=2, 
                    label=f'Media = {df_result["Error"].mean():.4f}')
    axes[0].set_xlabel("Error de Predicción", fontsize=12)
    axes[0].set_ylabel("Frecuencia", fontsize=12)
    axes[0].set_title("Distribución de Errores", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Boxplot de errores
    axes[1].boxplot(df_result["Error"], vert=True)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_ylabel("Error de Predicción", fontsize=12)
    axes[1].set_title("Boxplot de Errores", fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Estadísticas adicionales
    st.markdown("**Estadísticas de los Errores:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Media", f"{df_result['Error'].mean():.4f}")
    col2.metric("Desviación Estándar", f"{df_result['Error'].std():.4f}")
    col3.metric("Mediana", f"{df_result['Error'].median():.4f}")
    col4.metric("Rango Intercuartílico", f"{df_result['Error'].quantile(0.75) - df_result['Error'].quantile(0.25):.4f}")

# Tab 3: Comparación de Grados
with tab3:
    st.write("Comparación del rendimiento con diferentes grados polinómicos:")
    
    degrees_to_compare = [1, 2, 3, 4]
    results_comparison = []
    
    progress_bar = st.progress(0)
    
    for i, d in enumerate(degrees_to_compare):
        poly_temp = PolynomialFeatures(degree=d, include_bias=False)
        X_train_temp = poly_temp.fit_transform(X_train)
        X_test_temp = poly_temp.transform(X_test)
        
        model_temp = LinearRegression()
        model_temp.fit(X_train_temp, y_train)
        
        y_pred_train_temp = model_temp.predict(X_train_temp)
        y_pred_test_temp = model_temp.predict(X_test_temp)
        
        r2_train_temp = r2_score(y_train, y_pred_train_temp)
        r2_test_temp = r2_score(y_test, y_pred_test_temp)
        mse_test_temp = mean_squared_error(y_test, y_pred_test_temp)
        
        results_comparison.append({
            "Grado": d,
            "R² Train": r2_train_temp,
            "R² Test": r2_test_temp,
            "Diferencia R²": r2_train_temp - r2_test_temp,
            "MSE Test": mse_test_temp,
            "RMSE Test": np.sqrt(mse_test_temp),
            "Características": X_train_temp.shape[1]
        })
        
        progress_bar.progress((i + 1) / len(degrees_to_compare))
    
    df_comparison = pd.DataFrame(results_comparison)
    
    st.dataframe(df_comparison.style.format({
        "R² Train": "{:.4f}",
        "R² Test": "{:.4f}",
        "Diferencia R²": "{:.4f}",
        "MSE Test": "{:.4f}",
        "RMSE Test": "{:.4f}",
        "Características": "{:,}"
    }).highlight_max(subset=["R² Test"], color='lightgreen')
      .highlight_min(subset=["MSE Test", "Diferencia R²"], color='lightgreen'),
    use_container_width=True)
    
    # Gráfico de comparación
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² por grado
    x_pos = np.arange(len(degrees_to_compare))
    width = 0.35
    
    bars1 = axes[0].bar(x_pos - width/2, df_comparison["R² Train"], width, label='R² Train', color='steelblue')
    bars2 = axes[0].bar(x_pos + width/2, df_comparison["R² Test"], width, label='R² Test', color='coral')
    
    axes[0].set_xlabel("Grado del Polinomio", fontsize=12)
    axes[0].set_ylabel("R²", fontsize=12)
    axes[0].set_title("Comparación de R² por Grado Polinómico", fontsize=14)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(degrees_to_compare)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # MSE y número de características
    ax_twin = axes[1].twinx()
    
    line1, = axes[1].plot(degrees_to_compare, df_comparison["MSE Test"], 'b-o', 
                          linewidth=2, markersize=8, label='MSE Test')
    line2, = ax_twin.plot(degrees_to_compare, df_comparison["Características"], 'r-s', 
                          linewidth=2, markersize=8, label='Características')
    
    axes[1].set_xlabel("Grado del Polinomio", fontsize=12)
    axes[1].set_ylabel("MSE Test", fontsize=12, color='blue')
    ax_twin.set_ylabel("Número de Características", fontsize=12, color='red')
    axes[1].set_title("MSE y Complejidad del Modelo", fontsize=14)
    axes[1].tick_params(axis='y', labelcolor='blue')
    ax_twin.tick_params(axis='y', labelcolor='red')
    
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    axes[1].legend(lines, labels, loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig3)
    
    # Recomendación
    best_degree = df_comparison.loc[df_comparison["R² Test"].idxmax(), "Grado"]
    st.info(f"""
    **Recomendación:** Basado en el R² del conjunto de prueba, el grado **{best_degree}** 
    ofrece el mejor balance entre ajuste y generalización para este dataset.
    """)

st.markdown("---")

# ============================================
# EXPORTAR PREDICCIONES
# ============================================
st.subheader("💾 Exportar Resultados")

col1, col2 = st.columns(2)

with col1:
    # CSV de predicciones
    csv_predictions = df_result.to_csv(index=False)
    st.download_button(
        label="📥 Descargar Predicciones (CSV)",
        data=csv_predictions,
        file_name=f"predicciones_polinomicas_grado{degree}.csv",
        mime="text/csv",
        help="Descarga las predicciones del modelo para análisis posterior"
    )

with col2:
    # CSV de comparación de grados
    csv_comparison = df_comparison.to_csv(index=False)
    st.download_button(
        label="📥 Descargar Comparación de Grados (CSV)",
        data=csv_comparison,
        file_name="comparacion_grados_polinomicos.csv",
        mime="text/csv",
        help="Descarga la tabla comparativa de diferentes grados polinómicos"
    )

# Guardar archivo localmente también
df_result.to_csv("predicciones_polinomicas.csv", index=False)
st.success("✅ Archivo 'predicciones_polinomicas.csv' guardado localmente para evaluación posterior")

st.markdown("---")

# ============================================
# INFORMACIÓN ADICIONAL
# ============================================
with st.expander("ℹ️ Acerca de la Regresión Polinómica"):
    st.markdown("""
    ### ¿Qué es la Regresión Polinómica?
    
    La regresión polinómica es una extensión de la regresión lineal que permite modelar 
    relaciones no lineales entre las variables predictoras y la variable objetivo. 
    Esto se logra agregando términos polinómicos (potencias y productos) de las 
    características originales.
    
    ### Fórmula General
    
    Para una variable x y grado n:
    
    $y = β_0 + β_1x + β_2x^2 + β_3x^3 + ... + β_nx^n$
    
    ### Ventajas
    - Mayor flexibilidad para capturar relaciones no lineales
    - Sigue siendo un modelo lineal en los parámetros (fácil de entrenar)
    - Útil cuando la relación entre variables es curvilínea
    
    ### Desventajas
    - Propenso al sobreajuste con grados altos
    - El número de características crece exponencialmente
    - Sensible a outliers
    
    ### Consejos para elegir el grado
    1. Comenzar con grado 2 y aumentar gradualmente
    2. Monitorear la diferencia entre R² de train y test
    3. Usar validación cruzada para una evaluación más robusta
    4. Considerar técnicas de regularización (Ridge, Lasso) si hay sobreajuste
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Desarrollado para el curso de Algoritmos de Regresión</p>
    <p>Universidad Nacional de Trujillo - 2025</p>
</div>
""", unsafe_allow_html=True)
