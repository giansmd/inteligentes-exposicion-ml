# Explicación paso a paso del código - California Housing  
*(formato .md, tono de estudiante que está aprendiendo)*  

---

## 1. Importaciones básicas  
Primero importamos todas las librerías que vamos a usar:  

* **numpy** y **pandas** para manejo de datos.  
* **matplotlib.pyplot** para gráficos.  
* **streamlit** para crear la app web.  
* De **scikit-learn** traemos el dataset de California, el modelo `LinearRegression`, la función de separación `train_test_split` y la métrica `r2_score`.  

Con estas herramientas podemos leer los datos, entrenar el modelo y mostrar todo de forma interactiva.

---

## 2. Configuración de la página en Streamlit  
```python
st.set_page_config(page_title="Regresión Lineal - California Housing", layout="wide")
```
Le doy un título a la pestaña del navegador y elijo el layout “wide” para que los componentes ocupen todo el ancho posible. Luego:
```python
st.title(...)
st.caption(...)
```
Con esto coloco un título grande en la app y un subtítulo explicando qué va a ver el usuario.

---

## 3. Funciones auxiliares con caché  
Uso los decoradores de Streamlit:

* `@st.cache_data` para cachear la carga del dataset.  
* `@st.cache_resource` para cachear el entrenamiento del modelo.

El beneficio es que, si el usuario cambia algo que **no** afecta a estas partes, Streamlit no vuelve a ejecutar la función, haciendo la app más rápida.

---

## 4. Carga de datos (features y target)  
```python
data = fetch_california_housing(as_frame=True)
X = data.data      # variables de entrada
y = data.target    # variable objetivo (precio medio de casas)
```
California Housing es un dataset clásico para regresión; ya viene limpio como `pandas.DataFrame`, así que no necesitamos más pre-procesado en este ejemplo.

---

## 5. Mostrar un vistazo a los datos  
Con:
```python
st.dataframe(X.head(20))
```
enseño las primeras 20 filas de las variables de entrada y,
```python
st.dataframe(pd.DataFrame({target_name: y}).head(20))
```
hago lo mismo con la variable objetivo. Ver las shapes (`X.shape`, `y.shape`) me asegura que todo está en el formato correcto.

---

## 6. División train / test  
En el sidebar coloco dos widgets:

* Un **slider** para elegir `test_size` (porcentaje de datos que reservo para prueba).  
* Un **number_input** para fijar la semilla `random_state`.

Luego:
```python
train_test_split(X, y, test_size=..., random_state=...)
```
genero `X_train`, `X_test`, `y_train`, `y_test`.  
En la interfaz muestro el tamaño de cada parte con `st.metric`, así sé cuántas filas caen en entrenamiento y cuántas en prueba.

---

## 7. Entrenamiento del modelo  
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
`LinearRegression` ajusta un plano (en realidad un hiperplano) a todos los datos de entrenamiento. Gracias al “cache” se entrena una sola vez a menos que cambie la división de datos.

---

## 8. Evaluación rápida (R²)  
```python
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
st.metric("R² en test", f"{r2:.4f}")
```
R² indica qué proporción de la varianza de `y` explica el modelo. Cuanto más cerca de 1, mejor. Aquí simplemente lo muestro como un número grande en la app.

---

## 9. Tabla de predicciones  
Creo un `DataFrame` con dos columnas: “Real” y “Predicho”. Mostrarlo ayuda a comparar caso por caso si el modelo atina.

---

## 10. Visualización de resultados  
Uso `matplotlib` para un scatter plot donde:

* El eje X son los valores reales (`y_test`).  
* El eje Y son las predicciones (`y_pred`).  

Incluyo dos líneas:  
1. La identidad `y = x` (línea roja discontinua). Si los puntos caen aquí, la predicción es perfecta.  
2. Una recta de mejor ajuste (línea verde). La calculo con `np.polyfit` para ver la tendencia general de las predicciones frente a los valores reales.

Al final llamo `st.pyplot(fig)` para que el gráfico aparezca en la app.

---

## 11. Exportar las predicciones  
Con:
```python
df_result.to_csv("predicciones.csv", index=False)
```
guardo el archivo en disco (útil en local) y luego:
```python
st.download_button(...)
```
pongo un botón que permite bajarse el mismo CSV desde la interfaz web.

---

## 12. Pie de página  
`st.caption(...)` solo añade un texto aclarando qué entregables salen de la app: el modelo entrenado, la gráfica y el CSV con las predicciones.

---

### En resumen
Este script monta una demostración completa de regresión lineal sobre el dataset de California Housing, todo en Streamlit. El usuario puede:

1. Ver los datos originales.  
2. Cambiar la división train/test.  
3. Obtener el R² al instante.  
4. Explorar visualmente cómo de buenas son las predicciones.  
5. Descargar los resultados para analizarlos fuera de la app.
