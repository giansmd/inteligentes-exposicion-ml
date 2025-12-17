import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página de Streamlit
st.set_page_config(page_title="Regresión Logística - Breast Cancer", layout="wide")


# Función para graficar la matriz de confusión
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicción")
    plt.ylabel("Realidad")
    plt.title("Matriz de Confusión")
    st.pyplot(plt)


def main():
    st.title("Regresión Logística: Detección de Cáncer de Mama")
    st.write(
        """
        Esta aplicación ilustra cómo entrenar y evaluar un modelo de Regresión Logística
        para predecir si un tumor mamario es benigno o maligno, utilizando el dataset
        'Breast Cancer' de scikit-learn.
        """
    )

    # Sección: Carga de datos
    st.header("1) Carga de datos")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    st.write("Vista rápida del dataset (primeras 5 filas):")
    st.dataframe(X.head())
    st.write("Número de muestras:", X.shape[0])
    st.write("Número de características:", X.shape[1])

    # Sección: Configuración de partición y entrenamiento
    st.header("2) División y Entrenamiento del Modelo")

    test_size = st.slider(
        "Tamaño del conjunto de prueba (test_size)", 0.1, 0.4, 0.2, 0.05
    )
    random_state = st.number_input("Semilla aleatoria (random_state)", value=42, step=1)
    max_iter = st.number_input("Iteraciones máximas (max_iter)", value=1000, step=100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state), stratify=y
    )

    # Entrenamiento de la Regresión Logística
    model = LogisticRegression(max_iter=int(max_iter))
    model.fit(X_train, y_train)

    st.write("Entrenamiento completado. Modelo listo para predecir.")

    # Sección: Predicción y Evaluación
    st.header("3) Predicción y Evaluación")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Benigno", "Maligno"])

    st.subheader("Exactitud (Accuracy)")
    st.metric(label="Accuracy", value=f"{accuracy:.4f}")

    st.subheader("Reporte de Clasificación")
    st.text(report)

    # Mostrar algunas filas de predicciones
    num_mostrar = st.slider("Número de casos para mostrar:", 5, 25, 10)
    resultados_df = pd.DataFrame(
        {
            "Real (y_test)": y_test.reset_index(drop=True)[:num_mostrar],
            "Predicho (y_pred)": pd.Series(y_pred[:num_mostrar]),
            "Probabilidad (y_prob)": pd.Series(y_prob[:num_mostrar]),
        }
    )
    st.dataframe(resultados_df)

    # Gráfica de la matriz de confusión
    st.subheader("Matriz de Confusión")
    plot_confusion_matrix(y_test, y_pred)

    # Sección: Exportar resultados para evaluación posterior
    st.header("4) Exportar Resultados (y_pred, y_prob)")
    csv_data = resultados_df.to_csv(index=False)
    st.download_button(
        label="Descargar CSV",
        data=csv_data,
        file_name="resultados_logistica.csv",
        mime="text/csv",
    )
    st.write(
        """
        Con lo anterior, disponemos de:
        - y_pred: etiquetas predichas
        - y_prob: probabilidades asignadas a la clase 'maligno'
        """
    )


if __name__ == "__main__":
    main()
