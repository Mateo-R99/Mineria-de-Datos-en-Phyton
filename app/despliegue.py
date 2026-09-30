"""
Aplicación de despliegue — Predicción de supervivencia en el Titanic
Proyecto de Minería de Datos — Analítica de Datos (UPB)
Equipo: Nicolás Castrillón, Mateo Roldán, Matías Arango

Cómo ejecutarla (ver también el README.md del proyecto):
    streamlit run app/despliegue.py

Carga el modelo Random Forest optimizado con GridSearchCV (models/modelo_final.pkl),
generado en notebooks/modelos_mineria_datos.ipynb, y permite predecir si un pasajero
hipotético habría sobrevivido o no, a partir de sus características.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de supervivencia — Titanic",
    page_icon="🚢",
    layout="centered",
)

RUTA_MODELO = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_final.pkl")


@st.cache_resource
def cargar_modelo(ruta: str):
    """Carga el pipeline entrenado (preprocesamiento + Random Forest optimizado).
    Se cachea para no releer el archivo .pkl en cada interacción del usuario."""
    return joblib.load(ruta)


# ---------------------------------------------------------------------------
# Carga del modelo, con manejo de errores
# ---------------------------------------------------------------------------
modelo = None
error_carga = None
try:
    modelo = cargar_modelo(RUTA_MODELO)
except FileNotFoundError:
    error_carga = (
        f"No se encontró el archivo del modelo en '{RUTA_MODELO}'. "
        "Ejecuta primero el notebook 'notebooks/modelos_mineria_datos.ipynb' "
        "completo: la última celda guarda 'models/modelo_final.pkl'."
    )
except Exception as e:
    error_carga = f"Ocurrió un error al cargar el modelo: {e}"

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("🚢 Predicción de supervivencia — Titanic")
st.markdown(
    "Esta aplicación usa un modelo **Random Forest**, optimizado con "
    "**GridSearchCV**, entrenado con los datos históricos del Titanic, "
    "para estimar si un pasajero con determinadas características habría "
    "sobrevivido al naufragio."
)

if error_carga:
    st.error(error_carga)
    st.stop()

st.success("Modelo cargado correctamente.")

# ---------------------------------------------------------------------------
# Formulario de entrada
# ---------------------------------------------------------------------------
st.header("Datos del pasajero")

with st.form("formulario_prediccion"):
    col1, col2 = st.columns(2)

    with col1:
        pclass_label = st.selectbox(
            "Clase del boleto",
            options=["Primera clase", "Segunda clase", "Tercera clase"],
            index=2,
            help="Clase en la que viajaba el pasajero.",
        )
        sexo_label = st.selectbox(
            "Sexo",
            options=["Mujer", "Hombre"],
            help="Sexo del pasajero, tal como fue registrado en los datos históricos.",
        )
        edad = st.number_input(
            "Edad (años)",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
            help="Edad del pasajero, entre 0 y 100 años.",
        )
        tarifa = st.number_input(
            "Tarifa pagada (Fare, en libras de la época)",
            min_value=0.0,
            max_value=600.0,
            value=32.0,
            step=1.0,
            help="Valor pagado por el boleto. El promedio histórico es de unas 32 libras.",
        )

    with col2:
        hermanos_esposos = st.number_input(
            "Hermanos/as o cónyuge a bordo (SibSp)",
            min_value=0,
            max_value=8,
            value=0,
            step=1,
        )
        padres_hijos = st.number_input(
            "Padres o hijos a bordo (Parch)",
            min_value=0,
            max_value=6,
            value=0,
            step=1,
        )
        embarked_label = st.selectbox(
            "Puerto de embarque",
            options=["Southampton", "Cherbourg", "Queenstown"],
            help="Puerto en el que el pasajero subió al barco.",
        )

    enviado = st.form_submit_button("Predecir supervivencia")

# ---------------------------------------------------------------------------
# Validación y traducción de las etiquetas a los valores que espera el modelo
# ---------------------------------------------------------------------------
MAPA_PCLASS = {"Primera clase": 1, "Segunda clase": 2, "Tercera clase": 3}
MAPA_SEXO = {"Mujer": "female", "Hombre": "male"}
MAPA_EMBARKED = {"Southampton": "S", "Cherbourg": "C", "Queenstown": "Q"}

if enviado:
    errores = []
    if not (0 <= edad <= 100):
        errores.append("La edad debe estar entre 0 y 100 años.")
    if tarifa < 0:
        errores.append("La tarifa no puede ser negativa.")
    if hermanos_esposos < 0 or padres_hijos < 0:
        errores.append("El número de familiares a bordo no puede ser negativo.")

    if errores:
        for e in errores:
            st.error(e)
    else:
        try:
            entrada = pd.DataFrame([{
                "Pclass": MAPA_PCLASS[pclass_label],
                "Sex": MAPA_SEXO[sexo_label],
                "Age": float(edad),
                "SibSp": int(hermanos_esposos),
                "Parch": int(padres_hijos),
                "Fare": float(tarifa),
                "Embarked": MAPA_EMBARKED[embarked_label],
            }])

            prediccion = modelo.predict(entrada)[0]
            probabilidad = modelo.predict_proba(entrada)[0]
            prob_sobrevive = probabilidad[1]
            prob_no_sobrevive = probabilidad[0]

            st.header("Resultado")

            if prediccion == 1:
                st.success(f"**Predicción: SOBREVIVE** (probabilidad: {prob_sobrevive:.1%})")
            else:
                st.error(f"**Predicción: NO SOBREVIVE** (probabilidad: {prob_no_sobrevive:.1%})")

            col_a, col_b = st.columns(2)
            col_a.metric("Probabilidad de sobrevivir", f"{prob_sobrevive:.1%}")
            col_b.metric("Probabilidad de no sobrevivir", f"{prob_no_sobrevive:.1%}")

            st.caption(
                "Los datos ingresados por el usuario se muestran a continuación, "
                "exactamente como se enviaron al modelo:"
            )
            st.dataframe(entrada, hide_index=True)

        except Exception as e:
            st.error(f"No fue posible realizar la predicción: {e}")

# ---------------------------------------------------------------------------
# Información adicional sobre el modelo (barra lateral)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Acerca del modelo")
    st.markdown(
        "- **Algoritmo:** Random Forest\n"
        "- **Optimización:** GridSearchCV (validación cruzada de 5 particiones)\n"
        "- **Variables usadas:** Clase, Sexo, Edad, Hermanos/Cónyuge a bordo, "
        "Padres/Hijos a bordo, Tarifa y Puerto de embarque\n"
        "- **Dataset:** Titanic (891 pasajeros), el mismo usado en las prácticas "
        "anteriores del equipo"
    )

    try:
        prep = modelo.named_steps["prep"]
        clf = modelo.named_steps["clf"]
        nombres = prep.get_feature_names_out()
        importancias = pd.Series(clf.feature_importances_, index=nombres).sort_values(ascending=False)
        st.subheader("Importancia de variables")
        st.bar_chart(importancias)
    except Exception:
        pass

    st.caption(
        "Este es un modelo con fines académicos, entrenado sobre datos "
        "históricos del Titanic. No debe usarse para ningún propósito real."
    )
