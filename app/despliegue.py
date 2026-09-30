"""
Aplicación de despliegue — Predicción de supervivencia en el Titanic
Proyecto de Minería de Datos — Analítica de Datos (UPB)
Equipo: Nicolás Castrillón, Mateo Roldán, Matías Arango

Cómo ejecutarla (ver también el README.md del proyecto):
    streamlit run app/despliegue.py

Carga el modelo Random Forest optimizado con GridSearchCV (models/modelo_final.pkl),
generado en notebooks/modelos_mineria_datos.ipynb, y permite predecir si un pasajero
hipotético habría sobrevivido o no, a partir de sus características.

NOTA DE DISEÑO: la lógica de carga del modelo, validación de datos y predicción
es exactamente la misma que en la versión anterior de este archivo; lo único
que cambió es la presentación visual. Los componentes de interfaz (hero,
tarjetas, barras de probabilidad, gráfico de importancia) están en
`components.py` para mantener este archivo enfocado en el flujo de datos.
"""

import os

import joblib
import pandas as pd
import streamlit as st

from components import (
    cargar_css,
    render_feature_importance_chart,
    render_hero,
    render_probability_bars,
    render_result_card,
    render_sidebar_badges,
    render_sidebar_card_end,
    render_sidebar_card_start,
    render_section_title,
    render_stat_row,
)

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de supervivencia — Titanic",
    page_icon="🚢",
    layout="wide",
)

cargar_css()

RUTA_MODELO = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_final.pkl")


@st.cache_resource
def cargar_modelo(ruta: str):
    """Carga el pipeline entrenado (preprocesamiento + Random Forest optimizado).
    Se cachea para no releer el archivo .pkl en cada interacción del usuario."""
    return joblib.load(ruta)


# ---------------------------------------------------------------------------
# Carga del modelo, con manejo de errores (misma lógica que antes)
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
# Hero / encabezado
# ---------------------------------------------------------------------------
render_hero(
    titulo="🚢 Predicción de Supervivencia — Titanic",
    subtitulo=(
        "Estima si un pasajero habría sobrevivido al naufragio del Titanic, a partir de "
        "sus características, usando un modelo Random Forest optimizado con GridSearchCV."
    ),
    eyebrow="Proyecto de Minería de Datos · Analítica de Datos · UPB",
    badges=(
        ["🌲 Random Forest", "🎯 GridSearchCV", "📊 891 pasajeros históricos", "✅ Validación cruzada 5-fold"]
        + (["🟢 Modelo cargado"] if error_carga is None else ["🔴 Modelo no disponible"])
    ),
)

if error_carga:
    st.error(error_carga)
    st.stop()
else:
    # Aviso breve (no intrusivo) de que el modelo quedó cargado correctamente;
    # reemplaza el antiguo `st.success("Modelo cargado correctamente.")` fijo
    # por una notificación tipo "toast", más propia de una app moderna.
    st.toast("Modelo cargado correctamente ✅", icon="✅")

# ---------------------------------------------------------------------------
# Cuerpo: formulario (izquierda) + panel informativo (derecha)
# ---------------------------------------------------------------------------
col_form, col_info = st.columns([1.6, 1], gap="large")

with col_form:
    with st.container(border=True):
        render_section_title("📝", "Datos del pasajero", "Completa el formulario y presiona \"Predecir\".")

        with st.form("formulario_prediccion"):
            col1, col2 = st.columns(2)

            with col1:
                pclass_label = st.selectbox(
                    "🎫 Clase del boleto",
                    options=["Primera clase", "Segunda clase", "Tercera clase"],
                    index=2,
                    help="Clase en la que viajaba el pasajero.",
                )
                sexo_label = st.selectbox(
                    "👤 Sexo",
                    options=["Mujer", "Hombre"],
                    help="Sexo del pasajero, tal como fue registrado en los datos históricos.",
                )
                edad = st.number_input(
                    "🎂 Edad (años)",
                    min_value=0.0,
                    max_value=100.0,
                    value=30.0,
                    step=1.0,
                    help="Edad del pasajero, entre 0 y 100 años.",
                )
                tarifa = st.number_input(
                    "💰 Tarifa pagada (libras de la época)",
                    min_value=0.0,
                    max_value=600.0,
                    value=32.0,
                    step=1.0,
                    help="Valor pagado por el boleto. El promedio histórico es de unas 32 libras.",
                )

            with col2:
                hermanos_esposos = st.number_input(
                    "👨‍👩‍👧 Hermanos/as o cónyuge a bordo",
                    min_value=0,
                    max_value=8,
                    value=0,
                    step=1,
                    help="Número de hermanos, hermanas o cónyuge a bordo (SibSp).",
                )
                padres_hijos = st.number_input(
                    "👶 Padres o hijos a bordo",
                    min_value=0,
                    max_value=6,
                    value=0,
                    step=1,
                    help="Número de padres o hijos a bordo (Parch).",
                )
                embarked_label = st.selectbox(
                    "⚓ Puerto de embarque",
                    options=["Southampton", "Cherbourg", "Queenstown"],
                    help="Puerto en el que el pasajero subió al barco.",
                )

            enviado = st.form_submit_button("🔮 Predecir supervivencia", use_container_width=True)

    # -----------------------------------------------------------------------
    # Validación y traducción de las etiquetas a los valores que espera el
    # modelo (misma lógica que antes, sin cambios)
    # -----------------------------------------------------------------------
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
                st.error(f"⚠️ {e}")
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

                st.markdown("<br>", unsafe_allow_html=True)
                render_section_title("🔮", "Resultado de la predicción")

                render_result_card(sobrevive=(prediccion == 1), probabilidad=prob_sobrevive)

                with st.container(border=True):
                    render_probability_bars(prob_sobrevive, prob_no_sobrevive)

                    with st.expander("🔍 Ver datos enviados al modelo"):
                        st.dataframe(entrada, hide_index=True, use_container_width=True)

            except Exception as e:
                st.error(f"❌ No fue posible realizar la predicción: {e}")

with col_info:
    with st.container(border=True):
        render_section_title("ℹ️", "Sobre el dataset")
        render_stat_row([
            ("891", "Pasajeros"),
            ("7", "Variables usadas"),
            ("38.4%", "Tasa de supervivencia"),
        ])
        st.markdown(
            '<p class="section-subtitle" style="margin-top:0.9rem;">'
            "Dataset histórico del Titanic, el mismo usado en las prácticas "
            "anteriores del equipo. Ver <code>data/README.md</code> para el "
            "detalle de cada columna."
            "</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        render_section_title("🧠", "Sobre el modelo")
        render_stat_row([
            ("0.82", "Accuracy (CV)"),
            ("0.87", "ROC-AUC (CV)"),
        ])
        st.markdown(
            '<p class="section-subtitle" style="margin-top:0.9rem;">'
            "Random Forest optimizado con GridSearchCV sobre "
            "<code>n_estimators</code>, <code>max_depth</code>, "
            "<code>min_samples_leaf</code> y <code>max_features</code>, con "
            "validación cruzada estratificada de 5 particiones."
            "</p>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Barra lateral: información del modelo (misma información que antes, con
# una presentación más cuidada)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<h2 style="color:#f8fafc; font-size:1.3rem; margin-bottom:0.3rem;">🚢 Panel del modelo</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#94a3b8; font-size:0.85rem; margin-top:0;">'
        "Resumen técnico del modelo desplegado."
        "</p>",
        unsafe_allow_html=True,
    )

    render_sidebar_card_start("Acerca del modelo", "🌲")
    render_sidebar_badges(["Random Forest", "GridSearchCV", "CV 5-fold"])
    st.markdown(
        """
        <div class="sb-fact">🎯 <b>Algoritmo:</b> Random Forest</div>
        <div class="sb-fact">⚙️ <b>Optimización:</b> GridSearchCV (validación cruzada de 5 particiones)</div>
        <div class="sb-fact">📋 <b>Variables:</b> Clase, Sexo, Edad, SibSp, Parch, Tarifa, Puerto de embarque</div>
        <div class="sb-fact">💾 <b>Dataset:</b> Titanic (891 pasajeros)</div>
        """,
        unsafe_allow_html=True,
    )
    render_sidebar_card_end()

    try:
        prep = modelo.named_steps["prep"]
        clf = modelo.named_steps["clf"]
        nombres = prep.get_feature_names_out()
        importancias = pd.Series(clf.feature_importances_, index=nombres).sort_values(ascending=False)

        render_sidebar_card_start("Importancia de variables", "📊")
        fig = render_feature_importance_chart(importancias)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        render_sidebar_card_end()
    except Exception:
        pass

    st.markdown(
        '<p style="color:#64748b; font-size:0.76rem; margin-top:1rem;">'
        "⚠️ Modelo con fines académicos, entrenado sobre datos históricos del "
        "Titanic. No debe usarse para ningún propósito real."
        "</p>",
        unsafe_allow_html=True,
    )
