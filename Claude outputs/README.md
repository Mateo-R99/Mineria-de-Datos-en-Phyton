# Predicción de supervivencia en el Titanic — Proyecto de Minería de Datos

**Materia:** Analítica de Datos — Universidad Pontificia Bolivariana (UPB)
**Profesora:** Ana Isabel Oviedo
**Equipo:** Nicolás Castrillón, Mateo Roldán, Matías Arango

## Descripción del problema

Este proyecto usa el dataset histórico del Titanic (891 pasajeros) para predecir si un pasajero sobrevivió o no al naufragio, a partir de sus características: clase del boleto, sexo, edad, familiares a bordo, tarifa pagada y puerto de embarque. Es un problema de **clasificación binaria** (`Survived`: 1 = sobrevivió, 0 = no sobrevivió).

## Objetivo general

Desarrollar un flujo completo de minería de datos en Python — selección de factores, entrenamiento y evaluación de seis modelos de aprendizaje automático con validación cruzada, análisis de overfitting/underfitting, optimización de hiperparámetros con GridSearchCV, y despliegue del modelo final mediante una interfaz gráfica — aplicado a un problema real de clasificación.

## Tecnologías utilizadas

- Python 3
- pandas, numpy — manipulación de datos
- matplotlib, seaborn — visualización
- scikit-learn — preprocesamiento, modelos, validación cruzada, GridSearchCV
- XGBoost
- Streamlit — interfaz gráfica de despliegue
- Jupyter Notebook

## Estructura del proyecto

```
Mineria-de-Datos-en-Phyton/
├── notebooks/
│   ├── modelos_mineria_datos.ipynb   # Preparación, selección de factores, 6 modelos, CV, GridSearch
│   └── despliegue_modelo.ipynb       # Prueba del modelo guardado y explicación del despliegue
├── app/
│   └── despliegue.py                 # Aplicación Streamlit
├── models/
│   └── modelo_final.pkl              # Pipeline final (preprocesamiento + Random Forest optimizado)
├── data/
│   ├── titanic.csv                   # Dataset original (sin modificar)
│   └── README.md                     # Descripción de las columnas del dataset
├── requirements.txt
└── README.md
```

## Instrucciones de instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/Mateo-R99/Mineria-de-Datos-en-Phyton.git
   cd Mineria-de-Datos-en-Phyton
   ```
2. (Opcional pero recomendado) crea un entorno virtual:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Instrucciones para ejecutar los notebooks

Desde la raíz del proyecto:

```bash
jupyter notebook
```

Abre `notebooks/modelos_mineria_datos.ipynb` y ejecútalo de principio a fin (`Kernel > Restart & Run All`). Este notebook regenera `models/modelo_final.pkl`. Luego abre `notebooks/despliegue_modelo.ipynb` para ver la verificación del modelo guardado y la explicación del despliegue.

## Instrucciones para iniciar la aplicación

```bash
streamlit run app/despliegue.py
```

Se abrirá automáticamente en el navegador (`http://localhost:8501`). Completa el formulario con los datos de un pasajero y presiona "Predecir supervivencia" para ver el resultado. Ver `notebooks/despliegue_modelo.ipynb` para instrucciones detalladas y cómo tomar la captura de pantalla de entrega.

## Descripción de los modelos utilizados

Se entrenaron y compararon seis modelos, todos dentro de un `Pipeline` de scikit-learn (imputación + escalado/codificación ajustados solo con el conjunto de entrenamiento, para evitar fuga de información) y evaluados con validación cruzada estratificada de 5 particiones:

- **Árbol de Decisión** (`DecisionTreeClassifier`)
- **KNN** (`KNeighborsClassifier`)
- **Red Neuronal** (`MLPClassifier`)
- **SVM** (`SVC`, kernel RBF)
- **Random Forest** (`RandomForestClassifier`)
- **XGBoost** (`XGBClassifier`)

El mejor modelo (Random Forest) se optimizó con `GridSearchCV` sobre `n_estimators`, `max_depth`, `min_samples_leaf` y `max_features`.

## Resultados y conclusiones

**Selección de factores:** `Sex` (V de Cramér = 0.54) y `Pclass` (V de Cramér = 0.34) son las variables más asociadas a la supervivencia, seguidas de `Fare` (r = 0.26). `Age` tiene correlación lineal débil pero es relevante de forma no lineal (tercera variable más importante según Random Forest). Se seleccionaron las 7 variables `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked`; se descartaron `PassengerId, Name, Ticket` (sin valor predictivo) y `Cabin` (77% de datos faltantes).

**Comparación de los 6 modelos** (accuracy de validación cruzada / brecha train-test):

| Modelo | CV Accuracy | Brecha Train-Test | ROC-AUC |
|---|---|---|---|
| SVM | 0.825 | 0.026 | 0.851 |
| XGBoost | 0.825 | 0.127 (overfitting) | 0.855 |
| Random Forest | 0.822 | 0.060 | **0.868** |
| Red Neuronal | 0.808 | 0.060 | 0.847 |
| Árbol de Decisión | 0.801 | 0.070 | 0.841 |
| KNN | 0.775 | 0.061 | 0.813 |

XGBoost mostró el overfitting más marcado (93.3% train vs. 80.6% test); SVM generalizó mejor con un desempeño equivalente. Random Forest se eligió para optimizar por tener el mejor ROC-AUC sin sobreajuste severo.

**GridSearchCV sobre Random Forest:** mejores hiperparámetros `n_estimators=200, max_depth=6, min_samples_leaf=1, max_features='sqrt'`. En el conjunto de prueba (no usado durante la búsqueda): accuracy de 0.813 a 0.817, precisión de 0.844 a 0.865, ROC-AUC de 0.858 a 0.862, sin aumentar la brecha train-test. Mejora real pero modesta, consistente con un dataset de 891 filas y con hiperparámetros iniciales ya razonables.

**Limitaciones:** ningún modelo superó ~0.87 de ROC-AUC, lo que sugiere un límite en la señal predictiva disponible en estas 7 variables — la supervivencia también dependió de factores no registrados en los datos (ubicación exacta en el barco, decisiones individuales, azar). Los resultados están basados únicamente en los 891 registros disponibles y no se validaron contra datos externos.

## Datos

Ver `data/README.md` para la descripción completa de las columnas del dataset.
