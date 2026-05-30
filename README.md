# 🏥 Análisis Exploratorio y Modelado Predictivo: Readmisión Hospitalaria en Pacientes Diabéticos

**Universidad Industrial de Santander — Inteligencia Artificial I | 2026-1 | Grupo C2**

## 🎥 Presentación del Proyecto (Video)

[Ver presentación del proyecto en YouTube](https://youtu.be/_DWQ3M9nH7I)

## 📌 Descripción del Problema

El objetivo principal de este proyecto es predecir si un paciente diabético será **readmitido dentro de los próximos 30 días** tras el alta hospitalaria (`readmitted = <30`). Los reingresos tempranos indican posibles complicaciones o fallos en el tratamiento inicial y representan un alto costo clínico y económico.

El análisis se basa en el dataset *Diabetes 130-US Hospitals (1999–2008)*, que contiene **101,766 registros** de hospitalizaciones, integrando variables demográficas, clínicas, diagnósticos y medicamentos.

## 🚀 Estructura del Proyecto

El proyecto está estructurado en tres grandes fases (cortes), documentadas paso a paso en el notebook principal.

### Corte 1: Análisis Exploratorio de Datos (EDA) y Preprocesamiento
- **Limpieza de datos:** Tratamiento de valores faltantes, eliminación de registros inconsistentes (pacientes fallecidos/hospicios) y mapeo de IDs clínicos.
- **Análisis Univariado y Bivariado:** Estudio de la distribución de las variables clínicas frente a la variable objetivo de readmisión.

### Corte 2: Modelado Supervisado Clásico
- **Codificación y Escalado:** Uso de `StandardScaler` y `OneHotEncoder` evitando *data leakage*.
- **Balanceo de Clases:** Implementación de **SMOTE** para abordar el fuerte desbalance hacia la clase minoritaria (`<30`).
- **Modelos Evaluados:** Gaussian Naive Bayes, Decision Tree, Random Forest y Linear SVM.
- **Optimización:** Búsqueda de hiperparámetros con `RandomizedSearchCV` priorizando el `F1-macro` y `Recall <30`.

### Corte 3: Deep Learning y Aprendizaje No Supervisado
- **Deep Learning (DNN):** Implementación de una Red Neuronal Densa multicapa con `TensorFlow/Keras` para comparar su rendimiento frente a los modelos clásicos en datos tabulares.
- **Reducción de Dimensionalidad:** Aplicación de **PCA** para retención de varianza y **t-SNE** para visualización de estructuras locales en baja dimensión.
- **Clustering (No Supervisado):** Aplicación de **K-Means**, **DBSCAN** y **Clustering Aglomerativo** para descubrir perfiles clínicos intrínsecos de los pacientes, analizando la pureza de los clústeres frente a las readmisiones reales.

## 📊 Principales Conclusiones

- **Supervisado:** **Random Forest** demostró ser el modelo clásico más robusto en términos de F1 y AUC, mientras que la **Red Neuronal Densa (DNN)** y el **SVM** resultaron especialmente útiles para maximizar el Recall de la clase crítica (`<30`).
- **No Supervisado:** A través de clustering, logramos identificar subpoblaciones claras (ej. *Pacientes Crónicos Geriátricos* de alto riesgo vs *Pacientes de Bajo Riesgo*), lo cual valida la existencia de perfiles clínicos descubiertos puramente a través de características sin etiquetas previas.

## 🛠️ Cómo Ejecutar el Proyecto

1. Clona el repositorio.
2. Asegúrate de tener Python 3.12 y las dependencias (listadas en el entorno virtual o instalables vía `pip`).
3. Ejecuta el notebook principal ubicado en la carpeta `parcial-3/` (o en la raíz) llamado unsupervised_learning.ipynb; este contiene todos los avances de los 3 cortes. Se recomienda utilizar Jupyter Notebook, JupyterLab o Google Colab.
4. Asegúrate de tener el dataset `diabetic_data.csv` y el archivo `IDS_mapping.csv` disponibles en las rutas indicadas en el código.
