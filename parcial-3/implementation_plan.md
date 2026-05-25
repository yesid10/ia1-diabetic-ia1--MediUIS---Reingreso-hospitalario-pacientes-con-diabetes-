# Plan de Implementación — Tercer Entregable (Proyecto Final IA1)

## Contexto

El notebook actual (`parcial-3/Copia_de_EDA_Diabetes_Final_copy.ipynb`) contiene:
- **EDA completo** (§1-§10) sobre las 3 clases originales (`NO`, `>30`, `<30`)
- **Modelado supervisado binario** (§11-§24): se eliminó la clase `>30`, quedando solo `NO=0` vs `<30=1`
- 4 modelos entrenados: Gaussian NB, Decision Tree, Random Forest, Linear SVM

**Faltan:** DNN (red densa), aprendizaje no supervisado, reducción de dimensionalidad, y conclusiones finales.

---

## Estrategia de Datos para los Nuevos Algoritmos

> [!IMPORTANT]
> **DNN (supervisado):** Usa el mismo dataset **binario** existente (`readmitted_enc`: 0=NO, 1=<30, sin la clase `>30`). Reutiliza `X_train_balanced`, `X_test`, `y_train_balanced`, `y_test` del pipeline actual. Así la DNN es directamente comparable con los 4 modelos clásicos.

> [!IMPORTANT]
> **No Supervisado + Reducción Dim.:** Usa el dataset **completo con las 3 clases originales** (`NO`, `>30`, `<30`). El clustering no usa etiquetas para entrenar — las 3 clases solo se usan **después** para interpretar qué perfiles de pacientes descubrió cada algoritmo. Esto permite: más datos (~98K en vez de ~63K), patrones más ricos, y conexión directa con el EDA original que sí usaba 3 clases.

---

## Estructura del Notebook — Secciones a Agregar

Todo se agrega al notebook existente, después de la sección 24:

```
EXISTENTE (no se toca):
  §1-§10:   EDA (3 clases originales)
  §11-§24:  Supervisado binario (NB, DT, RF, SVM)

NUEVO — SUPERVISADO DL:
  §25:  Modelo 5: Red Neuronal Densa (DNN) — TensorFlow/Keras
  §26:  Comparativa Final Supervisado: Clásicos vs DNN

NUEVO — NO SUPERVISADO:
  §27:  Preparación de Datos para Análisis No Supervisado
  §28:  Reducción de Dimensionalidad — PCA
  §29:  Reducción de Dimensionalidad — t-SNE
  §30:  Clustering — K-Means
  §31:  Clustering — DBSCAN
  §32:  Clustering — Aglomerativo (Jerárquico)
  §33:  Comparativa de Métodos de Clustering
  §34:  Integración: Perfiles de Pacientes Descubiertos

NUEVO — CIERRE:
  §35:  Conclusiones Finales del Proyecto
```

---

## FASE 1: Red Neuronal Densa (DNN)

### §25 — Modelo 5: Red Neuronal Densa

**Datos:** Reutiliza exactamente los datos del pipeline existente:
- `X_train_balanced` / `y_train_balanced` (post-SMOTE, binario)
- `X_test` / `y_test` (binario)
- Los datos ya están escalados (StandardScaler) y codificados (OneHotEncoder)

**Implementación:**

```python
import tensorflow as tf
from tensorflow import keras

# Convertir target a one-hot para Keras
y_train_ohe = tf.keras.utils.to_categorical(y_train_balanced, num_classes=2)
y_test_ohe  = tf.keras.utils.to_categorical(y_test, num_classes=2)

# Arquitectura (inspirada en notebooks 12-13 del profesor)
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(X_train_balanced.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(2, activation='softmax')   # 2 clases: NO / <30
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Early stopping para evitar overfitting
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

history = model.fit(
    X_train_balanced, y_train_ohe,
    epochs=50, batch_size=256,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)
```

**Visualizaciones:**
- Curvas de aprendizaje: accuracy y loss (train vs validation) por época
- Matriz de confusión (misma paleta que los otros modelos)

**Métricas** (idénticas al pipeline existente para comparar):
- Accuracy, F1-macro, Recall <30, Precision <30, AUC-ROC
- Se almacenan en `acc_dnn`, `f1_dnn`, `recall_30_dnn`, etc.

### §26 — Comparativa Final Supervisado

- Extender `comparison_df` con la fila de DNN (5 modelos total)
- Gráfico de barras comparativo (5 columnas × 4 métricas)
- Discusión: ¿DNN supera a los clásicos? ¿Por qué sí/no en este tipo de datos tabulares?
- Tabla resumen con tiempos de entrenamiento

---

## FASE 2: Aprendizaje No Supervisado y Reducción de Dimensionalidad

### §27 — Preparación de Datos para No Supervisado

**Punto clave:** Aquí se carga el dataset **desde antes del filtrado de la clase `>30`**. Se re-aplica el mismo preprocesamiento pero conservando las 3 clases.

```python
# Recargar datos limpios ANTES del filtro binario
# (re-ejecutar desde df_raw con el mismo preprocesamiento pero SIN eliminar '>30')
df_unsup = df_raw.copy()
# ... aplicar mismos filtros (muertos, hospicios, nulos >50%)
# ... pero NO eliminar '>30'
# ... aplicar misma codificación ordinal/binaria
# ... guardar 'readmitted' original como etiqueta de referencia (NO la usa para entrenar)

# Seleccionar solo features numéricas para clustering
features_unsup = ['time_in_hospital','num_lab_procedures','num_procedures',
                  'num_medications','number_diagnoses',
                  'number_outpatient','number_emergency','number_inpatient',
                  'age_encoded','insulin_encoded']

X_unsup = df_unsup[features_unsup].values
y_labels_original = df_unsup['readmitted'].values  # Solo para interpretar DESPUÉS

# Escalar
from sklearn.preprocessing import StandardScaler
scaler_unsup = StandardScaler()
X_unsup_scaled = scaler_unsup.fit_transform(X_unsup)
```

**Celda Markdown:** Diagrama del workflow no supervisado:
```
Datos limpios (3 clases) → Selección features numéricas → StandardScaler
    → PCA / t-SNE (visualización y reducción)
    → K-Means / DBSCAN / Aglomerativo (clustering)
    → Interpretar clusters vs readmitted original
```

### §28 — PCA (Análisis de Componentes Principales)

Siguiendo el patrón del notebook 15 del profesor:

1. **PCA completo** (todos los componentes):
   - Scree plot: varianza explicada por cada componente
   - Gráfico de varianza acumulada → marcar umbral del 95%
   - Determinar `n_components` óptimo

2. **PCA con 2 componentes** → scatter plot:
   - Coloreado por `readmitted` (3 colores: NO, >30, <30)
   - Interpretar: ¿se separan las clases? ¿cuánta varianza capturan 2 componentes?

3. **Loadings / Pesos de cada componente:**
   - ¿Qué features dominan PC1? ¿Y PC2?
   - Heatmap de loadings

4. **Guardar `X_pca`** para usar como input del clustering

### §29 — t-SNE

Siguiendo el patrón del notebook 15:

1. **Subsampling estratificado** (~5,000-8,000 muestras) para que t-SNE sea viable
2. **PCA(50) como preprocesamiento** antes de t-SNE (como hace el profesor en su notebook)
3. **t-SNE con 2 componentes**:
   - Scatter coloreado por `readmitted` (3 clases)
   - Experimentar con 3 valores de perplexity (5, 30, 50) → panel de 3 gráficos
4. **Interpretación:** ¿Se forman clusters naturales? ¿La estructura local revela patrones que PCA no muestra?

### §30 — K-Means

Siguiendo el patrón del notebook 14:

1. **Método del codo:** K de 2 a 10, gráfico de inercia vs K
2. **Silhouette score** para cada K → determinar K óptimo
3. **K-Means con K óptimo:**
   - Scatter en espacio PCA-2D, coloreado por cluster
4. **Tabla cruzada** clusters × readmitted:
   ```
   Cluster | NO (%) | >30 (%) | <30 (%) | Total
   ```
   → ¿Algún cluster concentra pacientes `<30`?
5. **Perfiles de cluster:** Media/mediana de cada feature por cluster → ¿qué define a cada grupo?

### §31 — DBSCAN

Siguiendo el patrón del notebook 14:

1. **k-distance graph** (k=5) para estimar `eps`
2. **DBSCAN** con eps estimado y min_samples=5
3. **Resultados:** Nº clusters encontrados, nº puntos de ruido (-1)
4. **Scatter** en PCA-2D, coloreado por cluster (ruido en gris)
5. **Tabla cruzada** clusters × readmitted
6. **Comparación visual** con K-Means

### §32 — Clustering Aglomerativo (Jerárquico)

Siguiendo el patrón del notebook 14:

1. **Dendrograma** con linkage Ward (sobre submuestra de ~2,000-3,000 puntos para legibilidad)
2. **Identificar corte** donde hay gaps grandes en el dendrograma
3. **AgglomerativeClustering** con `n_clusters` del dendrograma, sobre datos completos
4. **Scatter** en PCA-2D
5. **Tabla cruzada** clusters × readmitted

### §33 — Comparativa de Métodos de Clustering

| Método | Nº Clusters | Silhouette | % de `<30` concentrado en mejor cluster | Ruido |
|--------|------------|------------|------------------------------------------|-------|
| K-Means | K | score | X% | N/A |
| DBSCAN | auto | score | X% | N puntos |
| Aglomerativo | K | score | X% | N/A |

- Gráfico de barras: Silhouette por método
- Panel visual: 3 scatter plots lado a lado (PCA-2D, uno por método)
- Discusión: ¿Cuál captura mejor la estructura? ¿Por qué?

### §34 — Integración: Perfiles de Pacientes Descubiertos

Con el mejor método de clustering:
1. **Scatter en t-SNE** coloreado por clusters (vs coloreado por readmitted) — panel lado a lado
2. **Tabla de perfiles clínicos** por cluster:
   ```
   Cluster | age_media | time_hospital | num_meds | num_inpatient | % <30 | Interpretación
   ```
3. **Narrativa clínica:** Describir cada cluster como un "perfil de paciente"
   - Ej: "Cluster 2: Pacientes ancianos (>70) con muchas visitas previas y alta medicación → mayor riesgo de readmisión temprana"
4. **Conexión con supervisado:** ¿Los features que el clustering identifica como discriminantes coinciden con los que Random Forest marcó como importantes?

---

## FASE 3: Conclusiones y Organización

### §35 — Conclusiones Finales del Proyecto

Reemplazar la sección 24 actual con conclusiones que integren **todo** el proyecto:

1. **EDA:** Hallazgos clave (desbalance, features discriminantes, distribuciones)
2. **Supervisado clásico:** Mejor modelo (RF esperado), métricas, limitaciones del desbalance
3. **DNN:** ¿Superó a los clásicos? ¿Qué implica para datos tabulares?
4. **No Supervisado:** Perfiles de pacientes descubiertos, relación con readmisión
5. **Reducción de Dimensionalidad:** Utilidad de PCA (varianza, preprocesamiento) y t-SNE (visualización)
6. **Conexión con el curso:** Cómo cada herramienta aporta al entendimiento del problema clínico
7. **Limitaciones:** Features con poca discriminación, desbalance, datos de 1999-2008
8. **Mejoras futuras:** Más features, modelos de series temporales, otros algoritmos

### Organización del Repositorio

1. **README.md** en la raíz con:
   - Título y descripción del proyecto
   - Integrantes del equipo
   - Dataset: fuente, cómo obtenerlo, estructura
   - Estructura del repositorio
   - Enlace al video (placeholder)
2. **Limpieza:**
   - Eliminar duplicados (`Copia_de_EDA_Diabetes_Final_copy.ipynb` en la raíz)
   - El notebook principal queda en la raíz con secciones marcadas en Markdown
3. **Topics** en GitHub: `uis-ia1`, `uis-ia1-c2`

---

## Orden de Ejecución

| Paso | Sección | Tiempo Est. | Dependencia |
|------|---------|-------------|-------------|
| 1 | §25 DNN | 15 min | Datos del pipeline existente |
| 2 | §26 Comparativa supervisado | 5 min | Paso 1 |
| 3 | §27 Prep. datos no supervisado | 10 min | Datos del EDA |
| 4 | §28 PCA | 10 min | Paso 3 |
| 5 | §29 t-SNE | 10 min | Paso 3 |
| 6 | §30 K-Means | 10 min | Paso 4 (usa PCA) |
| 7 | §31 DBSCAN | 10 min | Paso 4 |
| 8 | §32 Aglomerativo | 10 min | Paso 4 |
| 9 | §33 Comparativa clustering | 10 min | Pasos 6-8 |
| 10 | §34 Integración perfiles | 10 min | Pasos 5, 9 |
| 11 | §35 Conclusiones | 15 min | Todo |
| 12 | README + limpieza | 10 min | Todo |
| **Total** | | **~2 horas** | |

---

## Verificación

- [ ] Cada celda ejecuta sin errores de principio a fin
- [ ] DNN usa los mismos datos binarios que los 4 modelos clásicos
- [ ] No supervisado usa las 3 clases originales (solo como referencia)
- [ ] Silhouette scores calculados para K-Means, DBSCAN, Aglomerativo
- [ ] Varianza explicada de PCA mostrada con scree plot
- [ ] t-SNE con al menos 3 valores de perplexity
- [ ] Dendrograma legible para clustering aglomerativo
- [ ] Tablas cruzadas (cluster × readmitted) en los 3 métodos
- [ ] Conclusiones integran EDA + supervisado + DNN + no supervisado
- [ ] Visualizaciones coherentes con el estilo del EDA existente (misma paleta, mismos formatos)
- [ ] README.md creado y funcional
