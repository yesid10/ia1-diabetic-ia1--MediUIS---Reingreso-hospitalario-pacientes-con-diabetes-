# Plan Técnico: Modelado y Optimización - Segunda Parte Proyecto Diabetes

**Objetivo**: Entrenar, ajustar e interpretar 4 modelos (Gaussian NB, DT, RF, SVM) para predecir readmisión temprana (`<30`) en pacientes diabéticos.

**Contexto del problema**:
- Dataset: ~98,000 registros (post-limpieza)
- Clases: `NO` (mayoritaria) | `>30` (intermedia) | `<30` (minoritaria, objetivo crítico)
- Desbalance: `<30` es ~11-12% de los datos
- Features relevantes del EDA: `time_in_hospital`, `num_medications`, `number_diagnoses`, `number_inpatient`, `age`, `admission_type_label`, `insulin`, `diabetesMed`

---

## FASE 1: PREPROCESAMIENTO Y FEATURE ENGINEERING

### 1.1 Limpieza de datos
- Usar dataset post-EDA (ya filtrados muertos, hospicios, y columnas con >50% nulos)
- Verificar nulos restantes en features seleccionadas
- Imputar: moda para categóricas, mediana para numéricas

### 1.2 Codificación de variables categóricas
- **Ordinales** (`age`, `insulin`): OrdinalEncoder (respeta orden natural)
  - `[0-10)` → 0, `[10-20)` → 1, ..., `[90-100)` → 9
  - `No` → 0, `Steady` → 1, `Up` → 2, `Down` → 3
- **Nominales** (`race`, `gender`, `admission_type_label`, `discharge_disposition_label`): OneHotEncoder (crea dummies)
- **Binarias** (`change`, `diabetesMed`): LabelEncoder → {0, 1}

### 1.3 Feature Scaling/Normalización
**RECOMENDACIÓN: StandardScaler (media=0, desv.est.=1)**
- **Por qué**:
  - SVM es **muy sensible** a escalas (distancia euclidiana)
  - RF y NB son invariantes a escala, pero StandardScaler no daña
  - Permite comparación homogénea en GridSearch
  - Interpreta como "¿cuántas desviaciones estándar del promedio?"

### 1.4 Features a usar (validadas en EDA)
**Numéricas** (después de scaling):
- `time_in_hospital`
- `num_lab_procedures`
- `num_procedures`
- `num_medications`
- `number_diagnoses`
- `number_outpatient`
- `number_emergency`
- `number_inpatient`

**Categóricas** (después de encoding):
- `age` (ordinal →9 valores)
- `gender` (nominal → 2-3 dummies)
- `race` (nominal → 5-6 dummies)
- `admission_type_label` (nominal → 3-4 dummies)
- `discharge_disposition_label` (nominal → múltiples dummies)
- `admission_source_label` (nominal → múltiples dummies)
- `A1Cresult`, `insulin`, `change`, `diabetesMed` (ordinales/binarias)

**Total esperado**: ~30-40 features post-codificación

---

## FASE 2: ESTRATEGIA DE VALIDACIÓN Y DESBALANCE

### 2.1 Split de datos
**RECOMENDACIÓN: Train/Test estratificado (75/25)**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y  # Mantiene proporción de clases
)
```
- **Por qué stratify**: Garantiza que train y test tengan la misma distribución de `<30`, `>30`, `NO`
- **Por qué no K-Fold aquí**: Dataset es grande (~74k train, ~24k test), evita overhead computacional

### 2.2 Estrategia contra desbalance de clases
**RECOMENDACIÓN: Dos enfoques paralelos para COMPARAR**

#### Opción A: `class_weight='balanced'` en modelos (SVM, RF, LogReg)
- **Pros**: Simple, rápido, integrado en scikit-learn
- **Contras**: No funciona bien con datos muy desbalanceados (>1:10)
- **Modelos que lo soportan**: SVM, RF, NB con ajuste manual
- **Fórmula**: weight_i = n_samples / (n_classes × n_samples_i)
  - Si `<30` tiene 11% de datos → weight ≈ 1 / 0.11 ≈ 9x

#### Opción B: SMOTE en X_train (RECOMENDADO para este caso)
- **Pros**: Genera muestras sintéticas creíbles, menos pérdida de información
- **Contras**: Más tiempo computacional, riesgo de overfitting si se ajusta mal
- **Proporción sugerida**: Llevar `<30` a 25-30% del dataset (no 50%, evita desbalance inverso)
- **Implementación**:
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.3, random_state=42)  # <30 = 30% de las muestras
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**PLAN**: Entrenar PRIMERO con `class_weight='balanced'` (línea base rápida), LUEGO con SMOTE y comparar F1/Recall.

---

## FASE 3: BÚSQUEDA DE HIPERPARÁMETROS

### 3.1 Estrategia de búsqueda
**RECOMENDACIÓN: GridSearchCV (exhaustiva) + RandomizedSearchCV (si tiempo lo permite)**

- **GridSearchCV**: Define grilla cartesiana exacta
  - **Ventaja**: Garantiza encontrar la mejor combinación en el espacio definido
  - **Desventaja**: Lento si hay muchos parámetros/valores
  - **Casos de uso**: SVM (~6 parámetros), RF (~4 principales)

- **RandomizedSearchCV**: Muestreo aleatorio del espacio de búsqueda
  - **Ventaja**: Rápido, buen balance exploratorio
  - **Desventaja**: Puede perder óptimos locales
  - **Casos de uso**: Si tiempo es limitado o espacio muy grande

**Decisión final**: Usar **GridSearchCV** para todos (no hay restricción de tiempo)

### 3.2 Validación dentro del GridSearch
- **cv=5** dentro del GridSearch (5-Fold CV estratificado en los datos de train)
  - Valida ese modelo sin tocar el test set
  - Cada fold mantiene proporción de clases

### 3.3 Hiperparámetros por modelo

---

## MODELO 1: GAUSSIAN NAIVE BAYES

### Teoría
- Asume independencia entre features
- Assume distribución gaussiana (normal) de features
- **Ventaja**: Muy rápido, funciona bien con datos desbalanceados
- **Desventaja**: Supuesto de independencia no siempre realista

### Hiperparámetros a variar
**GaussianNB tiene pocos hyperparámetros nativos, pero sí:**
1. **var_smoothing** (important)
   - `var_smoothing`: Suma pequeña a varianza para evitar división por cero
   - **Rango sugerido**: `[1e-9, 1e-8, 1e-7, 1e-6]`
   - **Por qué**: Evita problemas numéricos; 1e-9 es conservative, 1e-6 más agresivo

### GridSearch Config
```python
param_grid_nb = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}
```
- **Total combinaciones**: 5 (muy rápido)
- **Método balanceo**: `class_weight` no aplica (NB no tiene ese param)
  → Usar SMOTE en X_train

---

## MODELO 2: DECISION TREE

### Teoría
- Divide features recursivamente para pureza máxima (Gini/Entropy)
- **Ventaja**: Interpreta features, maneja categorías bien
- **Desventaja**: Propensidad a overfitting, sensible a profundidad

### Hiperparámetros críticos
1. **max_depth**: Profundidad máxima del árbol
   - **Rango**: `[5, 10, 15, 20, None]`
   - **Por qué**: Evita memorizar; Dataset ~74k → profundidad 10-15 es típica

2. **min_samples_split**: Muestras mínimas para dividir un nodo
   - **Rango**: `[2, 5, 10, 20]`
   - **Por qué**: Evita splits en ruido; para ~74k registros, 10-20 es razonable

3. **min_samples_leaf**: Muestras mínimas en hoja final
   - **Rango**: `[1, 4, 8, 16]`
   - **Por qué**: Evita overfitting; si `<30` es 11% (~8k), leaf_min=4-8 garantiza ≥2 muestras por clase

4. **criterion**: Métrica de pureza
   - **Valores**: `['gini', 'entropy']`
   - **Por qué**: Gini es más rápido, entropy es ligeramente más estable en datos desbalanceados

### GridSearch Config
```python
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 4, 8, 16]
}
```
- **Total combinaciones**: 2 × 5 × 4 × 4 = 160 (aceptable, ~1-2 min)

---

## MODELO 3: RANDOM FOREST

### Teoría
- Ensemble de árboles, cada uno entrenado en muestras aleatorias (bagging)
- **Ventaja**: Reduce overfitting vs DT, maneja desbalance mejor, paralizable
- **Desventaja**: Menos interpretable que DT simple, más lento

### Hiperparámetros críticos
1. **n_estimators**: Número de árboles
   - **Rango**: `[50, 100, 200, 300]`
   - **Por qué**: Más árboles = más estabilidad; >200 suele saturar; comienza a saturar después de 200

2. **max_depth**: Profundidad de cada árbol
   - **Rango**: `[10, 15, 20, None]`
   - **Por qué**: RF es robusto, profundidad puede ser mayor que DT; None permite crecimiento libre

3. **min_samples_split**:
   - **Rango**: `[2, 5, 10]` (análogo a DT)

4. **min_samples_leaf**:
   - **Rango**: `[1, 4, 8]` (análogo a DT)

5. **max_features**: Características por split
   - **Valores**: `['sqrt', 'log2']`
   - **Por qué**: 'sqrt'=√n_features, 'log2'=log2(n_features); ambas reducen correlación entre árboles

6. **class_weight**:
   - **Valores**: `['balanced', 'balanced_subsample']`
   - **Por qué**: Penaliza errores en clase minoritaria durante construcción

### GridSearch Config
```python
param_grid_rf = {
    'n_estimators': [50, 100, 150, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 4, 8],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}
```
- **Total combinaciones**: 5 × 4 × 3 × 3 × 2 × 2 = 1,440 (MUCHO, ~10-20 min)
- **Optimización**: Usar RandomizedSearchCV con n_iter=50-100 si es muy lento

---

## MODELO 4: SUPPORT VECTOR MACHINE (SVM)

### Teoría
- Busca hiperplano óptimo que maximiza margen entre clases
- **Ventaja**: Excelente con datos escalados, kernel RBF maneja no-linearidad
- **Desventaja**: Lento con datasets grandes, sensible a scaling, necesita tuning fino

### Hiperparámetros críticos
1. **C**: Regularización (trade-off entre margen y error)
   - **Rango**: `[0.1, 1, 10, 100, 1000]`
   - **Por qué**: C bajo = margen ancho (tolerante), C alto = menos error en train (rígido)
   - **Para desbalance**: C bajo puede estar bien, SVM ya tiene `class_weight`

2. **kernel**: Tipo de función kernel
   - **Valores**: `['rbf', 'poly']` (no 'linear' porque es linealmente no separable)
   - **Por qué**: RBF es versátil (gaussiana), poly también captura relaciones; linear sería subóptimo

3. **gamma** (solo para 'rbf' y 'poly'):
   - **Rango**: `['scale', 'auto']` o `[0.001, 0.01, 0.1, 1]`
   - **Por qué**: Gamma controla influencia de cada punto; escala = 1/(n_features * X.var()), auto = 1/n_features

4. **degree** (solo para 'poly'):
   - **Rango**: `[2, 3, 4]`
   - **Por qué**: Complejidad del polinomio; 2-3 típico, >4 suele overfitear

5. **class_weight**:
   - **Valores**: `['balanced', None]`
   - **Por qué**: 'balanced' escala inverso a frecuencia de clase

### GridSearch Config
```python
param_grid_svm = {
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'degree': [2, 3],  # solo aplica si kernel='poly'
    'class_weight': ['balanced', None]
}
```
- **Total combinaciones**: Demasiadas (5 × 2 × 5 × 2 × 2 = 200, pero muchas inválidas)
- **Recomendación**: Usar **RandomizedSearchCV con n_iter=50** (SVM suele ser lento)

---

## FASE 4: ESTRATEGIA DE ENTRENAMIENTO

### 4.1 Pipeline unificada (qué significa)
Una Pipeline es una cadena de transformaciones + modelo final:
```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(...)),  # opcional, se aplica en X_train
    ('model', SVC(...))
])
```
**Ventaja**: Evita data leakage (scaling se fit SOLO en train)
**Desventaja**: Más código, menos flexible

**RECOMENDACIÓN**: Usar Pipeline para SVM (crítico), independiente para otros

### 4.2 Orden de ejecución

```
1. Cargar datos limpios del EDA
2. Codificar y escalar features (StandardScaler)
3. Split train/test con stratify
4. Para CADA modelo:
   a. Aplicar SMOTE en X_train
   b. Entrenar con GridSearchCV (cv=5):
      - Evaluar métrica: F1-score macro (default) + Recall <30
      - Guardar mejor modelo
   c. Predecir en X_test sin tocar (no re-aplicar SMOTE)
   d. Calcular métricas: Accuracy, Precision, Recall, F1 (macro y por clase), AUC-ROC
5. Comparar resultados entre 4 modelos
6. Seleccionar 2 mejores para análisis profundo
```

---

## FASE 5: MÉTRICAS DE EVALUACIÓN (RECOMENDADAS)

### Métricas principales (para optimizar en GridSearch)
1. **F1-score macro** (primary)
   - Promedio F1 de las 3 clases, sin pesar por frecuencia
   - **Por qué**: Balancea precisión y recall, no penaliza clase mayoritaria
   - **Fórmula**: F1_macro = (F1_NO + F1_>30 + F1_<30) / 3

2. **Recall en clase `<30`** (secondary)
   - Proporción de `<30` que el modelo detecta correctamente
   - **Por qué**: Minimizar falsos negativos; si patient tiene alto riesgo, el modelo DEBE detectarlo
   - **Fórmula**: Recall_<30 = TP_<30 / (TP_<30 + FN_<30)

### Métricas de validación (reporte final)
- **Accuracy**: (TP + TN) / Total (contexto, no principal)
- **Precision por clase**: Fiabilidad de predicción positiva
- **Recall por clase**: Cobertura de verdaderos positivos
- **F1 por clase**: Balance precision/recall
- **AUC-ROC macro**: Área bajo curva ROC (One-vs-Rest para 3 clases)
- **Confusion Matrix**: Tabla de predicciones vs reales

---

## FASE 6: ANÁLISIS PROFUNDO (post-modelado)

### 6.1 Comparativa entre 4 modelos
Tabla con:
- Modelo | Accuracy | F1-macro | Recall_<30 | Precision_<30 | AUC-ROC | Tiempo_train

### 6.2 Análisis de errores del mejor modelo
- Matriz de confusión: ¿A qué clase se asignan erróneamente los casos `<30`?
- Importancia de features (si aplica):
  - RF/DT: `.feature_importances_`
  - SVM: no tiene (black box)
- Subgroup analysis: ¿El modelo falla en grupos específicos?
  - Por edad: ¿Funciona igual en jóvenes vs ancianos?
  - Por tipo de admisión: ¿Function mejor en Emergency que Elective?
  - Por dosis de insulina: ¿Pacientes con cambios de insulina se predicen mejor?

---

## RESUMEN DE CONFIGURACIONES FINALES

| Aspecto | Decisión | Justificación |
|---------|----------|---------------|
| **Split** | Train/Test 75/25 estratificado | Dataset grande, evita K-Fold overhead |
| **Scaling** | StandardScaler | Crítico para SVM, no daña otros |
| **Desbalance** | SMOTE (ratio 0.3) + `class_weight='balanced'` en RF | SMOTE crea muestras, class_weight amplifica peso |
| **GridSearch** | GridSearchCV (cv=5) excepto SVM→RandomizedSearchCV(n_iter=50) | Exhaustiva para modelos pequeños, aleatoria para SVM lento |
| **Métrica_Optimizar** | F1-macro + maximizar Recall_<30 | Balancea clases, minimiza falsos negativos críticos |
| **Métrica_Reportar** | Accuracy, Precision, Recall, F1 (macro y por clase), AUC-ROC, CM | Análisis comprehensivo |
| **Features** | 30-40 post-codificación (del EDA) | Ya validadas, interpretables |
| **Random_State** | 42 (todos los modelos) | Reproducibilidad |

---

## TIMELINE ESTIMADO
- **Codificación y Scaling**: 15 min
- **GridSearch NB**: 2 min
- **GridSearch DT**: 2-5 min
- **GridSearch RF**: 10-20 min (puede ser Randomized si es lento)
- **GridSearch SVM**: 5-15 min (RandomizedSearchCV)
- **Evaluación y análisis**: 10 min
- **TOTAL**: ~50-60 minutos de ejecución

---

## DECISIONES PENDIENTES (retroalimentación del usuario)
1. ¿Aceptas SMOTE + class_weight como estrategia dual?
2. ¿RandomizedSearchCV para SVM o GridSearchCV exhaustivo?
3. ¿Incluir más features del EDA o solo las principales?
4. ¿Dejar RF con búsqueda exhaustiva (1440 combos) o Random?
