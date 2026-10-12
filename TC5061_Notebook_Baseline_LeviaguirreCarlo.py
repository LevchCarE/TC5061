#!/usr/bin/env python
# coding: utf-8

# # TC5061 · Actividad | Dataset - Notebook | Individual
# 
# 
# - **Alumno:** Carlo Eugenio Leviaguirre Chávez
# - **Matricula:**. A01022264
# - **Curso:** TC5061 — Maestría en Inteligencia Artificial Aplicada, Tecnológico de Monterrey
# - **Fecha:** 4 de octubre de 2026
# - **Dataset:** *Breast Cancer* (programa SEER del Instituto Nacional del Cáncer de EE. UU., publicado en Kaggle) — archivo `Breast_Cancer.csv`
# 

# ## 0. Configuración del entorno
# 
# 

# In[6]:


from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


# In[7]:


# Semilla aleatoria: fija todos los procesos aleatorios para que los resultados se repitan igual
seed= 42
np.random.seed(seed)

# Carpetas de salida (sus archivos se registran después en MLflow)
DIR_FIGURAS = Path("figuras")
DIR_REPORTES = Path("reportes")
DIR_DATOS = Path("datos")

for carpeta in (DIR_FIGURAS, DIR_REPORTES, DIR_DATOS):
    carpeta.mkdir(exist_ok=True)

# Color fijo por clase en todas las gráficas: viva (azul), fallecida (rojo)
COLORES = {"Alive": "#4C72B0", "Dead": "#C44E52"}
ETIQUETAS = {"Alive": "Viva", "Dead": "Fallecida"}


# ## Carga y descripción del dataset
# 
# **Origen.** Datos del programa **SEER** (*Surveillance, Epidemiology, and End Results*) del Instituto Nacional del Cáncer de EE. UU., actualización de noviembre de 2017, publicados en Kaggle por Reihaneh Namdari. Incluye **mujeres con carcinoma ductal infiltrante y lobulillar de mama diagnosticadas entre 2006 y 2010**. Los autores excluyeron a pacientes con tamaño tumoral desconocido, con número desconocido de ganglios examinados o positivos, y con menos de 1 mes de supervivencia, quedando 4,024 pacientes.
# 
# **Problema.** Clasificación binaria (dos clases): predecir si al final del seguimiento la paciente está **viva (`Alive`)** o **fallecida (`Dead`)** a partir de la información clínica disponible al momento del diagnóstico.

# In[8]:


RUTA_DATOS = Path("Breast_Cancer.csv")
df_original = pd.read_csv(RUTA_DATOS)

print(f"Registros (filas): {df_original.shape[0]}")
print(f"Variables (columnas): {df_original.shape[1]}")


# In[9]:


df_raw = df_original.copy()
df_raw.head()


# In[10]:


df_raw.info()


# In[11]:


# Estructura: tipo de dato, valores no nulos, valores distintos y un ejemplo por columna
estructura = pd.DataFrame({
    "tipo": df_raw.dtypes,
    "Valores_nulos": df_raw.isna().sum(),
    "valores_distintos": df_raw.nunique(),
    "ejemplo_fila_1": df_raw.iloc[0],
})
estructura


# In[12]:


#variables por tipo

NUMERICAS = ["Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months"]

CATEGORICAS = ["Race", "Marital Status", "T Stage ", "N Stage", "6th Stage",
               "differentiate", "Grade", "A Stage", "Estrogen Status", "Progesterone Status"]

OBJETIVO = "Status"

print('=='*10)
print(f"Numéricas ({len(NUMERICAS)}):\n\n{NUMERICAS}")
print('=='*10)
print(f"\nCategóricas ({len(CATEGORICAS)}):\n\n {CATEGORICAS}")
print('=='*10+'\n')
print(f"Objetivo: {OBJETIVO}")
print('=='*10)


# ### Diccionario de datos
# 
# | Variable (original) | Tipo | Significado |
# |---|---|---|
# | `edad` (*Age*) | entero | Edad al diagnóstico, en años. |
# | `raza` (*Race*) | categórica | Grupo racial registrado por SEER: `White`, `Black`, `Other`. |
# | `estado_civil` (*Marital Status*) | categórica | `Married`, `Single`, `Divorced`, `Widowed`, `Separated`. |
# | `estadio_T` (*T Stage*) | categórica ordenada | Tamaño y extensión del tumor primario: T1 (≤ 20 mm), T2 (21–50 mm), T3 (> 50 mm), T4 (cualquier tamaño con extensión a pared del tórax o piel). |
# | `estadio_N` (*N Stage*) | categórica ordenada | Grado de afectación de los ganglios linfáticos cercanos: N1 (menor) a N3 (mayor). En este archivo todas las pacientes tienen al menos N1. |
# | `estadio_AJCC6` (*6th Stage*) | categórica ordenada | Etapa global del cáncer según el manual AJCC, 6.ª edición (IIA, IIB, IIIA, IIIB, IIIC). Se obtiene combinando T y N. |
# | `diferenciacion` (*differentiate*) | categórica ordenada | Qué tanto se parecen las células del tumor a las células normales: bien, moderadamente, poco o no diferenciadas (menos parecido = más agresivo). |
# | `grado` (*Grade*) | categórica ordenada | Grado histológico 1, 2, 3 y "anaplastic; Grade IV". |
# | `estadio_A` (*A Stage*) | categórica | Extensión según la clasificación propia de SEER: `Regional` o `Distant` (a distancia). |
# | `tamano_tumor_mm` (*Tumor Size*) | entero | Tamaño del tumor en milímetros. |
# | `receptor_estrogeno` (*Estrogen Status*) | binaria | `Positive` si las células tumorales tienen receptores de estrógeno (responden a tratamiento hormonal). |
# | `receptor_progesterona` (*Progesterone Status*) | binaria | Igual, para receptores de progesterona. |
# | `ganglios_examinados` (*Regional Node Examined*) | entero | Número de ganglios linfáticos cercanos que se examinaron. |
# | `ganglios_positivos` (*Reginol Node Positive*) | entero | Número de esos ganglios en los que se encontró cáncer. |
# | `meses_supervivencia` (*Survival Months*) | entero | Meses desde el diagnóstico hasta la muerte o hasta el último contacto con la paciente. |
# | `estado_vital` (*Status*) | categórica | **Variable objetivo**: `Alive` (viva) o `Dead` (fallecida) al final del seguimiento. |
# 
# **Resumen de la estructura:** 4,024 registros y 16 columnas: 5 numéricas enteras, 10 categóricas de texto y la variable objetivo. `grado` se leyó como texto aunque casi todos sus valores son números, porque una de sus categorías es una palabra; se corrige en el Paso 3.

# ## EDA
# 
# 

# ### 2.1 Distribución de la variable objetivo
# Saber qué tan balanceadas están las clases determina cómo dividir los datos y qué métrica usar.

# In[13]:


# proporción de cada clase
conteo = df_raw[OBJETIVO].value_counts()
proporcion = df_raw[OBJETIVO].value_counts(normalize=True)
display(pd.DataFrame({"registros": conteo, "proporción": proporcion.round(3)}))





# Gráfica de barras de las dos clases
fig, ax = plt.subplots(figsize=(5.5, 3.8))
valores = [conteo["Alive"], conteo["Dead"]]
ax.bar(["Viva (Alive)", "Fallecida (Dead)"], valores, color=[COLORES["Alive"], COLORES["Dead"]])
for i, v in enumerate(valores):
    ax.text(i, v + 40, f"{v}  ({v / len(df_raw):.1%})", ha="center")
ax.set_ylim(0, max(valores) * 1.15)
ax.set_ylabel("Número de pacientes")
ax.set_title("Distribución de la variable objetivo")
plt.tight_layout()
plt.savefig(DIR_FIGURAS / "01_distribucion_clases.png", bbox_inches="tight")
plt.show()


# **Lectura.** Hay 3,408 pacientes vivas (84.7 %) y 616 fallecidas (15.3 %). Es un **desbalance marcado** (más de 5 vivas por cada fallecida), con tres consecuencias:
# - Un modelo que siempre dijera "viva" acertaría 84.7 % de las veces sin aprender nada. Por eso la exactitud (*accuracy*) es engañosa aquí y se incluye un modelo de referencia "tonto" en el Paso 4.
# - La división entrenamiento/prueba debe ser **estratificada** (mantener la misma proporción de clases en ambos conjuntos).
# - Hay que decidir cómo tratar el desbalance al entrenar (se analiza en el Paso 4).

# ### 2.2 Estadísticas descriptivas
# 

# In[14]:


# Estadísticas descriptivas de las variables numéricas + asimetría

resumen = df_raw[NUMERICAS].describe().T
resumen["asimetria"] = df_raw[NUMERICAS].skew()
resumen.round(2)


# Edad entre 30 y 69 años, distribución casi simétrica.
# Tamaño del tumor y ganglios positivos tienen colas largas a la derecha (asimetría de 1.7 y 2.7): la mayoría tiene tumores pequeños (50\% menor que la media) y pocos ganglios con cáncer, pero hay casos extremos (tumores de 140 mm, 46 ganglios positivos).

# **Variables categóricas:** número de categorías, la más y la menos frecuente.

# In[15]:


# Resumen de las variables categóricas

resumen_cat = pd.DataFrame({
    "categorias distintas": df_raw[CATEGORICAS].nunique(),
    "mas_frecuente": [df_raw[c].value_counts().idxmax() for c in CATEGORICAS],
    "proporcion_mas_frecuente": [df_raw[c].value_counts(normalize=True).max() for c in CATEGORICAS],
    "menos_frecuente": [df_raw[c].value_counts().idxmin() for c in CATEGORICAS],
    "registros_menos_frecuente": [df_raw[c].value_counts().min() for c in CATEGORICAS],
}, index=CATEGORICAS)

resumen_cat.round(3)


# In[16]:


# Valores de cada variable categórica

for columna in CATEGORICAS:
    valores = {repr(k): v for k, v in df_raw[columna].value_counts().items()}
    print(f"{columna}: {valores}\n")


# ### 2.3 Distribución de las variables numéricas por estado vital

# In[17]:


# Histogramas de las 5 variables numéricas, separados por clase (normalizados para comparar formas)

COLOR_TRASLAPE = "black"

fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))

for ax, col in zip(axes, NUMERICAS):

    # Mismos límites de barra para ambas clases, para poder comparar barra por barra
    bordes = np.histogram_bin_edges(df_raw[col], bins=25)
    ancho = np.diff(bordes)

    # Altura de cada barra (densidad) para cada clase
    altura_viva, _ = np.histogram(df_raw.loc[df_raw[OBJETIVO] == "Alive", col], bins=bordes, density=True)
    altura_fallecida, _ = np.histogram(df_raw.loc[df_raw[OBJETIVO] == "Dead", col], bins=bordes, density=True)

    # Traslape: la parte que comparten ambas clases (la menor de las dos alturas)
    traslape = np.minimum(altura_viva, altura_fallecida)

    # Primero el traslape en negro; encima, lo que le sobra a la clase más alta en su color
    ax.bar(bordes[:-1], traslape, width=ancho, align="edge", color=COLOR_TRASLAPE, label="Traslape")
    ax.bar(bordes[:-1], altura_viva - traslape, width=ancho, align="edge", bottom=traslape,
           color=COLORES["Alive"], alpha=0.8, label=ETIQUETAS["Alive"])
    ax.bar(bordes[:-1], altura_fallecida - traslape, width=ancho, align="edge", bottom=traslape,
           color=COLORES["Dead"], alpha=0.8, label=ETIQUETAS["Dead"])

    ax.set_title(col, fontsize=11)
    ax.set_yticks([])

axes[0].legend(title="Estado vital")
fig.suptitle("Distribución de las variables numéricas por estado vital", fontsize=14, y=1.03)
plt.tight_layout()
plt.savefig(DIR_FIGURAS / "02_histogramas_por_clase.png", bbox_inches="tight")
plt.show()


# Hay mucho traslape de vivas y fallecidas, pero es claro que conforme más edad hay más fallecimiento. Las fallecidas tienden a tener **tumores más grandes** y **más ganglios positivos despues del 5**, aunque con mucho traslape. Es claro que la variable que más peligro tiene de fallecimiento es el cancer, en los primeros 50 meses es mas peligroso.

# In[18]:


# Conversión temporal (solo para el EDA) de las variables ordenadas a números según su orden
ordinal_eda = pd.DataFrame({
    "estadio_T": df_raw["T Stage "].str.strip().map({"T1": 1, "T2": 2, "T3": 3, "T4": 4}),
    "estadio_N": df_raw["N Stage"].str.strip().map({"N1": 1, "N2": 2, "N3": 3}),
    "estadio_AJCC6": df_raw["6th Stage"].str.strip().map({"IIA": 1, "IIB": 2, "IIIA": 3, "IIIB": 4, "IIIC": 5}),
    "grado": df_raw["Grade"].str.strip().map({"1": 1, "2": 2, "3": 3, "anaplastic; Grade IV": 4}),
})


tabla_spearman = pd.concat([df_raw[NUMERICAS[:-1]], ordinal_eda], axis=1)
corr = tabla_spearman.corr()

mascara = np.triu(np.ones_like(corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, mask=mascara, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0,
            square=True, linewidths=0.4, cbar_kws={"shrink": 0.75, "label": "Correlación de Spearman"}, ax=ax)
ax.set_title("Correlación entre variables", fontsize=13)
ax.grid(False)
plt.tight_layout()
plt.savefig(DIR_FIGURAS / "04_matriz_correlacion.png", bbox_inches="tight")
plt.show()


# ##  Limpieza de datos
# 

# ### 3.1 Valores faltantes
# Además de los nulos explícitos, se revisan textos que suelen usarse en lugar de un dato faltante (por ejemplo `"Unknown"` o `"?"`, comunes en registros clínicos).

# In[19]:


# 
faltantes = df_raw.isnull().sum()
print(f"Valores nulos en total: {faltantes}\n")

# Textos que suelen representar un faltante (se comparan sin espacios y en minúsculas)

marcadores = {"", "?", "na", "n/a", "nan", "null", "unknown", "none",".", "!", "*"}
columnas_texto = [c for c in df_raw.columns if not pd.api.types.is_numeric_dtype(df_raw[c])]
sospechosos = sum(df_raw[c].str.strip().str.lower().isin(marcadores).sum() for c in columnas_texto)
print(f"Marcadores de faltante en columnas de texto: {sospechosos}\n")

# Ceros o negativos en conteos y medidas que deben ser positivos
print(f"Edad, tamaño o ganglios positivos menores o iguales a 0: "
      f"{(df_raw[['Age', 'Tumor Size', 'Reginol Node Positive']] <= 0).sum().sum()}")


# ### 3.2 Duplicados
# 

# In[20]:


duplicados = df_raw.duplicated(keep=False)

print(f"Filas duplicadas (copias adicionales): {df_raw.duplicated().sum()}")
df_raw[duplicados]


# In[21]:


# Se elimina la copia adicional y se crea la copia de trabajo
df_trabajo = df_raw.drop_duplicates().reset_index(drop=True)
# Variable objetivo: 1 = fallecida, 0 = viva
df_trabajo["objetivo"] = (df_trabajo["Status"] == "Dead").astype("int64")

print(f"Registros antes: {len(df_raw)} | después: {len(df_trabajo)}")


# ### 3.7 Valores atípicos (*outliers*)
#  Un valor se marca como atípico si cae por debajo de Q1 − 1.5·IQR o por encima de Q3 + 1.5·IQR. Es la misma regla que dibujan los diagramas de caja.

# In[22]:


# Variables numéricas que entran al modelo (ya sin meses de supervivencia)
NUMERICAS_MODELO = ["Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive"]

# Límites de la regla del rango intercuartílico para cada variable
q1 = df_trabajo[NUMERICAS_MODELO].quantile(0.25)
q3 = df_trabajo[NUMERICAS_MODELO].quantile(0.75)
iqr = q3 - q1
es_atipico = (df_trabajo[NUMERICAS_MODELO] < q1 - 1.5 * iqr) | (df_trabajo[NUMERICAS_MODELO] > q3 + 1.5 * iqr)

# Conteo por variable y límite superior de la regla
display(pd.DataFrame({"atipicos": es_atipico.sum(), "limite_superior": (q3 + 1.5 * iqr).round(1),
                      "maximo": df_trabajo[NUMERICAS_MODELO].max()}))

# Filas que tienen al menos un valor atípico
fila_con_atipico = es_atipico.any(axis=1)
print(f"Filas con al menos un valor atípico: {fila_con_atipico.sum()} de {len(df_trabajo)} ({fila_con_atipico.mean():.1%})")


# In[23]:


# ¿En qué clase se concentran las filas con valores atípicos?
es_fallecida = df_trabajo["objetivo"] == 1
es_viva = df_trabajo["objetivo"] == 0

# Conteos de cada combinación: clase y si tiene o no algún atípico
fallecidas_con = (es_fallecida & fila_con_atipico).sum()
fallecidas_sin = (es_fallecida & ~fila_con_atipico).sum()
vivas_con = (es_viva & fila_con_atipico).sum()
vivas_sin = (es_viva & ~fila_con_atipico).sum()

# Armar la tabla a partir de un diccionario
tabla_atipicos = pd.DataFrame({
    "con al menos un atípico": [fallecidas_con, vivas_con, fallecidas_con + vivas_con],
    "sin atípicos": [fallecidas_sin, vivas_sin, fallecidas_sin + vivas_sin],
    "Total": [fallecidas_con + fallecidas_sin, vivas_con + vivas_sin, len(df_trabajo)],
}, index=["Fallecida", "Viva", "Total"])
display(tabla_atipicos)

# Porcentaje de cada clase que tiene algún atípico
porcentaje_f = fallecidas_con / (fallecidas_con + fallecidas_sin)
porcentaje_v = vivas_con / (vivas_con + vivas_sin)
print(f"Fallecidas con algún atípico: {porcentaje_f:.1%} | Vivas con algún atípico: {porcentaje_v:.1%}")


# In[24]:


# Diagramas de caja

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
colores_por_clase = {0: COLORES["Alive"], 1: COLORES["Dead"]}   # 0 = viva, 1 = fallecida

for ax, col in zip(axes, NUMERICAS_MODELO):
    sns.boxplot(data=df_trabajo, x="objetivo", y=col, hue="objetivo",
                palette=colores_por_clase, legend=False, fliersize=2.5, ax=ax)
    ax.set_xticks([0, 1], ["Viva", "Fallecida"])
    ax.set(xlabel="", title=col)

fig.suptitle("Valores atípicos por estado vital", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(DIR_FIGURAS / "07_boxplots_atipicos.png", bbox_inches="tight")
plt.show()


# ### 3.9 Codificación final y dataset limpio
# 

# In[25]:


 # Tablas fijas de conversión

MAPA_T = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
MAPA_N = {"N1": 1, "N2": 2, "N3": 3}
MAPA_GRADO = {"1": 1, "2": 2, "3": 3, "anaplastic; Grade IV": 4}

df_limpio = pd.DataFrame({
    "edad": df_trabajo["Age"],
    "tamano_tumor_mm": df_trabajo["Tumor Size"],
    "ganglios_examinados": df_trabajo["Regional Node Examined"],
    "ganglios_positivos": df_trabajo["Reginol Node Positive"],
    "estadio_T": df_trabajo["T Stage "].map(MAPA_T),
    "estadio_N": df_trabajo["N Stage"].map(MAPA_N),
    "grado": df_trabajo["Grade"].str.strip().map(MAPA_GRADO),
    "estadio_A_distante": (df_trabajo["A Stage"] == "Distant").astype("int64"),
    "receptor_estrogeno_positivo": (df_trabajo["Estrogen Status"] == "Positive").astype("int64"),
    "receptor_progesterona_positivo": (df_trabajo["Progesterone Status"] == "Positive").astype("int64"),
    "raza": df_trabajo["Race"],
    "estado_civil": df_trabajo["Marital Status"],
    "objetivo": df_trabajo["objetivo"],
})

# Control: ninguna conversión debe haber dejado valores vacíos
assert df_limpio.notna().all().all(), "Una conversión produjo valores faltantes"
df_limpio.head()


# In[26]:


print(df_limpio.isna().sum())


# In[49]:


# Grupos de variables para el pipeline del modelo
VARS_NUMERICAS = ["edad", "tamano_tumor_mm", "ganglios_examinados", "ganglios_positivos",
                  "estadio_T", "estadio_N", "grado", "estadio_A_distante",
                  "receptor_estrogeno_positivo", "receptor_progesterona_positivo"]
VARS_NOMINALES = ["raza", "estado_civil"]

# Separar variables explicativas (X) y variable objetivo (y)
X = df_limpio[VARS_NUMERICAS + VARS_NOMINALES]
y = df_limpio["objetivo"]

# Guardar una copia del dataset limpio (se registra en MLflow como evidencia)
df_limpio.to_csv(DIR_DATOS / "breast_cancer_limpio.csv", index=False)


# ## Entrenamiento del modelo base
# 

# In[28]:


TEST_SIZE = 0.20

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
)

# Verificar tamaños y que la proporción de fallecidas se mantenga
pd.DataFrame({
    "registros": [len(y), len(y_train), len(y_test)],
    "fallecidas": [y.sum(), y_train.sum(), y_test.sum()],
    "proporción_fallecidas": [y.mean(), y_train.mean(), y_test.mean()],
}, index=["total", "entrenamiento", "prueba"]).round(3)


# ### 4.2 Referencia mínima: clasificador "tonto"
# Antes del baseline se entrena un DummyClassifier que siempre predice la clase más frecuente (viva), sin mirar las variables.
# Cualquier modelo que no lo supere claramente no está aprendiendo nada.

# In[50]:


# Modelo de referencia: siempre predice la clase mayoritaria
modelo_dummy = DummyClassifier(strategy="most_frequent")
modelo_dummy.fit(X_train, y_train)


# ### 4.3 Modelo base: regresión logística
# elegí este modelo porque es el modelo más elemental de clasificación binaria

# In[51]:


PARAMS_LR = {
    "C": 1.0,            
    "solver": "lbfgs",  
    "max_iter": 1000,    
}


def construir_modelo(peso_clases):

    preprocesamiento = ColumnTransformer(transformers=[
        ("numericas", StandardScaler(), VARS_NUMERICAS),
        ("nominales", OneHotEncoder(handle_unknown="ignore", sparse_output=False), VARS_NOMINALES),
    ])
    return Pipeline(steps=[
        ("preprocesamiento", preprocesamiento),
        ("clasificador", LogisticRegression(**PARAMS_LR, class_weight=peso_clases, random_state=seed)),
    ])


# In[52]:


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
METRICAS = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]


comparacion = {}
for peso, nombre in [(None, "sin ponderar"), ("balanced", "balanceado")]:
    resultado = cross_validate(construir_modelo(peso), X_train, y_train, cv=cv, scoring=METRICAS)
    comparacion[nombre] = {m: resultado[f"test_{m}"].mean() for m in METRICAS}

pd.DataFrame(comparacion).round(3)


# In[53]:


# Modelo base elegido
PESO_CLASES = "balanced"
modelo_lr = construir_modelo(PESO_CLASES)


resultados_cv = cross_validate(modelo_lr, X_train, y_train, cv=cv, scoring=METRICAS)
tabla_cv = pd.DataFrame({m: resultados_cv[f"test_{m}"] for m in METRICAS},
                        index=[f"pliegue {i + 1}" for i in range(cv.get_n_splits())])
resumen_cv = tabla_cv.agg(["mean", "std"]).rename(index={"mean": "promedio", "std": "desv. estándar"})
pd.concat([tabla_cv, resumen_cv]).round(3)


# In[54]:


# Entrenamiento final 
modelo_lr.fit(X_train, y_train)
n_iter = modelo_lr.named_steps["clasificador"].n_iter_[0]
n_columnas = len(modelo_lr.named_steps["preprocesamiento"].get_feature_names_out())
print(f"Modelo entrenado. Iteraciones hasta converger: {n_iter} (máximo permitido: {PARAMS_LR['max_iter']})")
print(f"Columnas que recibe la regresión tras el preprocesamiento: {n_columnas}")
modelo_lr


# ## Evaluación del modelo

# In[55]:


def evaluar(modelo, X_eval, y_eval):
    prediccion = modelo.predict(X_eval)
    probabilidad = modelo.predict_proba(X_eval)[:, 1]   # probabilidad de fallecer
    return {
        "accuracy": accuracy_score(y_eval, prediccion),
        "precision": precision_score(y_eval, prediccion, zero_division=0),
        "recall": recall_score(y_eval, prediccion),
        "f1": f1_score(y_eval, prediccion, zero_division=0),
        "roc_auc": roc_auc_score(y_eval, probabilidad),
        "average_precision": average_precision_score(y_eval, probabilidad),
    }


# In[56]:


# Métricas en prueba para ambos modelos, y en entrenamiento para revisar overfitting
metricas_dummy = evaluar(modelo_dummy, X_test, y_test)
metricas_lr = evaluar(modelo_lr, X_test, y_test)
metricas_lr_train = evaluar(modelo_lr, X_train, y_train)

tabla_metricas = pd.DataFrame({
    "Dummy · prueba": metricas_dummy,
    "Regresión logística · entrenamiento": metricas_lr_train,
    "Regresión logística · validación cruzada": resumen_cv.loc["promedio"],
    "Regresión logística · prueba": metricas_lr,
}).round(3)
tabla_metricas


# In[57]:


# Matriz de confusión del baseline en el conjunto de prueba
prediccion_lr = modelo_lr.predict(X_test)
matriz = confusion_matrix(y_test, prediccion_lr)
vn, fp, fn, vp = matriz.ravel()

fig, ax = plt.subplots(figsize=(5, 4.3))
ConfusionMatrixDisplay(matriz, display_labels=["Viva (0)", "Fallecida (1)"]).plot(
    ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set(xlabel="Predicción del modelo", ylabel="Estado vital real",
       title="Matriz de confusión · regresión logística (prueba)")
ax.grid(False)
plt.tight_layout()
plt.savefig(DIR_FIGURAS / "08_matriz_confusion.png", bbox_inches="tight")
plt.show()



# In[37]:


# Curva ROC y curva precisión–sensibilidad en el conjunto de prueba
probabilidad_lr = modelo_lr.predict_proba(X_test)[:, 1]
tasa_fp, tasa_vp, _ = roc_curve(y_test, probabilidad_lr)
precision_c, sensibilidad_c, _ = precision_recall_curve(y_test, probabilidad_lr)
prevalencia = y_test.mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].plot(tasa_fp, tasa_vp, lw=2, color=COLORES["Dead"],
             label=f"Regresión logística (AUC = {metricas_lr['roc_auc']:.3f})")
axes[0].plot([0, 1], [0, 1], ls="--", color="gray", label="Azar (AUC = 0.500)")
axes[0].set(xlabel="Tasa de falsos positivos", ylabel="Sensibilidad (tasa de verdaderos positivos)",
            title="Curva ROC · conjunto de prueba")
axes[0].legend(loc="lower right")

axes[1].plot(sensibilidad_c, precision_c, lw=2, color=COLORES["Dead"],
             label=f"Regresión logística (AP = {metricas_lr['average_precision']:.3f})")
axes[1].axhline(prevalencia, ls="--", color="gray", label=f"Azar (AP = {prevalencia:.3f})")
axes[1].set(xlabel="Sensibilidad", ylabel="Precisión", ylim=(0, 1.02),
            title="Curva precisión–sensibilidad · conjunto de prueba")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig(DIR_FIGURAS / "09_curvas_roc_y_precision.png", bbox_inches="tight")
plt.show()


# In[58]:


# Reporte completo 
reporte = classification_report(y_test, prediccion_lr, target_names=["Viva (0)", "Fallecida (1)"], digits=3)
(DIR_REPORTES / "reporte_clasificacion.txt").write_text(reporte)
print(reporte)


# In[39]:


# Distribución de la probabilidad de fallecer asignada por el modelo, por estado vital real
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for clase, valor in [("Alive", 0), ("Dead", 1)]:
    ax.hist(probabilidad_lr[y_test.values == valor], bins=25, range=(0, 1), alpha=0.6, density=True,
            color=COLORES[clase], label=ETIQUETAS[clase])
ax.axvline(0.5, ls="--", color="black", lw=1, label="Umbral de decisión (0.5)")
ax.set(xlabel="Probabilidad de fallecer asignada por el modelo", ylabel="Densidad", yticks=[],
       title="¿Qué tan separadas quedan las clases? (prueba)")
ax.legend()
plt.tight_layout()
plt.savefig(DIR_FIGURAS / "10_distribucion_probabilidades.png", bbox_inches="tight")
plt.show()

# Mediana de la probabilidad asignada a cada grupo
print(f"Mediana de probabilidad · vivas: {np.median(probabilidad_lr[y_test.values == 0]):.3f} | "
      f"fallecidas: {np.median(probabilidad_lr[y_test.values == 1]):.3f}")


# ## Registro del experimento en MLflow
# 
# se hicieron dos modelos:  Dummy y logística, para que la comparación quede visible en la interfaz de MLflow.

# In[59]:


# Dónde guarda MLflow la información: base de datos SQLite local en la carpeta del notebook
URI_TRACKING = "sqlite:///mlflow.db"
NOMBRE_EXPERIMENTO = "TC5061_Baseline_SEER_CancerMama"

mlflow.set_tracking_uri(URI_TRACKING)
experimento = mlflow.set_experiment(NOMBRE_EXPERIMENTO)

print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experimento: {experimento.name} (id {experimento.experiment_id})")


# In[62]:


# Información común a ambos modelos
PARAMS_COMUNES = {
    "test_size": TEST_SIZE,
    "random_state": seed,
    "division_estratificada": True,
    "n_entrenamiento": len(X_train),
    "n_prueba": len(X_test),
    "n_variables": X.shape[1],
    "variables_excluidas": "diferenciacion, estadio_AJCC6, meses_supervivencia",
}

ETIQUETAS_COMUNES = {
    "dataset": "SEER Breast Cancer (NCI, nov 2017) - Kaggle reihanenamdari",
    "tipo_problema": "clasificacion_binaria",
    "clase_positiva": "fallecida (1)",
    "etapa": "baseline",
    "autor": "Carlo Eugenio Leviaguirre Chavez",
    "curso": "TC5061",
}


MLFLOW_V3 = int(mlflow.__version__.split(".")[0]) >= 3


def registrar_modelo(modelo, nombre):
    # Guarda el modelo 
    firma = infer_signature(X_train, modelo.predict(X_train))
    if MLFLOW_V3:
        return mlflow.sklearn.log_model(sk_model=modelo, name=nombre, signature=firma)
    return mlflow.sklearn.log_model(sk_model=modelo, artifact_path=nombre, signature=firma)


# In[63]:


# Modelo 1: dummy
with mlflow.start_run(run_name="dummy_clase_mayoritaria") as run_dummy:
    mlflow.set_tags({**ETIQUETAS_COMUNES, "mlflow.note.content": "Referencia mínima: siempre predice viva."})
    mlflow.log_params({**PARAMS_COMUNES, "modelo": "DummyClassifier", "estrategia": "most_frequent"})
    mlflow.log_metrics({f"test_{k}": v for k, v in metricas_dummy.items()})
    registrar_modelo(modelo_dummy, "modelo_dummy")

print(f"Ejecución Dummy registrada · run_id = {run_dummy.info.run_id}")


# In[65]:


# Modelo 2: regresión logística
with mlflow.start_run(run_name="baseline_regresion_logistica") as run_lr:
    mlflow.set_tags({**ETIQUETAS_COMUNES,
                     "mlflow.note.content": "Baseline: estandarización + one-hot + LogisticRegression balanceada."})

    # Parámetros: configuración del modelo, del preprocesamiento y de la división de datos
    mlflow.log_params({
        **PARAMS_COMUNES,
        "modelo": "LogisticRegression",
        "escalado": "StandardScaler",
        "codificacion_nominales": "OneHotEncoder(raza, estado_civil)",
        "penalizacion": "l2",
        **PARAMS_LR,
        "class_weight": PESO_CLASES,
        "umbral_decision": 0.5,
        "cv_pliegues": cv.get_n_splits(),
    })

    # Métricas: prueba, validación cruzada (promedio y desviación) y entrenamiento
    mlflow.log_metrics({f"test_{k}": v for k, v in metricas_lr.items()})
    mlflow.log_metrics({f"cv_{k}_promedio": v for k, v in resumen_cv.loc["promedio"].items()})
    mlflow.log_metrics({f"cv_{k}_desv": v for k, v in resumen_cv.loc["desv. estándar"].items()})
    mlflow.log_metrics({f"train_{k}": v for k, v in metricas_lr_train.items()})
    mlflow.log_metrics({"test_verdaderos_negativos": vn, "test_falsos_positivos": fp,
                        "test_falsos_negativos": fn, "test_verdaderos_positivos": vp})

    # gráficas, reporte, dataset limpio y modelo entrenado
    mlflow.log_artifacts(str(DIR_FIGURAS), artifact_path="figuras")
    mlflow.log_artifacts(str(DIR_REPORTES), artifact_path="reportes")
    mlflow.log_artifact(str(DIR_DATOS / "breast_cancer_limpio.csv"), artifact_path="datos")
    info_modelo_lr = registrar_modelo(modelo_lr, "modelo_baseline")

print(f"Ejecución baseline registrada · run_id = {run_lr.info.run_id}")


# In[66]:


ejecuciones = mlflow.search_runs(experiment_names=[NOMBRE_EXPERIMENTO], order_by=["start_time DESC"])
columnas_ver = ["tags.mlflow.runName", "params.modelo", "metrics.test_accuracy", "metrics.test_recall",
                "metrics.test_precision", "metrics.test_f1", "metrics.test_roc_auc", "metrics.test_average_precision"]
ejecuciones[columnas_ver].round(3)


# In[68]:


#archivos guardados en la ejecución del baseline
cliente = mlflow.tracking.MlflowClient()
for artefacto in cliente.list_artifacts(run_lr.info.run_id):
    print(("[carpeta] " if artefacto.is_dir else "[archivo] ") + artefacto.path)
    if artefacto.is_dir:
        for archivo in cliente.list_artifacts(run_lr.info.run_id, artefacto.path):
            print("    " + archivo.path)


# In[69]:


#  reproducibilidad, cargar el modelo desde MLflow y comparar sus predicciones con el original
modelo_recuperado = mlflow.sklearn.load_model(info_modelo_lr.model_uri)
mismas_predicciones = np.array_equal(modelo_recuperado.predict(X_test), modelo_lr.predict(X_test))

print(f"Modelo guardado en: {info_modelo_lr.model_uri}")
print(f"El modelo recuperado da exactamente las mismas predicciones: {mismas_predicciones}")


# ### Evidencia: interfaz de MLflow

# In[71]:


# Arrancar la interfaz de MLflow desde el notebook (mismo entorno y misma carpeta)
import subprocess, sys

print("mlflow.db existe en esta carpeta:", Path("mlflow.db").exists())
proceso_mlflow = subprocess.Popen([sys.executable, "-m", "mlflow", "ui",
                                   "--backend-store-uri", "sqlite:///mlflow.db", "--port", "5001"])
print("Abre en el navegador: http://127.0.0.1:5001")


# ![image.png](attachment:e12d746e-9538-4ffd-bbf7-66c018808d98.png)

# ## Reflexión sobre MLOps
# 
# este proyecto muestra la produccion y despliege end to end de un modelo de machine learning y su seguimiento en MLflow, 
# 
# Creo que mlflow es de gran relevancia para entender como van mejorando los modelos y tener comparaciones historicas, en lo particular, creo que me falta comprenderlo a profundidad, tuve que acudir a que cloude me ayudara puesto que tenia muchos bugs, resultó que tenia que importar ´import subprocess, sys´ para conseguir hacer el deploy. Creo que es una herramienta sumamente importante, por su parte en cuanto a la parte de el pipepline end to end, intenté llevar todo el proceso sin convertirlo en un estudio exhaustivo de ciencias estadísticas y no buscar mejores modelos, ni optimización de hiperparametros, sino simplemente seguir un pipeline end to end para llegar al deploy. Considero además que con agentes de IA, es posible generar automaticamente distintas pruebas de datos que se revisen en mlflow, y con esto ver su mejoría. Algo que considero importante, es que la fuga de información (data leakage) no es evidente en mlfow, entonces es importante tener las versiónes guardadas.  Considero particularmente importante a mlflow a la hora de optimizar un modelo para poder ver la mejoria con distintos hiperparametros. En mi contexto de trabajo, como físico que implementa tecnicas de ML para la física de altas energías,
# para realizar modelos de identificación de partículas, la optimización de hiperparametros es constante de acuerdo a las distintas variables que ocurren en una colisión en el CERN. Mlflow es util para llevar a cabo este seguimiento.

# In[ ]:




