# Predicción de abandono de clientes de tarjeta de crédito (BankChurners)

Proyecto de Minería de Datos en Python. Se construye un modelo que anticipa qué clientes de tarjeta de crédito van a abandonar el banco, para que el banco pueda ofrecerles acciones de retención a tiempo. El modelo se despliega como una aplicación web con Streamlit.

**Integrantes:** Sebastian Muñoz · Santiago Martinez · Carlos Baena

**Datos:** [Credit Card Customers (Kaggle)](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) · 10.127 clientes, 23 columnas. Variable objetivo: `Attrition_Flag` (`Attrited Customer` / `Existing Customer`), con ≈ 16 % de clientes que abandonan.

## Resultados

| | |
|---|---|
| **Modelo final** | XGBoost con `max_depth=3`, `n_estimators=400`, `learning_rate=0.1` |
| **Variables usadas** | 11 (de 19), todas de comportamiento transaccional y de relación con el banco |
| **f1_macro en clientes reales** | ≈ 0,945 |
| **Exactitud** | ≈ 0,97 |
| **Clientes que abandonan detectados (recall)** | ≈ 92 % |

Estas cifras salen de la validación cruzada de 10 folds aplicando SMOTE solo dentro del entrenamiento de cada fold y evaluando con clientes reales (sección 6.1 del notebook).

## Qué se hizo

1. **Calidad de datos:** diagnóstico de seis dimensiones (completitud, unicidad, validez, consistencia, exactitud y oportunidad). Se detectaron valores `Unknown` en educación, estado civil e ingreso, y dos columnas de un modelo Naive Bayes que contienen la respuesta (fuga de datos) y se eliminaron.
2. **Selección de factores:** correlaciones entre predictoras y con la variable objetivo, más la importancia de variables de un Random Forest. Se pasó de 19 a 11 variables.
3. **Balanceo:** SMOTE para que la clase de abandono quede en el 50 % de la mayoritaria.
4. **Validación cruzada estratificada de 10 folds** de seis modelos (árbol, Random Forest, XGBoost, KNN, red neuronal y SVM), con revisión de overfitting y underfitting. Todos quedaron en buen ajuste; XGBoost fue el mejor.
5. **Hiperparametrización** de XGBoost con `GridSearchCV` (48 combinaciones × 10 folds). Se eligió la versión con `max_depth=3`: la misma calidad que la más compleja, pero con menos overfitting.
6. **Despliegue** del modelo final en una aplicación Streamlit.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `Proyecto_Attrition_BankChurners.ipynb` | Notebook con todo el análisis, los modelos y las conclusiones |
| `app.py` | Aplicación web de Streamlit |
| `modelo-attrition.pkl` | Modelo final guardado (modelo, codificador de la variable objetivo, lista de variables y normalizador) |
| `requirements.txt` | Librerías necesarias, con sus versiones |
| `.streamlit/config.toml` | Colores del tema de la aplicación |

## Cómo ejecutarlo

Requiere Python 3.11 o superior.

```bash
# 1. Instalar las librerías
pip install -r requirements.txt

# 2. Abrir la aplicación
streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`. Tiene tres pestañas:

* **Evaluar un cliente:** se ingresan los datos de un cliente y devuelve la probabilidad de abandono, el nivel de riesgo, la acción recomendada y las cinco variables que más pesaron en la predicción.
* **Evaluar una cartera:** se sube un CSV con clientes (la plantilla descargable o el archivo original `BankChurners.csv`) y devuelve la lista ordenada por riesgo.
* **Cómo funciona el modelo:** resumen de la construcción, la calidad y la importancia de las variables.

### Para ejecutar el notebook

El notebook lee `BankChurners.csv` desde la misma carpeta. Descárgalo desde [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) y colócalo junto al notebook. Además de las librerías de `requirements.txt`, necesita `imbalanced-learn` y `matplotlib`:

```bash
pip install imbalanced-learn matplotlib jupyter
```

## Limitaciones

* La base es una foto sin fecha: no se sabe a qué periodo corresponde y el modelo conviene reentrenarlo con datos nuevos.
* Los niveles de riesgo de la aplicación (30 % y 60 %) son un punto de partida que el negocio debe ajustar según el costo real de una campaña de retención.
* La exactitud de los datos frente al banco de origen no se puede verificar, porque es un dataset público de Kaggle.
