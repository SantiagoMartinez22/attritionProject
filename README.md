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
| `BankChurners.csv` | Base de datos original de Kaggle (10.127 clientes, 23 columnas) |
| `app.py` | Aplicación web de Streamlit |
| `modelo-attrition.pkl` | Modelo final guardado (modelo, codificador de la variable objetivo, lista de variables y normalizador) |
| `requirements.txt` | Librerías necesarias, con sus versiones |
| `.streamlit/config.toml` | Colores del tema de la aplicación |
