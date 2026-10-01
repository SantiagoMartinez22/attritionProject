#!/usr/bin/env python
# coding: utf-8

# # Despliegue del modelo predictivo – Riesgo de abandono (BankChurners)
# 
# **Integrantes:** Sebastian Muñoz · Santiago Martinez · Carlos Baena
# 
# Aplicación web del modelo final (**XGBoost hiperparametrizado con GridSearchCV**) construida con **Streamlit**.
# 
# Este notebook funciona de **dos formas** con el mismo código:
# 
# | Cómo se ejecuta | Qué hace |
# |---|---|
# | **En Jupyter** (Run All) | Prueba el modelo: carga el `.pkl`, predice clientes de ejemplo, muestra el medidor de riesgo, la explicación de cada predicción, evalúa la cartera completa y grafica la importancia de variables. |
# | **Descargado como `app.py`** y ejecutado con `streamlit run app.py` | Abre la aplicación web con sus tres pestañas: evaluar un cliente, evaluar una cartera y cómo funciona el modelo. |
# 
# El código detecta automáticamente en cuál de los dos modos está (variable `EN_STREAMLIT`).
# 
# **Archivos necesarios en la misma carpeta:** `modelo-attrition.pkl` (generado en el notebook del proyecto),
# `requirements.txt` y, para las pruebas de cartera, `BankChurners.csv`.

# ## 1. Librerías y detección del modo de ejecución

# In[1]:


#Despliegue del modelo de abandono de clientes (BankChurners) con Streamlit
#Ejecutar con:  streamlit run app.py
import importlib.util
import subprocess
import sys

#Instala las librerías que falten (por ejemplo en Google Colab, que no trae streamlit).
#Se hace con Python y no con "!pip" para que el archivo siga funcionando al descargarlo como app.py
for paquete in ["streamlit", "xgboost", "altair"]:
    if importlib.util.find_spec(paquete) is None:
        print(f"Instalando {paquete}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", paquete])

import io
import math
import pickle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from streamlit.runtime import exists as _streamlit_activo

#True cuando se ejecuta con "streamlit run app.py"; False dentro de Jupyter
EN_STREAMLIT = _streamlit_activo()
print("Modo:", "aplicación Streamlit" if EN_STREAMLIT else "notebook (pruebas del modelo)")


# ## 2. Estilos de la interfaz
# Colores, tipografías y tarjetas (CSS). Se usan en la app y también en las pruebas visuales del notebook.

# In[2]:


ESTILOS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap');

:root { --tinta:#0f2a3d; --tinta2:#3d5566; --papel:#f6f3ee; --linea:#e3ddd3;
        --alto:#c2412d; --medio:#d99a1e; --bajo:#2f8a5b; --acento:#1f6f8b; }
html, body, [class*="css"], .stMarkdown, .stText, button, input, label { font-family:'Manrope', system-ui, sans-serif; }
.stApp { background: var(--papel); }
.block-container { padding-top: 1.6rem; max-width: 1240px; }
h1, h2, h3, h4 { font-family:'Fraunces', Georgia, serif !important; color: var(--tinta); letter-spacing:-.01em; }
#MainMenu, footer { visibility:hidden; }

.hero { background: var(--tinta); color:#f6f3ee; border-radius:18px; padding:1.8rem 2.2rem; margin-bottom:1.2rem;
        display:flex; justify-content:space-between; align-items:flex-end; gap:2rem; flex-wrap:wrap; }
.hero h1 { color:#f6f3ee !important; font-size:2.35rem; margin:0 0 .35rem; line-height:1.1; }
.hero p { margin:0; color:#c9d6de; max-width:620px; font-size:1.02rem; }
.hero .datos { display:flex; gap:1.8rem; }
.hero .dato b { display:block; font-family:'Fraunces', serif; font-size:1.7rem; color:#fff; }
.hero .dato span { font-size:.82rem; color:#9fb4c1; }

.stTabs [data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid var(--linea); }
.stTabs [data-baseweb="tab"] { font-weight:600; padding:.55rem 1.1rem; border-radius:10px 10px 0 0; }
.stTabs [aria-selected="true"] { background:#fff; color:var(--tinta) !important; }

[class*="st-key-tarjeta"] { background:#fff; border-radius:14px; border-color:var(--linea) !important; box-shadow:0 1px 2px rgba(15,42,61,.04); }
.seccion { font-family:'Fraunces', serif; font-size:1.15rem; color:var(--tinta); margin:0 0 .1rem; }
.ayuda { color:var(--tinta2); font-size:.86rem; margin:0 0 .6rem; }

.veredicto { border-radius:14px; padding:1.1rem 1.3rem; margin-top:.2rem; }
.veredicto.alto  { background:#fbe9e5; border-left:6px solid var(--alto); }
.veredicto.medio { background:#fdf3de; border-left:6px solid var(--medio); }
.veredicto.bajo  { background:#e5f3eb; border-left:6px solid var(--bajo); }
.veredicto h3 { margin:0 0 .25rem; font-size:1.35rem; }
.veredicto p { margin:.15rem 0; color:#2b3b46; }

.factor { display:grid; grid-template-columns: 1fr 120px; gap:.6rem; align-items:center; padding:.45rem 0;
          border-bottom:1px dashed var(--linea); }
.factor:last-child { border-bottom:none; }
.factor .txt b { color:var(--tinta); font-weight:600; }
.factor .txt small { display:block; color:var(--tinta2); }
.barra { height:10px; background:#eee8df; border-radius:6px; position:relative; overflow:hidden; }
.barra::after { content:""; position:absolute; left:50%; top:-2px; bottom:-2px; width:2px; background:#b9ad9c; }
.barra i { position:absolute; top:0; bottom:0; border-radius:6px; }

.kpi { background:#fff; border:1px solid var(--linea); border-radius:14px; padding:1rem 1.2rem; }
.kpi b { font-family:'Fraunces', serif; font-size:2rem; color:var(--tinta); display:block; line-height:1.1; }
.kpi span { color:var(--tinta2); font-size:.88rem; }

.stButton button, .stDownloadButton button { border-radius:10px; font-weight:600; }
.stButton button[kind="primary"] { background:var(--tinta); border-color:var(--tinta); }
</style>
"""

if EN_STREAMLIT:
    st.set_page_config(page_title="Radar de Abandono · BankChurners", page_icon="📉", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(ESTILOS, unsafe_allow_html=True)


# ## 3. Carga del modelo
# Se carga `modelo-attrition.pkl` con el formato de clase: `[modelo, labelencoder, variables, min_max_scaler]`.
# En la app, `@st.cache_resource` hace que el modelo se cargue una sola vez; en el notebook se carga directamente.

# In[3]:


#En la app se guarda en caché; en el notebook se usa la función tal cual
cache = st.cache_resource if EN_STREAMLIT else (lambda f: f)

@cache
def cargar_modelo():
    with open("modelo-attrition.pkl", "rb") as f:
        return pickle.load(f)

modelo, labelencoder, variables, min_max_scaler = cargar_modelo()
variables = list(variables)
IDX_ABANDONO = list(labelencoder.classes_).index("Attrited Customer")
#El modelo final es XGBoost (árboles): se entrenó sin normalizar, por eso el scaler no se aplica.

if not EN_STREAMLIT:
    print("Modelo:", type(modelo).__name__)
    print("Clases:", list(labelencoder.classes_))
    print("Variables:", variables)


# ## 4. Datos de referencia
# Medianas de los clientes que se quedan (base original), para comparar, y nombres de las variables en español.

# In[4]:


#Medianas de la base original por tipo de cliente (para comparar)
MEDIANA_SE_QUEDA = {"Total_Trans_Ct": 71, "Total_Trans_Amt": 4100, "Total_Ct_Chng_Q4_Q1": 0.72, "Total_Revolving_Bal": 1364,
                    "Avg_Utilization_Ratio": 0.21, "Total_Amt_Chng_Q4_Q1": 0.74, "Total_Relationship_Count": 4,
                    "Contacts_Count_12_mon": 2, "Months_Inactive_12_mon": 2, "Credit_Limit": 4644, "Customer_Age": 46}

NOMBRES = {"Total_Trans_Ct": "Número de transacciones", "Total_Trans_Amt": "Monto transado",
           "Total_Ct_Chng_Q4_Q1": "Cambio en transacciones (Q4 vs Q1)", "Total_Revolving_Bal": "Saldo rotativo",
           "Avg_Utilization_Ratio": "Utilización del cupo", "Total_Amt_Chng_Q4_Q1": "Cambio en monto (Q4 vs Q1)",
           "Total_Relationship_Count": "Productos con el banco", "Contacts_Count_12_mon": "Contactos con el banco",
           "Months_Inactive_12_mon": "Meses inactivo", "Credit_Limit": "Cupo de crédito", "Customer_Age": "Edad"}


# ## 5. Funciones de predicción
# * `predecir`: probabilidad de abandono de uno o varios clientes.
# * `contribuciones`: aporte de cada variable a la predicción de un cliente (valores SHAP nativos de XGBoost).
# * `nivel_riesgo`: riesgo alto (≥ 60 %), medio (30–60 %) o bajo (< 30 %) con la acción recomendada.
# * `medidor`, `html_resultado` y `html_factores`: piezas visuales del resultado.
# * `perfil_a_cliente`: arma el registro del cliente con las variables en el orden del entrenamiento; calcula la utilización del cupo.

# In[5]:


def formato(var, valor):
    if var in ("Total_Trans_Amt", "Total_Revolving_Bal", "Credit_Limit"): return f"${valor:,.0f}"
    if var == "Avg_Utilization_Ratio": return f"{valor:.0%}"
    if var in ("Total_Ct_Chng_Q4_Q1", "Total_Amt_Chng_Q4_Q1"): return f"{valor:.2f}"
    return f"{valor:,.0f}"

def predecir(df):
    return modelo.predict_proba(df[variables])[:, IDX_ABANDONO]

def contribuciones(df):
    """Aporte de cada variable a la predicción (valores SHAP nativos de XGBoost), en sentido de abandono."""
    c = modelo.get_booster().predict(xgb.DMatrix(df[variables]), pred_contribs=True)[0, :-1]
    signo = -1 if IDX_ABANDONO == 0 else 1   #el modelo modela la clase 1; se orienta hacia 'abandono'
    return pd.Series(signo * c, index=variables)

def nivel_riesgo(p):
    if p >= 0.60: return "alto", "Riesgo alto", "Contactarlo esta semana con una oferta de retención personalizada."
    if p >= 0.30: return "medio", "Riesgo medio", "Incluirlo en la campaña de fidelización y vigilar su actividad mensual."
    return "bajo", "Riesgo bajo", "Cliente estable: mantener la relación y ofrecer productos adicionales."

def medidor(p):
    """Medidor semicircular en SVG."""
    color = {"alto": "#c2412d", "medio": "#d99a1e", "bajo": "#2f8a5b"}[nivel_riesgo(p)[0]]
    ang = math.pi * (1 - p)
    x, y = 150 + 110 * math.cos(ang), 150 - 110 * math.sin(ang)
    grande = 1 if p > 0.5 else 0
    texto = "&lt;1%" if p < 0.01 else (">99%" if p > 0.99 else f"{p:.0%}")
    arco = "" if p < 0.01 else f'<path d="M40 150 A110 110 0 {grande} 1 {x:.1f} {y:.1f}" fill="none" stroke="{color}" stroke-width="22" stroke-linecap="round"/>'
    return f"""
    <svg viewBox="0 0 300 175" style="width:100%;max-width:340px;display:block;margin:auto">
      <path d="M40 150 A110 110 0 0 1 260 150" fill="none" stroke="#eee8df" stroke-width="22" stroke-linecap="round"/>
      {arco}
      <text x="150" y="128" text-anchor="middle" font-family="Fraunces, Georgia, serif" font-size="46" font-weight="700" fill="#0f2a3d">{texto}</text>
      <text x="150" y="156" text-anchor="middle" font-family="Manrope, sans-serif" font-size="13" fill="#3d5566">probabilidad de abandono</text>
      <text x="40" y="172" text-anchor="middle" font-size="11" fill="#8a9aa5">0%</text>
      <text x="260" y="172" text-anchor="middle" font-size="11" fill="#8a9aa5">100%</text>
    </svg>"""

def perfil_a_cliente(d):
    """Convierte los datos capturados en un DataFrame con las variables del modelo."""
    util = min(d["rev"] / d["lim"], 0.999) if d["lim"] else 0.0
    return pd.DataFrame([{
        "Total_Trans_Ct": d["ct"], "Total_Trans_Amt": d["amt"], "Total_Ct_Chng_Q4_Q1": d["ct_chng"],
        "Total_Revolving_Bal": d["rev"], "Avg_Utilization_Ratio": util, "Total_Amt_Chng_Q4_Q1": d["amt_chng"],
        "Total_Relationship_Count": d["rel"], "Contacts_Count_12_mon": d["cont"], "Months_Inactive_12_mon": d["inact"],
        "Credit_Limit": d["lim"], "Customer_Age": d["age"],
    }])[variables]

def html_resultado(prob):
    """Medidor + nivel de riesgo + acción recomendada."""
    clase, titulo, accion = nivel_riesgo(prob)
    gauge = "".join(l.strip() for l in medidor(prob).splitlines())
    return (gauge + f'<div class="veredicto {clase}"><h3>{titulo}</h3>'
            f'<p><b>Qué hacer:</b> {accion}</p></div>')

def html_factores(cliente, n=5):
    """Las n variables que más pesaron en la predicción, comparadas con un cliente típico que se queda."""
    aportes = contribuciones(cliente)
    top = aportes.reindex(aportes.abs().sort_values(ascending=False).index)[:n]
    maximo = max(aportes.abs().max(), 1e-9)
    filas = ""
    for var, val in top.items():
        sube = val > 0
        color = "var(--alto)" if sube else "var(--bajo)"
        ancho = 50 * abs(val) / maximo
        pos = "left:50%" if sube else f"left:{50 - ancho}%"
        efecto = "↑ aumenta el riesgo" if sube else "↓ reduce el riesgo"
        filas += (f'<div class="factor"><div class="txt"><b>{NOMBRES[var]}: {formato(var, cliente[var].iloc[0])}</b>'
                  f'<small>{efecto} · típico que se queda: {formato(var, MEDIANA_SE_QUEDA[var])}</small></div>'
                  f'<div class="barra"><i style="{pos};width:{ancho}%;background:{color}"></i></div></div>')
    return filas


# ## 6. Perfiles de ejemplo
# En la app llenan el formulario con un clic; en el notebook se usan para probar el modelo.

# In[6]:


PERFILES = {
    "Cliente promedio": dict(ct=67, amt=3900, ct_chng=0.70, amt_chng=0.74, rel=4, inact=2, cont=2, age=46, lim=4550, rev=1280),
    "Cliente en riesgo": dict(ct=40, amt=2100, ct_chng=0.45, amt_chng=0.60, rel=2, inact=3, cont=4, age=47, lim=4200, rev=0),
    "Cliente leal": dict(ct=105, amt=9500, ct_chng=0.85, amt_chng=0.80, rel=5, inact=1, cont=1, age=44, lim=9000, rev=1800),
}

def cargar_perfil(nombre):
    for k, v in PERFILES[nombre].items():
        st.session_state[k] = v


# # Pruebas del modelo (solo en el notebook)
# Las siguientes celdas se ejecutan en Jupyter para verificar el modelo antes de desplegarlo.
# Cuando el archivo corre como app de Streamlit se omiten (condición `if not EN_STREAMLIT`).

# ## 7. Predicción de los perfiles de ejemplo

# In[7]:


if not EN_STREAMLIT:
    pruebas = pd.concat([perfil_a_cliente(v) for v in PERFILES.values()], ignore_index=True)
    pruebas.insert(0, "Perfil", list(PERFILES))
    pruebas["Prob_abandono"] = predecir(pruebas)
    pruebas["Predicción"] = labelencoder.inverse_transform(modelo.predict(pruebas[variables]))
    pruebas["Riesgo"] = pruebas["Prob_abandono"].apply(lambda p: nivel_riesgo(p)[1])
    display(pruebas[["Perfil", "Total_Trans_Ct", "Total_Revolving_Bal", "Total_Relationship_Count",
                     "Prob_abandono", "Predicción", "Riesgo"]].style.format({"Prob_abandono": "{:.2%}"}))


# ## 8. Resultado visual de un cliente (igual al de la app)
# Cliente en riesgo: pocas transacciones, actividad en caída, saldo rotativo en 0 y 4 contactos con el banco.

# In[8]:


if not EN_STREAMLIT:
    from IPython.display import HTML, display
    cliente = perfil_a_cliente(PERFILES["Cliente en riesgo"])
    prob = float(predecir(cliente)[0])
    display(HTML(f"""{ESTILOS}
    <div style="display:grid;grid-template-columns:1fr 1.2fr;gap:24px;max-width:900px;background:#fff;
                padding:18px;border-radius:14px;border:1px solid #e3ddd3">
      <div>{html_resultado(prob)}</div>
      <div><p class="seccion">¿Por qué este resultado?</p>{html_factores(cliente)}</div>
    </div>"""))


# ## 9. Prueba de sensibilidad
# ¿Cómo cambia el riesgo del cliente promedio si hace menos transacciones? Confirma que el modelo responde con lógica de negocio.

# In[9]:


if not EN_STREAMLIT:
    base = dict(PERFILES["Cliente promedio"])
    sensibilidad = pd.DataFrame({"Transacciones al año": range(20, 121, 10)})
    sensibilidad["Prob_abandono"] = [float(predecir(perfil_a_cliente({**base, "ct": n}))[0])
                                     for n in sensibilidad["Transacciones al año"]]
    display(alt.Chart(sensibilidad).mark_line(point=True, color="#c2412d").encode(
        x="Transacciones al año:Q", y=alt.Y("Prob_abandono:Q", axis=alt.Axis(format="%"), title="Probabilidad de abandono"))
        .properties(width=520, height=260, title="Riesgo según número de transacciones (resto igual al cliente promedio)"))


# ## 10. Evaluación de la cartera completa
# Se predice toda la base original, como en la pestaña *Evaluar una cartera* de la app.

# In[10]:


if not EN_STREAMLIT:
    import os
    if os.path.exists("BankChurners.csv"):
        cartera = pd.read_csv("BankChurners.csv")
        cartera["Prob_abandono"] = predecir(cartera)
        cartera["Riesgo"] = cartera["Prob_abandono"].apply(lambda p: nivel_riesgo(p)[1])
        print(cartera["Riesgo"].value_counts().to_string())
        acierto = ((cartera["Prob_abandono"] >= 0.5) == (cartera["Attrition_Flag"] == "Attrited Customer")).mean()
        print(f"\nExactitud sobre la base (incluye clientes de entrenamiento): {acierto:.1%}")
        display(cartera.sort_values("Prob_abandono", ascending=False)
                [["CLIENTNUM", "Attrition_Flag", "Prob_abandono", "Riesgo", "Total_Trans_Ct", "Total_Revolving_Bal"]]
                .head(10).style.format({"Prob_abandono": "{:.2%}"}))
    else:
        print("No se encontró BankChurners.csv en la carpeta; se omite esta prueba.")


# ## 11. Importancia de variables del modelo desplegado

# In[11]:


if not EN_STREAMLIT:
    imp = pd.DataFrame({"Variable": [NOMBRES[v] for v in variables], "Importancia": modelo.feature_importances_})
    display(alt.Chart(imp).mark_bar(color="#1f6f8b").encode(
        x=alt.X("Importancia:Q", axis=alt.Axis(format="%")),
        y=alt.Y("Variable:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)))
        .properties(width=480, height=300))


# # Interfaz gráfica (Streamlit)
# Las siguientes funciones construyen la aplicación web. En el notebook solo se **definen**;
# se ejecutan al correr `streamlit run app.py`.

# ## 12. Encabezado

# In[12]:


def encabezado():
    st.markdown("""
    <div class="hero">
      <div>
        <h1>Radar de abandono de tarjetas</h1>
        <p>Estime qué tan probable es que un cliente cancele su tarjeta de crédito, entienda por qué
           y decida la acción de retención. Evalúe un cliente a la vez o toda su cartera.</p>
      </div>
      <div class="datos">
        <div class="dato"><b>0.945</b><span>f1_macro en<br>clientes reales</span></div>
        <div class="dato"><b>92%</b><span>de los que se van<br>son detectados</span></div>
        <div class="dato"><b>11</b><span>variables<br>seleccionadas</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ## 13. Pestaña 1 – Evaluar un cliente
# Formulario con las 11 variables; la predicción se recalcula en cada cambio.

# In[13]:


def pestana_cliente():
    c_txt, *c_btn = st.columns([1.3, 1, 1, 1])
    c_txt.markdown("**¿No tiene datos a mano?** Cargue un ejemplo:")
    for col, nombre in zip(c_btn, PERFILES):
        col.button(nombre, on_click=cargar_perfil, args=(nombre,), use_container_width=True)

    izq, der = st.columns([1.35, 1], gap="large")

    with izq:
        with st.container(border=True, key="tarjeta_1"):
            st.markdown('<p class="seccion">💳 Uso de la tarjeta</p><p class="ayuda">Actividad en los últimos 12 meses.</p>',
                        unsafe_allow_html=True)
            a, b = st.columns(2)
            a.number_input("Número de transacciones", 10, 150, key="ct", help="Clientes que se quedan: ≈71 al año")
            b.number_input("Monto total transado (USD)", 500, 20000, step=100, key="amt")
            st.slider("¿Cómo cambió su número de transacciones del 1er al 4º trimestre?", 0.0, 2.0, step=0.05, key="ct_chng",
                      help="1.0 = igual que antes · 0.5 = hizo la mitad de transacciones · 1.5 = aumentó 50 %")
            st.slider("¿Cómo cambió el monto transado del 1er al 4º trimestre?", 0.0, 2.0, step=0.05, key="amt_chng")

        with st.container(border=True, key="tarjeta_2"):
            st.markdown('<p class="seccion">🏦 Relación con el banco</p><p class="ayuda">Vínculo y señales de insatisfacción.</p>',
                        unsafe_allow_html=True)
            a, b = st.columns(2)
            a.slider("Productos que tiene con el banco", 1, 6, key="rel")
            b.slider("Meses sin usar la tarjeta (últimos 12)", 0, 6, key="inact")
            a.slider("Veces que contactó al banco (últimos 12)", 0, 6, key="cont", help="Muchos contactos suelen indicar quejas")
            b.number_input("Edad", 18, 90, key="age")

        with st.container(border=True, key="tarjeta_3"):
            st.markdown('<p class="seccion">💰 Crédito</p><p class="ayuda">La utilización se calcula sola.</p>',
                        unsafe_allow_html=True)
            a, b, c = st.columns([1, 1, .8])
            a.number_input("Cupo de crédito (USD)", 1000, 40000, step=100, key="lim")
            b.number_input("Saldo rotativo (USD)", 0, 2600, step=50, key="rev", help="Deuda que el cliente arrastra de un mes a otro")
            util = min(st.session_state.rev / st.session_state.lim, 0.999) if st.session_state.lim else 0.0
            c.metric("Utilización del cupo", f"{util:.0%}")

    s = st.session_state
    cliente = perfil_a_cliente(dict(ct=s.ct, amt=s.amt, ct_chng=s.ct_chng, amt_chng=s.amt_chng, rel=s.rel,
                                    inact=s.inact, cont=s.cont, age=s.age, lim=s.lim, rev=s.rev))
    prob = float(predecir(cliente)[0])

    with der:
        with st.container(border=True, key="tarjeta_4"):
            st.markdown('<p class="seccion">Resultado</p><p class="ayuda">Se actualiza al instante con cada cambio.</p>',
                        unsafe_allow_html=True)
            st.markdown(html_resultado(prob), unsafe_allow_html=True)

        with st.container(border=True, key="tarjeta_5"):
            st.markdown('<p class="seccion">¿Por qué este resultado?</p>'
                        '<p class="ayuda">Las 5 variables que más pesaron para este cliente, comparadas con un cliente típico que se queda.</p>',
                        unsafe_allow_html=True)
            st.markdown(html_factores(cliente), unsafe_allow_html=True)


# ## 14. Pestaña 2 – Evaluar una cartera
# Carga un CSV (plantilla o `BankChurners.csv`), predice todos los clientes y los ordena por riesgo.

# In[14]:


def pestana_cartera():
    st.markdown("### Priorice su cartera")
    st.write("Suba un archivo CSV con sus clientes y obtenga la lista ordenada por riesgo, lista para la campaña de retención. "
             "Sirve la plantilla de abajo **o directamente el archivo original `BankChurners.csv`**.")

    plantilla = pd.DataFrame([{**{"ID_cliente": f"C-00{i+1}"},
                               **{v: PERFILES[p][k] for v, k in zip(
                                   ["Total_Trans_Ct", "Total_Trans_Amt", "Total_Ct_Chng_Q4_Q1", "Total_Amt_Chng_Q4_Q1",
                                    "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon",
                                    "Customer_Age", "Credit_Limit", "Total_Revolving_Bal"],
                                   ["ct", "amt", "ct_chng", "amt_chng", "rel", "inact", "cont", "age", "lim", "rev"])}}
                              for i, p in enumerate(PERFILES)])
    c1, c2 = st.columns([1, 2.2])
    c1.download_button("⬇️  Descargar plantilla CSV", plantilla.to_csv(index=False).encode("utf-8"),
                       "plantilla_clientes.csv", "text/csv", use_container_width=True)
    archivo = c2.file_uploader("Archivo de clientes (.csv)", type="csv", label_visibility="collapsed")

    if archivo is not None:
        try:
            df = pd.read_csv(archivo)
            if "Avg_Utilization_Ratio" not in df.columns and {"Total_Revolving_Bal", "Credit_Limit"} <= set(df.columns):
                df["Avg_Utilization_Ratio"] = (df["Total_Revolving_Bal"] / df["Credit_Limit"]).clip(0, 0.999)
            faltan = [v for v in variables if v not in df.columns]
            if faltan:
                st.error("Al archivo le faltan estas columnas: " + ", ".join(faltan) + ". Use la plantilla como guía.")
            else:
                df["Prob_abandono"] = predecir(df)
                df["Riesgo"] = df["Prob_abandono"].apply(lambda p: nivel_riesgo(p)[1])
                df = df.sort_values("Prob_abandono", ascending=False)

                n = len(df); altos = (df["Riesgo"] == "Riesgo alto").sum(); medios = (df["Riesgo"] == "Riesgo medio").sum()
                k1, k2, k3, k4 = st.columns(4)
                for col, valor, texto in [(k1, f"{n:,}", "clientes evaluados"), (k2, f"{altos:,}", "en riesgo alto"),
                                          (k3, f"{medios:,}", "en riesgo medio"), (k4, f"{altos / n:.1%}", "de la cartera a contactar ya")]:
                    col.markdown(f'<div class="kpi"><b>{valor}</b><span>{texto}</span></div>', unsafe_allow_html=True)

                st.write("")
                g1, g2 = st.columns([1, 1.6], gap="large")
                with g1:
                    conteo = df["Riesgo"].value_counts().reindex(["Riesgo alto", "Riesgo medio", "Riesgo bajo"]).fillna(0).reset_index()
                    conteo.columns = ["Riesgo", "Clientes"]
                    graf = alt.Chart(conteo).mark_bar(cornerRadiusEnd=6).encode(
                        y=alt.Y("Riesgo:N", sort=None, title=None), x=alt.X("Clientes:Q", title="Clientes"),
                        color=alt.Color("Riesgo:N", legend=None, scale=alt.Scale(
                            domain=["Riesgo alto", "Riesgo medio", "Riesgo bajo"], range=["#c2412d", "#d99a1e", "#2f8a5b"])),
                        tooltip=["Riesgo", "Clientes"]).properties(height=190, title="Clientes por nivel de riesgo")
                    st.altair_chart(graf, use_container_width=True)
                with g2:
                    if "Attrition_Flag" in df.columns:
                        acierto = ((df["Prob_abandono"] >= 0.5) == (df["Attrition_Flag"] == "Attrited Customer")).mean()
                        st.info(f"El archivo trae la respuesta real (`Attrition_Flag`): el modelo acierta en el **{acierto:.1%}** de los clientes. "
                                "Ojo: incluye clientes con los que se entrenó; la exactitud esperada en clientes nuevos es ≈97 %.")
                    st.markdown("**Cómo usar la lista:** empiece por los clientes de riesgo alto (arriba), que son los que "
                                "tienen mayor probabilidad de irse en el corto plazo. Los de riesgo medio van a campañas "
                                "de fidelización masivas.")

                id_col = [c for c in ["ID_cliente", "CLIENTNUM"] if c in df.columns]
                mostrar = id_col + ["Prob_abandono", "Riesgo", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Total_Revolving_Bal",
                                    "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon"]
                st.dataframe(df[mostrar], hide_index=True, use_container_width=True, height=380,
                             column_config={"Prob_abandono": st.column_config.ProgressColumn("Prob. abandono", format="percent", min_value=0, max_value=1),
                                            "Total_Trans_Ct": "Transacciones", "Total_Ct_Chng_Q4_Q1": "Cambio trans.",
                                            "Total_Revolving_Bal": "Saldo rotativo", "Total_Relationship_Count": "Productos",
                                            "Months_Inactive_12_mon": "Meses inactivo", "Contacts_Count_12_mon": "Contactos"})
                st.download_button("⬇️  Descargar resultados", df.to_csv(index=False).encode("utf-8"),
                                   "clientes_con_riesgo.csv", "text/csv", type="primary")
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
    else:
        st.caption("Aún no se ha subido ningún archivo.")


# ## 15. Pestaña 3 – Cómo funciona el modelo

# In[15]:


def pestana_modelo():
    a, b = st.columns([1.1, 1], gap="large")
    with a:
        st.markdown("### Cómo se construyó")
        st.markdown("""
1. **Datos:** 10.127 clientes de tarjeta de crédito; 16 % abandonó.
2. **Selección de factores:** con correlaciones e importancia de Random Forest se pasó de 19 a **11 variables**.
   Se descartaron las redundantes (`Avg_Open_To_Buy`, `Months_on_book`) y las demográficas, que no aportaban.
3. **Balanceo:** SMOTE para que la clase abandono fuera el 50 % de la mayoritaria.
4. **Validación cruzada estratificada (10 folds)** de 6 modelos: árbol, KNN, red neuronal, SVM, Random Forest y XGBoost.
5. **GridSearchCV** sobre XGBoost (48 combinaciones). Se eligió la versión con `max_depth=3`,
   igual de precisa que la más compleja pero sin overfitting.
""")
        st.markdown("### Calidad en clientes reales")
        m1, m2, m3 = st.columns(3)
        m1.markdown('<div class="kpi"><b>0.945</b><span>f1_macro</span></div>', unsafe_allow_html=True)
        m2.markdown('<div class="kpi"><b>97%</b><span>exactitud</span></div>', unsafe_allow_html=True)
        m3.markdown('<div class="kpi"><b>92%</b><span>recall abandono</span></div>', unsafe_allow_html=True)
        st.caption("Medidas de validación cruzada aplicando SMOTE solo dentro del entrenamiento de cada fold.")
    with b:
        st.markdown("### Qué mira el modelo")
        imp = pd.DataFrame({"Variable": [NOMBRES[v] for v in variables], "Importancia": modelo.feature_importances_})
        graf = alt.Chart(imp).mark_bar(cornerRadiusEnd=5, color="#1f6f8b").encode(
            x=alt.X("Importancia:Q", axis=alt.Axis(format="%"), title="Importancia en el modelo"),
            y=alt.Y("Variable:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)), tooltip=["Variable", alt.Tooltip("Importancia:Q", format=".1%")]
        ).properties(height=360)
        st.altair_chart(graf, use_container_width=True)
        st.markdown("El abandono se anticipa por el **comportamiento**, no por el perfil: pocas transacciones, "
                    "saldo rotativo en cero y pocos productos con el banco son las señales más fuertes.")


# ## 16. Ejecución de la aplicación
# Solo se ejecuta con `streamlit run app.py`.

# In[16]:


if EN_STREAMLIT:
    if "ct" not in st.session_state:
        cargar_perfil("Cliente promedio")
    encabezado()
    tab1, tab2, tab3 = st.tabs(["👤  Evaluar un cliente", "📋  Evaluar una cartera", "🧠  Cómo funciona el modelo"])
    with tab1:
        pestana_cliente()
    with tab2:
        pestana_cartera()
    with tab3:
        pestana_modelo()
else:
    print("Notebook ejecutado. Para abrir la aplicación: descargar como app.py y ejecutar  streamlit run app.py")

