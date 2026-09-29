#Despliegue del modelo de abandono de clientes (BankChurners) con Streamlit
#Ejecutar con:  streamlit run app.py
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Riesgo de abandono · BankChurners", page_icon="💳", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Serif:wght@600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Serif', serif !important; color: #12324a; }
.block-container { padding-top: 2rem; max-width: 1150px; }
.resultado { border-radius: 10px; padding: 1.4rem 1.6rem; border-left: 8px solid; }
.alto { background: #fdecea; border-color: #c0392b; }
.medio { background: #fff4e0; border-color: #d68910; }
.bajo { background: #e8f5ee; border-color: #1e8449; }
.resultado .prob { font-size: 3rem; font-weight: 600; line-height: 1; margin: .2rem 0 .5rem; }
.resultado p { margin: .25rem 0; color: #333; }
button[kind="primaryFormSubmit"], .stFormSubmitButton button { background-color:#12324a !important; border-color:#12324a !important; color:#fff !important; }
</style>
""", unsafe_allow_html=True)

#Se carga el modelo (mismo formato de clase: modelo, labelencoder, variables, normalizador)
@st.cache_resource
def cargar_modelo():
    with open("modelo-attrition.pkl", "rb") as f:
        return pickle.load(f)

modelo, labelencoder, variables, min_max_scaler = cargar_modelo()
#El modelo final es XGBoost (árboles), se entrenó sin normalizar: el scaler no se aplica.
clase_abandono = list(labelencoder.classes_).index("Attrited Customer")

st.title("¿Este cliente va a cancelar su tarjeta?")
st.write("Modelo XGBoost hiperparametrizado con GridSearchCV · f1_macro ≈ 0.95 en clientes reales. "
         "Ingrese el comportamiento del cliente en los últimos 12 meses.")

with st.form("cliente"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Actividad")
        trans_ct = st.number_input("Número de transacciones (12 meses)", 10, 139, 67)
        trans_amt = st.number_input("Monto total transado (USD)", 500, 18500, 3899, step=100)
        ct_chng = st.number_input("Cambio en # transacciones Q4 vs Q1", 0.0, 3.8, 0.70, step=0.05,
                                  help="1.0 = igual actividad; menor a 1 = la actividad cayó")
        amt_chng = st.number_input("Cambio en monto Q4 vs Q1", 0.0, 3.4, 0.74, step=0.05)
    with c2:
        st.subheader("Relación con el banco")
        relationship = st.slider("Productos que tiene con el banco", 1, 6, 4)
        inactive = st.slider("Meses inactivo (últimos 12)", 0, 6, 2)
        contacts = st.slider("Contactos con el banco (últimos 12)", 0, 6, 2)
        age = st.number_input("Edad del cliente", 26, 73, 46)
    with c3:
        st.subheader("Crédito")
        credit_limit = st.number_input("Cupo de crédito (USD)", 1400, 34600, 4549, step=100)
        revolving = st.number_input("Saldo rotativo (USD)", 0, 2517, 1276, step=50)
        utilization = min(revolving / credit_limit, 0.999) if credit_limit else 0.0
        st.metric("Utilización del cupo", f"{utilization:.1%}", help="Saldo rotativo / cupo (se calcula automáticamente)")
    enviar = st.form_submit_button("Calcular riesgo de abandono", type="primary", use_container_width=True)

if enviar:
    cliente = pd.DataFrame([{
        "Total_Trans_Ct": trans_ct, "Total_Trans_Amt": trans_amt, "Total_Ct_Chng_Q4_Q1": ct_chng,
        "Total_Revolving_Bal": revolving, "Avg_Utilization_Ratio": utilization, "Total_Amt_Chng_Q4_Q1": amt_chng,
        "Total_Relationship_Count": relationship, "Contacts_Count_12_mon": contacts,
        "Months_Inactive_12_mon": inactive, "Credit_Limit": credit_limit, "Customer_Age": age,
    }])[list(variables)]  #mismo orden de variables del entrenamiento

    prob = float(modelo.predict_proba(cliente)[0, clase_abandono])
    prediccion = labelencoder.inverse_transform(modelo.predict(cliente))[0]

    if prob >= 0.6:
        nivel, css, accion = "Riesgo alto", "alto", "Contactar esta semana con una oferta de retención."
    elif prob >= 0.3:
        nivel, css, accion = "Riesgo medio", "medio", "Incluir en campaña de fidelización y monitorear su actividad."
    else:
        nivel, css, accion = "Riesgo bajo", "bajo", "Sin acción especial; mantener la relación actual."

    izq, der = st.columns([1, 1.3])
    with izq:
        st.markdown(f"""
        <div class="resultado {css}">
          <p><b>{nivel}</b></p>
          <div class="prob">{prob:.1%}</div>
          <p>Probabilidad de abandono</p>
          <p>Predicción del modelo: <b>{prediccion}</b></p>
          <p>{accion}</p>
        </div>""", unsafe_allow_html=True)
    with der:
        st.markdown("**Factores a revisar** (según la selección de factores del proyecto)")
        alertas = []
        if trans_ct < 55: alertas.append(f"Pocas transacciones ({trans_ct}); la mediana de la base es 67.")
        if ct_chng < 0.6: alertas.append(f"Su actividad cayó entre Q1 y Q4 (índice {ct_chng:.2f}).")
        if revolving < 500: alertas.append("Saldo rotativo muy bajo: casi no usa la tarjeta.")
        if relationship <= 2: alertas.append(f"Solo tiene {relationship} producto(s) con el banco.")
        if inactive >= 3: alertas.append(f"{inactive} meses inactivo en el último año.")
        if contacts >= 4: alertas.append(f"{contacts} contactos con el banco: posibles quejas.")
        for a in alertas or ["Ninguna señal de alerta en las variables principales."]:
            st.write("• " + a)
