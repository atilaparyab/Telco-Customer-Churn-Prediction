import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

model_columns = model.feature_names_in_

st.set_page_config(page_title="Churn Tahmin Paneli", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #004aad;
        color: white;
        font-weight: bold;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #003380;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Müşteri Terk Analiz Paneli")
st.caption("Yapay Zeka Destekli Risk Tahmin ve Müşteri Elde Tutma Strateji Aracı")

st.markdown("---")

def kullanici_girdileri():
    st.sidebar.header("Müşteri Parametreleri")
    st.sidebar.subheader("Finansal Veriler")
    tenure = st.sidebar.slider("Müşteri Kıdemi (Ay)", 1, 72, 12)
    monthly_charges = st.sidebar.number_input("Aylık Fatura Tutarı ($)", 18.0, 120.0, 60.0)
    total_charges = st.sidebar.number_input("Toplam Fatura Tutarı ($)", 18.0, 9000.0, 500.0)
    st.sidebar.subheader("Hizmet Detayları")
    contract = st.sidebar.selectbox("Sözleşme Tipi", ["Month-to-month", "One year", "Two year"])
    internet = st.sidebar.selectbox("İnternet Servis Tipi", ["DSL", "Fiber optic", "No"])
    tech_support = st.sidebar.selectbox("Teknik Destek", ["Yes", "No", "No internet service"])
    online_security = st.sidebar.selectbox("Online Güvenlik", ["Yes", "No", "No internet service"])

    features = pd.DataFrame(0, index=[0], columns=model_columns)
    features['tenure'] = tenure
    features['MonthlyCharges'] = monthly_charges
    features['TotalCharges'] = total_charges
    features['Monthly_to_Total_Ratio'] = monthly_charges / total_charges if total_charges > 0 else 0

    if f"Contract_{contract}" in model_columns:
        features[f"Contract_{contract}"] = 1
    if f"InternetService_{internet}" in model_columns:
        features[f"InternetService_{internet}"] = 1
    if f"TechSupport_{tech_support}" in model_columns:
        features[f"TechSupport_{tech_support}"] = 1
    if f"OnlineSecurity_{online_security}" in model_columns:
        features[f"OnlineSecurity_{online_security}"] = 1
    return features

input_df = kullanici_girdileri()
st.subheader("Model Analiz Sonucu")
st.write("Müşterinin şirkette kalma veya ayrılma ihtimalini hesaplamak için tıklayın.")

if st.button("RİSK ANALİZİNİ BAŞLAT", use_container_width=True):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction[0] == 1:
        st.error(f"### ⚠️ YÜKSEK RİSK: Ayrılma İhtimali %{probability * 100:.2f}")
        st.progress(probability)
        st.markdown("""
        **Sistemin Önerdiği Aksiyon Planı:**
        * Müşteri temsilcisi tarafından **acil** arama planlayın.
        * Fatura tutarında geçici süreyle %15 indirim tanımlayın.
        * Teknik destek hizmetlerini ücretsiz teklif edin.
        """)
    else:
        st.success(f"### ✅ DÜŞÜK RİSK: Ayrılma İhtimali Sadece %{probability * 100:.2f}")
        st.progress(probability)
        st.balloons()
        st.markdown("""
        **Sistem Analizi:**
        Müşterinin profili sadık kullanıcı davranışlarıyla eşleşiyor. 
        Mevcut standart hizmet kalitesini korumaya devam edebilirsiniz.
        """)

st.subheader("Müşteri Veri Profili")
st.info("Aşağıdaki veriler arka planda makine öğrenmesi modeline gönderilmektedir.")
gosterilecek_df = input_df.T.rename(columns={0: 'Değer'}).reset_index()
gosterilecek_df.rename(columns={'index': 'Değişken / İndeks'}, inplace=True)
st.dataframe(gosterilecek_df, use_container_width=True, hide_index=True)

