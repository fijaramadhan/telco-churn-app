import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model dan pendukungnya
# Gunakan try-except agar aplikasi tidak crash jika file tidak ada
try:
    model = joblib.load('model_churn_best.pkl')
    scaler = joblib.load('scaler_churn.pkl')
    features = joblib.load('feature_columns.pkl')
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

st.title("📊 Telco Customer Churn Prediction")

# Membuat form input
with st.form("prediction_form"):
    st.subheader("Data Pelanggan")
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure (Bulan)", min_value=0, max_value=100, value=1)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

    with col2:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=50.0)

    # PASTIKAN BARIS INI MASUK DALAM INDENTASI 'with st.form'
    submit = st.form_submit_button("Prediksi Sekarang")

# Logika Prediksi (DI LUAR BLOK FORM)
if submit:
    # Buat data frame agar urutan fitur sesuai dengan model
    input_df = pd.DataFrame(np.zeros((1, len(features))), columns=features)
    
    # Mapping manual (Sesuaikan dengan nama kolom asli saat training)
    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = monthly_charges
    input_df['TotalCharges'] = total_charges
    # Catatan: Untuk kolom kategori, kamu perlu melakukan encoding yang sama dengan saat training

    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]

        st.divider()
        if prediction[0] == 1:
            st.error(f"⚠️ Pelanggan diprediksi: **CHURN** (Probabilitas: {probability:.2%})")
        else:
            st.success(f"✅ Pelanggan diprediksi: **STAY** (Probabilitas Churn: {probability:.2%})")
    except Exception as e:
        st.warning(f"Kesalahan saat prediksi: {e}")