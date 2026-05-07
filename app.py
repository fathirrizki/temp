import streamlit as st
import pandas as pd
import joblib

# Load Model
# Menggunakan cache agar model tidak di-load ulang setiap kali user berinteraksi
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

model = load_model()

# Judul Aplikasi
st.title("🚢Titanic Survival Predictor", text_alignment='center')
st.markdown("Masukkan data penumpang di bawah ini untuk melihat prediksi keselamatan.", text_alignment='center')
st.divider()

# Form Input User
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Kelas Tiket (Pclass)", [1, 2, 3], help="1=Atas, 2=Menengah, 3=Bawah")
        sex = st.selectbox("Jenis Kelamin", ["male", "female"])
        sibsp = st.number_input("Jumlah Saudara/Pasangan (SibSp)", 0, 10, 0)

    with col2:
        parch = st.number_input("Jumlah Orang Tua/Anak (Parch)", 0, 10, 0)
        fare = st.number_input("Tarif Tiket (Fare)", 0.0, 600.0, 32.0, 0.50)
        embarked = st.selectbox("Pelabuhan Keberangkatan", ["S", "C", "Q"])

    submit = st.form_submit_button("Prediksi")

# Logika Inferensi
if submit:
    # Buat DataFrame sesuai format input saat training di notebook
    input_data = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }])

    # Jalankan Prediksi
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Tampilkan Hasil
    st.divider()
    if prediction == 1:
        st.success(f"**Prediksi: Selamat!** 🎉 (Peluang: {probability:.2%})")
        st.image('assets/yey.jfif', width='stretch')
    else:
        st.error(f"**Prediksi: Tidak Selamat** 😔 (Peluang: {probability:.2%})")
        st.image('assets/died.png', width='stretch')