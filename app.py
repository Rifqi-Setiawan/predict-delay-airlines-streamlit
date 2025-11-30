import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

# --- SETUP HALAMAN ---
st.set_page_config(page_title="Prediksi Delay Penerbangan", layout="centered")

st.title("✈️ Aplikasi Prediksi Delay Penerbangan")
st.write("Masukkan detail penerbangan untuk memprediksi apakah akan terjadi delay.")

# --- LOAD MODEL DAN KOLOM ---
@st.cache_resource
def load_assets():
    try:
        # Load Model
        model_xgb = pickle.load(open('Tuned_Best_XGBoost.pkl', 'rb'))
        
        # Load Daftar Kolom (yang dibuat di Langkah 1)
        # Ini PENTING agar urutan kolom input sama persis dengan saat training
        model_cols = pickle.load(open('model_columns.pkl', 'rb'))
        return model_xgb, model_cols
    except FileNotFoundError:
        st.error("⚠️ File model atau model_columns.pkl tidak ditemukan.")
        st.stop()

model, model_columns = load_assets()

# --- INPUT USER ---
# Kita buat form agar terlihat rapi
with st.form(key='flight_data_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        # Input Numerik (Sesuai dataset kamu: Flight, Time, Length)
        flight_num = st.number_input("Flight Number (Nomor Penerbangan)", min_value=1, step=1, value=100)
        time_mins = st.number_input("Time (Waktu Keberangkatan dalam menit)", min_value=0, step=10, help="Contoh: Jam 10:30 = 630 menit")
        length_mins = st.number_input("Length (Durasi Penerbangan dalam menit)", min_value=0, step=10)
        
    with col2:
        # Input Kategori (Airline, AirportFrom, AirportTo, DayOfWeek)
        # Note: Idealnya opsi selectbox diambil dari unique values dataset asli.
        # Di sini saya tulis manual beberapa contoh umum, sesuaikan jika perlu.
        
        airline = st.selectbox("Airline", ['CO', 'US', 'AA', 'AS', 'DL', 'B6', 'HA', 'OO', 'OH', 'EV', 'XE', 'UA', 'MQ', 'WN', 'F9', 'YV', '9E'])
        
        # DayOfWeek biasanya 1-7. Sesuaikan labelnya jika di data kamu pakai nama hari.
        day_of_week = st.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7], format_func=lambda x: f"Hari ke-{x}")

        # Input Bandara (Bisa Text Input atau Selectbox)
        airport_from = st.text_input("Airport From (Kode Bandara Asal, e.g., JFK, LAX)", value="ATL").upper()
        airport_to = st.text_input("Airport To (Kode Bandara Tujuan, e.g., SFO, ORD)", value="SFO").upper()

    submit_button = st.form_submit_button(label='🔍 Prediksi Delay')

# --- LOGIKA PREDIKSI ---
if submit_button:
    # 1. Buat DataFrame dari input user (Data Mentah)
    input_data = pd.DataFrame({
        'Flight': [flight_num],
        'Time': [time_mins],
        'Length': [length_mins],
        'Airline': [airline],
        'AirportFrom': [airport_from],
        'AirportTo': [airport_to],
        'DayOfWeek': [day_of_week]
    })
    
    st.write("Data Input:", input_data)

    # 2. PREPROCESSING (One-Hot Encoding)
    # Lakukan get_dummies pada data input
    input_encoded = pd.get_dummies(input_data)

    # 3. ALIGNMENT (Penyelarasan Kolom) - INI BAGIAN KUNCI
    # Kita memaksa dataframe input memiliki kolom yang SAMA PERSIS dengan model
    # Jika ada kolom kurang (karena One-Hot), diisi 0.
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Tampilkan preview data yang akan masuk ke model (untuk debugging)
    # st.write("Data Siap Prediksi (Encoded):", input_final)

    # 4. PREDIKSI
    try:
        prediction = model.predict(input_final)
        probability = model.predict_proba(input_final)
        
        # Hasilnya biasanya 0 (Tidak Delay) atau 1 (Delay)
        if prediction[0] == 1:
            st.error(f"🚨 Hasil Prediksi: TERLAMBAT (Delay) \n\nProbabilitas: {probability[0][1]*100:.2f}%")
        else:
            st.success(f"✅ Hasil Prediksi: TEPAT WAKTU \n\nProbabilitas Tepat Waktu: {probability[0][0]*100:.2f}%")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        st.info("Tips: Pastikan Kode Bandara yang dimasukkan valid dan sesuai dengan data training.")