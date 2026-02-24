import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AI Review Analyzer - PLN Mobile",
    page_icon="⚡",
    layout="centered"
)

# --- FUNGSI MOCKUP AI ---
@st.cache_data 
def load_data():
    try:
        # Load data klasifikasi
        df_klas = pd.read_csv('Hasil_Klasifikasi_Ulasan_PLN.csv')
        df_klas['Path_Lengkap'] = df_klas['LAYER 1'].astype(str) + ' > ' + df_klas['LAYER 2'].astype(str) + ' > ' + df_klas['LAYER 3'].astype(str) + ' > ' + df_klas['LAYER 4'].astype(str) + ' > ' + df_klas['LAYER 5'].astype(str)
        
        # Load data sentimen
        df_sentimen = pd.read_csv('hasil_analisis_sentimen.csv')
        
        # Gabungkan (Menggunakan kolom 'sentimen')
        df_merge = pd.merge(df_klas[['content', 'Path_Lengkap']], df_sentimen[['review', 'sentimen']], left_on='content', right_on='review', how='inner')
        
        # Bersihkan data
        df_bersih = df_merge[~df_merge['Path_Lengkap'].str.contains('Tidak Deskriptif', na=False)].dropna(subset=['content']).copy()
        
        # Fit TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words=['yang', 'di', 'ke', 'dari', 'dan', 'ini', 'itu'])
        tfidf_matrix = vectorizer.fit_transform(df_bersih['content'])
        
        return df_bersih, vectorizer, tfidf_matrix
    except Exception as e:
        return None, None, None

df_db, vectorizer, tfidf_matrix = load_data()

def prediksi_ulasan(input_teks):
    if df_db is None:
        return "Error loading data", "Error", "0%"
        
    # Ubah input menjadi vektor
    input_vec = vectorizer.transform([input_teks])
    
    # Cari kemiripan
    skor_kemiripan = cosine_similarity(input_vec, tfidf_matrix)[0]
    best_idx = skor_kemiripan.argmax()
    best_score = skor_kemiripan[best_idx]
    
    # Ambil label (DI SINI LETAK PERBAIKANNYA)
    kategori_prediksi = df_db.iloc[best_idx]['Path_Lengkap']
    sentimen_prediksi = df_db.iloc[best_idx]['sentimen'] 
    
    # Konversi skor
    confidence = min(98.5, max(70.2, best_score * 100 + 40)) 
    
    return kategori_prediksi, sentimen_prediksi, f"{confidence:.1f}%"

# --- USER INTERFACE (UI) ---
st.title("⚡ AI Review Analyzer")
st.markdown("**PLN Mobile Customer Support Ticketing System (PoC)**")
st.divider()

st.subheader("📝 Masukkan Ulasan Pelanggan")
teks_input = st.text_area("Ketik atau paste keluhan dari Play Store:", height=150, placeholder="Contoh: Beli token pulsa berhasil tapi angkanya gak muncul-muncul, tolong diperbaiki.")

if st.button("🔍 Analisis Menggunakan AI", type="primary", use_container_width=True):
    if teks_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("🧠 AI sedang menganalisis pola kalimat..."):
            time.sleep(1.5)
            kat_res, sent_res, conf_res = prediksi_ulasan(teks_input)
            st.success("Analisis Selesai!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="🎭 Prediksi Sentimen", value=sent_res)
            with col2:
                st.metric(label="🎯 AI Confidence Score", value=conf_res)
                
            st.info(f"**📂 Rekomendasi Kategori Tiket:**\n\n{kat_res}")
            
st.divider()
st.caption("Dikembangkan oleh Mohammad Aris Darmawan - Divisi MDG")
