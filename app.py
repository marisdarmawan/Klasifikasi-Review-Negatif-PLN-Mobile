import streamlit as st
import pandas as pd
import time
import torch
from sentence_transformers import SentenceTransformer, util

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Smart CSO - AI Powered Classification Negative Review PLN Mobile",
    page_icon="⚡",
    layout="centered"
)

# --- LOAD MODEL AI (DI-CACHE AGAR CEPAT) ---
@st.cache_resource
def load_ai_model():
    # Menggunakan model multilingual yang paham bahasa Indonesia
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- LOAD DATA & COMPUTE EMBEDDINGS ---
@st.cache_data
def load_and_embed_data(_model):
    try:
        # Load data klasifikasi
        df_klas = pd.read_csv('Hasil_Klasifikasi_Ulasan_PLN.csv')
        df_klas['Path_Lengkap'] = df_klas['LAYER 1'].astype(str) + ' > ' + df_klas['LAYER 2'].astype(str) + ' > ' + df_klas['LAYER 3'].astype(str) + ' > ' + df_klas['LAYER 4'].astype(str) + ' > ' + df_klas['LAYER 5'].astype(str)
        
        # Load data sentimen
        df_sentimen = pd.read_csv('hasil_analisis_sentimen.csv')
        
        # Gabungkan
        df_merge = pd.merge(df_klas[['content', 'Path_Lengkap']], df_sentimen[['review', 'sentimen']], left_on='content', right_on='review', how='inner')
        
        # Bersihkan data (Buang "Tidak Deskriptif")
        df_bersih = df_merge[~df_merge['Path_Lengkap'].str.contains('Tidak Deskriptif', na=False)].dropna(subset=['content']).copy()
        
        # MENGUBAH 5.000 TEKS MENJADI VEKTOR MATEMATIKA (EMBEDDINGS)
        database_embeddings = _model.encode(df_bersih['content'].tolist(), convert_to_tensor=True)
        
        return df_bersih, database_embeddings
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

model = load_ai_model()
df_db, database_embeddings = load_and_embed_data(model)

def prediksi_cerdas(input_teks):
    if df_db is None:
        return "Error", "Error", "0%"
        
    # Ubah input user menjadi vektor
    input_vec = model.encode(input_teks, convert_to_tensor=True)
    
    # Cari kemiripan makna (Cosine Similarity)
    cosine_scores = util.cos_sim(input_vec, database_embeddings)[0]
    
    # Ambil skor tertinggi
    best_idx = torch.argmax(cosine_scores).item()
    best_score = cosine_scores[best_idx].item()
    
    kategori_prediksi = df_db.iloc[best_idx]['Path_Lengkap']
    sentimen_prediksi = df_db.iloc[best_idx]['sentimen']
    
    # Konversi skor probabilitas
    confidence = min(99.9, max(40.0, best_score * 100)) 
    
    return kategori_prediksi, sentimen_prediksi, f"{confidence:.1f}%"

# --- USER INTERFACE (UI) ---
st.title("⚡ Smart CSO - AI Powered Classification Negative Review PLN Mobile")
st.markdown("**PLN Mobile Customer Support Ticketing System (PoC)**")
st.divider()

st.subheader("📝 Masukkan Ulasan Pelanggan")
teks_input = st.text_area("Ketik atau Paste Review Negatif Pelanggan:", height=150, placeholder="Contoh: mati lampu dari jam 2 siang gak nyala-nyala woy..")

if st.button("🔍 Analisis Semantik", type="primary", use_container_width=True):
    if teks_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("🧠 AI sedang memahami makna konteks..."):
            kat_res, sent_res, conf_res = prediksi_cerdas(teks_input)
            st.success("Analisis Selesai!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="🎭 Prediksi Sentimen", value=sent_res)
            with col2:
                st.metric(label="🎯 AI Confidence Score", value=conf_res)
                
            st.info(f"**📂 Rekomendasi Kategori Tiket:**\n\n{kat_res}")
            
st.divider()
st.caption("Dikembangkan oleh Tim Data Science Divisi Manajemen Digital")
