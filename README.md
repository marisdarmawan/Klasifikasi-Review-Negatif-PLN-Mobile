# ⚡ AI Review Analyzer - PLN Mobile (PoC)

Aplikasi berbasis web ini adalah *Proof of Concept* (PoC) untuk sistem *ticketing customer support* PLN Mobile. Dibangun dengan pendekatan *Natural Language Processing* (NLP), aplikasi ini bertujuan untuk mengotomatisasi klasifikasi keluhan pelanggan dan mendeteksi sentimen secara instan.

## 🎯 Fitur Utama
* **Smart Ticket Categorization:** Memprediksi kategori pengaduan hingga 5 layer kedalaman (Layer 1 hingga Layer 5) berdasarkan pola kalimat pelanggan.
* **Sentiment Analysis:** Mendeteksi apakah ulasan pelanggan bernada positif, negatif, atau netral.
* **AI Confidence Score:** Menampilkan tingkat keyakinan (probabilitas) dari model AI terhadap hasil prediksinya.
* **Interactive UI:** Dibangun menggunakan antarmuka Streamlit yang bersih dan *user-friendly*.

## 🛠️ Tech Stack
* **Bahasa Pemrograman:** Python
* **Web Framework:** Streamlit
* **Data Processing:** Pandas
* **Machine Learning:** Scikit-Learn (TF-IDF Vectorizer & Cosine Similarity)

## 🚀 Cara Menjalankan Aplikasi Secara Lokal

1. **Clone repositori ini:**
   ```bash
   git clone [https://github.com/username-anda/pln-review-analyzer.git](https://github.com/username-anda/pln-review-analyzer.git)
   cd pln-review-analyzer
