# DiabeteSense: Prediksi Risiko Metabolik & Klasifikasi Diabetes

Project ini menggunakan Machine Learning untuk menganalisa dan memprediksi risiko penyakit diabetes dan mengelompokkan risiko metabolik berdasarkan data medis dari pasien. Platform ini menyediakan antarmuka *(dashboard)* yang berbasis web interaktif.

## Fitur Utama

Dashboard ini memiliki 3 Tab utama:
1. **🧑‍⚕️ Tenaga Medis (Batch CSV):** Fitur bagi tenaga profesional untuk mengupload data pasien dalam format CSV dan melakukan prediksi massal.
2. **👤 Masyarakat Umum (Manual Input):** Fitur untuk mengecek risiko diabetes/metabolik dengan mengisikan indikator kesehatan secara manual.
3. **📊 Model Performance:** Fitur untuk meninjau secara transparan tingkat keakuratan dari model Machine Learning yang telah dilatih.

## Model Machine Learning yang Digunakan

Proyek ini tidak hanya memprediksi 1 hasil, tetapi memberikan analisis komprehensif menggunakan 2 pendekatan model:

1. **Classification (Klasifikasi) dengan Logistic Regression:**
   - **Tujuan:** Memprediksi apakah seseorang memiliki potensi diabetes berdasarkan nilai metrik klinis mereka.
   - **Model File:** `logreg_model.pkl`

2. **Clustering (Pengelompokan) dengan K-Means:**
   - **Tujuan:** Mengelompokkan individu ke dalam 2 *cluster* risiko (Kelompok Usia Muda dengan Risiko Dini, dan Kelompok Dewasa dengan Risiko Tinggi) untuk memberikan wawasan tambahan terkait status kesehatan yang di luar prediksi biner sederhana (seperti penyakit penyerta).
   - **Model File:** `kmeans_model.pkl`

## Cara Menjalankan Project (Setup Instalasi)

Untuk dapat menjalankan proyek ini secara lokal, pastikan **Python** telah terpasang di komputer Anda. Disarankan menggunakan *Virtual Environment* agar dependensi antar proyek tetap bersih.

### 1. Clone & Buka Folder Repository
```bash
git clone <URL_GITHUB_ANDA>
cd <NAMA_FOLDER_CLONE>
```

### 2. Membuat & Mengaktifkan Virtual Environment
Buka terminal dan jalankan perintah:

**Windows Command Prompt:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**MacOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Menginstal Dependensi (Libraries)
Lakukan instruksi untuk meng-*install* semua *framework* dan modul ML melalui perintah pip:
```bash
pip install -r requirements.txt
```

### 4. Menjalankan Dashboard Web
Setelah proses instalasi selesai, kita jalankan aplikasi *dashboard*-nya dengan Streamlit:
```bash
streamlit run app.py
```
Aplikasi secara otomatis akan terbuka melalui `http://localhost:8501/` di browser web Anda!

## File Pendukung di Dalam Repository
* `app.py`: File program utama yang menjalankan Dashboard Web via Streamlit.
* `model_training.ipynb`: Jupyter Notebook yang berisi proses utuh *Data Preparation*, *Exploratory Data Analysis (EDA)*, *Model Training*, hingga ekspor _file-file_ `.pkl`.
* `*.pkl`: Berisi *scaler* data (`scaler_kmeans.pkl`, `scaler_logistic.pkl`), *array* struktur fitur input (`numerical_columns.pkl`, `model_columns.pkl`), beserta *metrics* (evaluasi akurasi).
* `diabetes_dataset00.csv`: Dataset contoh yang dapat digunakan untuk melakukan percobaan fitur prediksi pada Tab 1 (Tenaga Medis).
