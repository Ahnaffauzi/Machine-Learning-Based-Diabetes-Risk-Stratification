import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan scaler
logreg_model = joblib.load('logreg_model.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')
scaler_kmean = joblib.load('scaler_kmeans.pkl')
scaler_logistic = joblib.load('scaler_logistic.pkl')
numerical_cols = joblib.load('numerical_columns.pkl')
all_columns = joblib.load('model_columns.pkl')

cluster_stats = {
    0: {
        "label": "Kelompok Usia Muda, Risiko Metabolik Dini",
        "deskripsi": "Cluster ini ditandai dengan usia yang sangat muda, berat badan rendah, namun memiliki tingkat gula darah yang tinggi dan fungsi paru yang cenderung rendah.",
        "Age": -1.25,  # Contoh Z-score (rata-rata usia di bawah rata-rata keseluruhan)
        "Blood Pressure": -0.85 # Contoh Z-score (rata-rata tekanan darah di bawah rata-rata keseluruhan)
    },
    1: {
        "label": "Kelompok Dewasa, Risiko Metabolik Tinggi",
        "deskripsi": "Cluster ini berisi individu dewasa dengan BMI, kolesterol, tekanan darah, dan lingkar pinggang yang tinggi, menunjukkan risiko sindrom metabolik.",
        "Age": 1.10, # Contoh Z-score (rata-rata usia di atas rata-rata keseluruhan)
        "Blood Pressure": 1.50 # Contoh Z-score (rata-rata tekanan darah di atas rata-rata keseluruhan)
    }
}

# Fitur yang digunakan
feature_columns = [
    'Insulin Levels', 'Age', 'BMI', 'Physical Activity', 'Blood Pressure',
    'Cholesterol Levels', 'Waist Circumference', 'Blood Glucose Levels',
    'Socioeconomic Factors', 'Alcohol Consumption', 'Glucose Tolerance Test',
    'Weight Gain During Pregnancy', 'Pancreatic Health', 'Pulmonary Function',
    'Neurological Assessments', 'Liver Function Tests', 'Digestive Enzyme Levels',
    'Birth Weight', 'Genetic Markers_Positive', 'Autoantibodies_Positive',
    'Family History_Yes', 'Dietary Habits_Unhealthy', 'Environmental Factors_Present',
    'Ethnicity_Low Risk', 'Smoking Status_Smoker', 'History of PCOS_Yes',
    'Previous Gestational Diabetes_Yes', 'Pregnancy History_Normal',
    'Cystic Fibrosis Diagnosis_Yes', 'Steroid Use History_Yes',
    'Genetic Testing_Positive', 'Urine Test_Ketones Present', 'Urine Test_Normal',
    'Urine Test_Protein Present', 'Early Onset Symptoms_Yes'
]

numerical_cols = ['Insulin Levels', 'Age', 'BMI', 'Blood Pressure', 'Cholesterol Levels', 
                  'Waist Circumference', 'Blood Glucose Levels', 'Weight Gain During Pregnancy', 
                  'Pancreatic Health', 'Pulmonary Function', 'Neurological Assessments', 
                  'Digestive Enzyme Levels', 'Birth Weight']

# UI layout
st.set_page_config(page_title="Prediksi Kesehatan", layout="wide")
tab1, tab2, tab3 = st.tabs(["🧑‍⚕️ Tenaga Medis (Batch CSV)", "👤 Masyarakat Umum (Manual Input)", "📊 Model Performance"])

# =======================
# TAB 1 - TENAGA MEDIS
# =======================
with tab1:
    st.title("🧑‍⚕️ Upload Data CSV - Untuk Tenaga Medis")

    uploaded_file = st.file_uploader("Upload file CSV berisi data pasien", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📋 Data yang diupload", df.head())

        if st.button("🔍 Prediksi Pasien"):
            # ===============================
            # 1️⃣ PREDIKSI DIABETES - LOGREG
            # ===============================
            df_numerical = df[numerical_cols]
            df_non_numerical = df.drop(columns=numerical_cols)

            df_numerical_scaled = pd.DataFrame(
                scaler_logistic.transform(df_numerical),
                columns=numerical_cols,
                index=df.index
            )

            df_scaled_logreg = pd.concat([df_numerical_scaled, df_non_numerical], axis=1)
            df_scaled_logreg = df_scaled_logreg[all_columns]

            diabetes_pred = logreg_model.predict(df_scaled_logreg)

            try:
                probs = logreg_model.predict_proba(df_scaled_logreg)
                prob_df = pd.DataFrame(probs, columns=[f"Probabilitas_Diabetes_{i}" for i in range(probs.shape[1])])
            except:
                prob_df = pd.DataFrame()

            # ===============================
            # 2️⃣ CLUSTERING DIABETES - KMEANS
            # ===============================
            kmeans_features = ['Insulin Levels', 'Age', 'BMI', 'Blood Pressure',
                            'Cholesterol Levels', 'Waist Circumference', 'Blood Glucose Levels',
                            'Weight Gain During Pregnancy', 'Pulmonary Function',
                            'Digestive Enzyme Levels', 'Birth Weight']

            df_kmeans_input = df[kmeans_features]
            df_kmeans_scaled = scaler_kmean.transform(df_kmeans_input)
            cluster_labels = kmeans_model.predict(df_kmeans_scaled)

            # Gabung hasil prediksi
            df_result = df.copy()
            df_result["Prediksi_Diabetes"] = diabetes_pred
            if not prob_df.empty:
                df_result = pd.concat([df_result, prob_df], axis=1)
            df_result["Cluster"] = cluster_labels

            st.write("### 🔎 Hasil Prediksi Pasien:")

            for idx, row in df_result.iterrows():
                diabetes = "Berpotensi Diabetes" if row["Prediksi_Diabetes"] == 1 else "Berpotensi Tidak Diabetes"
                cluster = int(row["Cluster"])

                # Probabilitas jika tersedia
                if "Probabilitas_Diabetes_1" in row:
                    prob = round(row["Probabilitas_Diabetes_1"] * 100, 2)
                    prob_text = f"Probabilitas terkena diabetes: **{prob}%**"
                else:
                    prob_text = "Probabilitas tidak tersedia"

                # Insight Cluster
                if cluster == 0:
                    cluster_info = """
                    **Cluster 0** → *Kelompok Usia Muda, Risiko Metabolik Dini*:
                    - Usia muda, berat badan rendah
                    - Gula darah tinggi, fungsi paru rendah
                    """
                elif cluster == 1:
                    cluster_info = """
                    **Cluster 1** → *Kelompok Dewasa, Risiko Metabolik Tinggi*:
                    - Usia dewasa, BMI dan kolesterol tinggi
                    - Tekanan darah dan lingkar pinggang besar
                    """

                # Tampilkan hasil untuk setiap pasien
                st.markdown(f"""
                ---
                ### 🧍 Pasien {idx+1}
                - Prediksi: **{diabetes}**
                - {prob_text}
                - Termasuk: {cluster_info}
                """)


# =======================
# TAB 2 - MASYARAKAT UMUM
# =======================
with tab2:
    st.title("👤 Input Manual - Untuk Masyarakat Umum")
    st.write("Silakan isi data numerik berikut:")

    insulin = st.number_input(
    "Tingkat Insulin (0-300 μU/mL)", 
    help="Insulin membantu mengatur kadar gula darah. Nilai normal biasanya antara 2–25 μU/mL.",
    min_value=0, max_value=300, value=100
    )

    age = st.number_input(
        "Umur Anda (tahun)", 
        help="Masukkan umur Anda dalam tahun.",
        min_value=1, max_value=100, value=30
    )

    bmi = st.number_input(
        "Indeks Massa Tubuh (BMI)", 
        help="BMI adalah indikator berat badan ideal berdasarkan tinggi badan. Rentang normal: 18.5–24.9",
        min_value=10, max_value=60, value=22
    )

    bp = st.number_input(
        "Tekanan Darah (mmHg)", 
        help="Masukkan tekanan darah sistolik (angka atas). Normal: sekitar 120 mmHg.",
        min_value=60, max_value=180, value=120
    )

    chol = st.number_input(
        "Kadar Kolesterol Total (mg/dL)", 
        help="Kolesterol tinggi bisa meningkatkan risiko penyakit jantung. Normal: < 200 mg/dL.",
        min_value=100, max_value=400, value=200
    )

    waist = st.number_input(
        "Lingkar Pinggang (cm)", 
        help="Lingkar pinggang tinggi bisa menunjukkan risiko penyakit metabolik.",
        min_value=50, max_value=150, value=85
    )

    glucose = st.number_input(
        "Tingkat Gula Darah (mg/dL)", 
        help="Masukkan kadar gula darah puasa Anda. Normal: 70–100 mg/dL.",
        min_value=50, max_value=300, value=100
    )

    weight_gain = st.number_input(
        "Penambahan Berat Badan Saat Hamil (kg)", 
        help="Masukkan 0 jika Anda tidak sedang hamil. Kenaikan normal saat hamil: 11–16 kg.",
        min_value=0, max_value=30, value=10
    )

    pulmonary = st.number_input(
        "Fungsi Paru-Paru (skor PFT)", 
        help="Didapat dari Pulmonary Function Test. Skor 100 dianggap normal.",
        min_value=50, max_value=150, value=90
    )

    enzyme = st.number_input(
        "Tingkat Enzim Pencernaan", 
        help="Menunjukkan seberapa baik tubuh mencerna makanan. Skor 50–100 biasanya menunjukkan fungsi baik.",
        min_value=0, max_value=100, value=50
    )

    birth_weight = st.number_input(
        "Berat Badan Saat Lahir (gram)", 
        help="Normal: 2500–4000 gram. Berat lahir bisa memengaruhi risiko penyakit metabolik di masa depan.",
        min_value=1000, max_value=5000, value=3000
    )



    input_vector = np.array([
        insulin, age, bmi, bp, chol,
        waist, glucose, weight_gain,
        pulmonary, enzyme, birth_weight
    ]).reshape(1, -1)

    scaled_input = scaler_kmean.transform(input_vector)

    if st.button("🔍 Prediksi"):
        cluster = kmeans_model.predict(scaled_input)[0]
        st.success(f"Hasil Prediksi: Anda masuk ke **Cluster {cluster}**")

        if cluster == 0:
            st.markdown("""
            ### 🧬 Karakteristik Cluster 0 - *Kelompok Usia Muda, Risiko Metabolik Dini*
            - Umumnya Berusia **sangat muda (anak-anak/remaja)**.
            - Umumnya **Berat badan dan BMI rendah**, cenderung kurus.
            - Umumnya **Gula darah sangat tinggi**.
            - Umumnya **Fungsi paru dan enzim pencernaan rendah**.
            - Umumnya **Berat lahir rendah**, underweight saat kelahiran.

            ⚠️ **Saran**: Lakukan pemeriksaan lanjutan untuk gula darah dan metabolisme, terutama jika pasien masih anak-anak.
            """)
        elif cluster == 1:
            st.markdown("""
            ### 🧬 Karakteristik Cluster 1 - *Kelompok Dewasa, Risiko Metabolik Tinggi*
            - Umumnya Berusia **dewasa** dengan kecenderungan overweight (*BMI tinggi*).
            - Umumnya **Tekanan darah, kolesterol, dan lingkar pinggang tinggi** → berisiko *sindrom metabolik*.
            - Umumnya **Gula darah tinggi**, walau tidak setinggi Cluster 0.
            - Umumnya **Berat lahir normal**, fungsi paru dan pencernaan lebih baik.

            ⚠️ **Saran**: Jaga pola makan, olahraga teratur, dan lakukan cek rutin kolesterol/gula darah untuk mencegah komplikasi.
            """)
with tab3:
    st.subheader("Performance Model")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Logistic Regression**")
        st.metric("Accuracy", "93%")
        st.metric("Precision", "91%")
        st.metric("Recall", "95%")

    with col2:
        st.markdown("**K-Means Clustering**")
        st.metric("Silhouette Score", "0.40")
        st.metric("Jumlah Cluster", "2")

        st.markdown("**Karakteristik Cluster**")
        selected_cluster = st.selectbox("Pilih Cluster", [0, 1, 2])
        st.write(f"Label: {cluster_stats[selected_cluster]['label']}")
        st.write(cluster_stats[selected_cluster]['deskripsi'])
        st.write("Parameter Z-Score:")
        st.write(f"- Usia: {cluster_stats[selected_cluster]['Age']:.2f}")
        st.write(f"- Tekanan Darah: {cluster_stats[selected_cluster]['Blood Pressure']:.2f}")
