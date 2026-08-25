%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Dashboard ML Longitudinal Matematika", layout="wide")

st.title("📊 Dashboard Prediktif & Analisis Longitudinal Matematika")
st.markdown("Aplikasi Machine Learning berbasis Streamlit untuk memprediksi capaian belajar siswa dan klasterisasi profil akademik.")

# 1. LOAD DATA
@st.cache_data
def load_data():
    file_path = "Dashboard_Data_Longitudinal_Matematika.xlsx"
    df = pd.read_excel(file_path, sheet_name="Data_Longitudinal")
    return df

df = load_data()

# Sidebar Navigasi
st.sidebar.header("Navigasi Menu")
menu = st.sidebar.selectbox("Pilih Analisis:", [
    "1. Overview Data", 
    "2. Model Prediktif (Klasifikasi)", 
    "3. Klasterisasi Siswa (K-Means)"
])

# Fitur Prediktor
features = [
    'rata_akademik_t', 'rata_tf_t', 'rata_uk_t', 'rata_praktik_t', 
    'nilai_min_t', 'nilai_maks_t', 'stabilitas_std_t', 'jumlah_nilai_bawah_75_t', 
    'n_akademik_terisi_t', 'kehadiran_t', 'tugas_terlambat_t', 
    'partisipasi_kelas_t', 'aktivitas_lms_t'
]

if menu == "1. Overview Data":
    st.subheader("Eksplorasi Data Longitudinal")
    st.write(f"Total Baris Data: {df.shape[0]} | Total Kolom: {df.shape[1]}")
    st.dataframe(df.head(10))
    
    st.markdown("### Distribusi Kategori Target (Periode t+1)")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.countplot(data=df, x='kategori_target_t1', order=['Rendah', 'Sedang', 'Tinggi'], palette='viridis', ax=ax)
    st.pyplot(fig)

elif menu == "2. Model Prediktif (Klasifikasi)":
    st.subheader("🤖 Prediksi Kategori Capaian Siswa (Random Forest)")
    
    X = df[features]
    y = df['kategori_target_t1']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="Akurasi Model pada Data Uji", value=f"{acc*100:.2f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred, labels=['Rendah', 'Sedang', 'Tinggi'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Rendah', 'Sedang', 'Tinggi'], yticklabels=['Rendah', 'Sedang', 'Tinggi'], ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)
        
    with col2:
        st.markdown("#### Feature Importance (Pengaruh Fitur)")
        fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        fi.plot(kind='barh', color='teal', ax=ax)
        ax.set_xlabel("Tingkat Kepentingan")
        st.pyplot(fig)

elif menu == "3. Klasterisasi Siswa (K-Means)":
    st.subheader("👥 Klasterisasi Profil Siswa Berdasarkan Perilaku & Akademik")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    n_clusters = st.sidebar.slider("Jumlah Klaster (K)", min_value=2, max_value=5, value=3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    st.markdown(f"#### Karakteristik Rata-rata Tiap Klaster (K = {n_clusters})")
    cluster_summary = df.groupby('cluster')[features + ['target_nilai_t1']].mean()
    st.dataframe(cluster_summary.style.background_cmap('Blues'))
    
    st.markdown("#### Visualisasi Sebaran Klaster")
    x_axis = st.selectbox("Pilih Sumbu X", features, index=0)
    y_axis = st.selectbox("Pilih Sumbu Y", features, index=9)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='cluster', palette='Set1', alpha=0.8, ax=ax)
    st.pyplot(fig)