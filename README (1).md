# 🏠 Eksplorasi & Analisis Regresi Linear Berganda: Harga Rumah Jabodetabek

Repositori ini disusun untuk memenuhi tugas **Ujian Tengah Semester (UTS)** mata kuliah **Eksplorasi dan Visualisasi Data Multivariat (26.1)**.

## 👤 Identitas Pembuat
* **Nama Lengkap**    : Juliana Riska Betek Leki
* **NIM**             : 231068006
* **Program Studi**   : Statistika

## 📋 Deskripsi Proyek
Proyek ini mengimplementasikan *dashboard* interaktif berbasis web menggunakan **Streamlit** untuk mengeksplorasi, menganalisis, dan memodelkan faktor-faktor yang memengaruhi harga penawaran rumah di wilayah Jabodetabek. Analisis ini menggunakan data sekunder dari platform **Rumah123.com** (Dataset Kaggle: *Jabodetabek House Price*).

## 🗂️ Struktur Dashboard & Fitur Aplikasi
Aplikasi interaktif ini dibagi menjadi **6 tab utama** untuk menjawab seluruh rangkaian analisis multivariat:

1. **Masalah & Data**: Latar belakang, tujuan analisis, definisi variabel, serta tahapan pembersihan data (*data cleaning*, penanganan data hilang, dan pencilan/IQR).
2. **Eksplorasi Univariat**: Statistik deskriptif lengkap dan 3 visualisasi univariat (Histogram + Boxplot untuk variabel numerik, serta diagram batang untuk variabel kategorik kota dan sertifikat).
3. **Eksplorasi Bivariat**: Analisis hubungan antar dua variabel dan 3 visualisasi bivariat (Scatter plot harga vs luas, Boxplot harga per kota, dan Boxplot harga per jenis sertifikat).
4. **Eksplorasi Multivariat**: Analisis hubungan tiga variabel atau lebih disertai 3 visualisasi multivariat (Bubble plot, Facet scatter plot per kota, dan Heatmap matriks korelasi Pearson).
5. **Regresi Linear Berganda**: Pemodelan regresi lengkap dengan 6 visualisasi diagnostik (Aktual vs Prediksi, Residual vs Fitted, Histogram residual, Q-Q Plot normalitas, Cook's Distance, dan Forest Plot Koefisien Regresi dengan CI 95%), serta uji asumsi klasik (Normalitas & Homoskedastisitas).
6. **Kesimpulan & Rekomendasi**: Ringkasan temuan utama model dan rekomendasi praktis bagi pembeli, agen, maupun peneliti lanjutan.

## 📊 Sumber Data
* **Platform**: Rumah123.com (via Kaggle *Jabodetabek House Price*)
* **Variabel Utama**: Harga penawaran (`price_in_rp`), luas tanah (`land_size_m2`), luas bangunan (`building_size_m2`), kota (`city`), dan jenis sertifikat (`certificate`).

## ⚙️ Cara Menjalankan Aplikasi Secara Lokal
Pastikan Python sudah terinstal di komputer Anda. Ikuti langkah-langkah berikut di terminal/command prompt:

1. **Clone repository ini:**
   ```bash
   git clone [https://github.com/username-kamu/nama-repo.git](https://github.com/username-kamu/nama-repo.git)
   cd nama-repo
