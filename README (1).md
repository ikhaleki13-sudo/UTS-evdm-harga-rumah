# Eksplorasi dan Regresi Harga Rumah Jabodetabek

UTS Eksplorasi dan Visualisasi Data Multivariat (26.1), Program Studi Statistika, Universitas Akprind Indonesia.

## Identitas Pembuat
* Nama: Juliana Riska Betek Leki
* NIM: 231068006
* Program Studi: Statistika

## Isi aplikasi
Dashboard Streamlit dengan 6 tab: masalah & data, univariat (3 visualisasi), bivariat (3), multivariat (3, termasuk heatmap korelasi), regresi linear berganda (6 visualisasi termasuk diagnostik residual, uji asumsi, dan forest plot), kesimpulan & rekomendasi. Tiap grafik disertai interpretasi.

## Sumber data
Rumah123.com (hasil scraping), dataset Kaggle *Jabodetabek House Price*: [https://www.kaggle.com/datasets/nafisbarizki/daftar-harga-rumah-jabodetabek](https://www.kaggle.com/datasets/nafisbarizki/daftar-harga-rumah-jabodetabek).
Variabel dipakai: harga, luas tanah, luas bangunan, kota, sertifikat. Pembersihan data dijelaskan di tab 1.

## Menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py
