# Eksplorasi dan Regresi Harga Rumah Jabodetabek

UTS Eksplorasi dan Visualisasi Data Multivariat (26.1), Program Studi Statistika, Universitas Akprind Indonesia.

## Isi aplikasi
Dashboard Streamlit dengan 6 tab: masalah & data, univariat (3 visualisasi), bivariat (3), multivariat (3, termasuk heatmap korelasi), regresi linear berganda (5 visualisasi termasuk diagnostik residual dan uji asumsi), kesimpulan & rekomendasi. Tiap grafik disertai interpretasi.

## Sumber data
Rumah123.com (hasil scraping), dataset Kaggle *Jabodetabek House Price*: **[isi link Kaggle di sini]**.
Variabel dipakai: harga, luas tanah, luas bangunan, kota, sertifikat. Pembersihan data dijelaskan di tab 1.

## Menjalankan
```
pip install -r requirements.txt
streamlit run app.py
```
Aplikasi online: **[isi link Streamlit Cloud di sini]**

## Berkas
`app.py` (data sudah tertanam di dalamnya), `requirements.txt`, `jabodetabek_house_price.csv` (salinan data asli, opsional)
