# DASHBOARD UTS EKSPLORASI & VISUALISASI DATA MULTIVARIAT
# Topik : Harga Rumah Jabodetabek (data iklan Rumah123)
# Alur  : data, univariat, bivariat, multivariat, regresi, kesimpulan

# 0. IMPORT LIBRARY
import numpy as np                      # hitungan numerik (log, exp, akar, dll)
import pandas as pd                     # mengolah tabel data (DataFrame)
import plotly.express as px             # grafik interaktif cepat (histogram, scatter, box, dll)
import plotly.graph_objects as go       # menambah garis/elemen khusus pada grafik
import streamlit as st                  # membuat tampilan aplikasi web
from scipy import stats                 # uji normalitas (Shapiro, KS) dan Q-Q plot
import statsmodels.formula.api as smf   # regresi: smf.ols setara lm() di R
from statsmodels.stats.diagnostic import het_breuschpagan          # uji heteroskedastisitas
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif  # VIF
from statsmodels.stats.stattools import jarque_bera                # uji normalitas Jarque-Bera

# 1. KONFIGURASI HALAMAN
# Judul tab browser, ikon, dan layout lebar supaya grafik muat
st.set_page_config(page_title="Harga Rumah Jabodetabek", page_icon="🏠", layout="wide")

# 2. KAMUS LABEL
# LBL: nama kolom di data ke nama yang tampil di sumbu/judul grafik
LBL = {"price_miliar": "Harga (Rp miliar)", "land_size_m2": "Luas tanah (m²)",
       "building_size_m2": "Luas bangunan (m²)", "city": "Kota", "certificate": "Sertifikat"}
# NUM: daftar variabel numerik (dipakai untuk statistik deskriptif dan pilihan variabel)
NUM = ["price_miliar", "land_size_m2", "building_size_m2"]


# 3. MEMUAT DAN MEMBERSIHKAN DATA
# @st.cache_data: hasil fungsi disimpan, jadi data tidak dibaca ulang tiap klik
@st.cache_data
def load(path="jabodetabek_house_price.csv"):
    # Baca file CSV. File harus satu folder dengan app.py di GitHub
    d = pd.read_csv(path)
    n_var = d.shape[1]                  # jumlah kolom (variabel) di file
    # log: catatan jumlah baris di setiap langkah, ditampilkan di Tab 1 (syarat B.6 soal)
    log = [("Data awal", len(d), "Hasil scraping Rumah123, 5 variabel terpilih")]
    # Rapikan nama kota: hapus spasi di depan/belakang (" Bogor" ke "Bogor")
    d["city"] = d["city"].str.strip()
    # Sederhanakan sertifikat jadi 4 kategori (teks asli di file tidak diubah)
    # SHM, HGB, "Tidak diketahui" (kosong), dan sisanya "Lainnya"
    c = d.certificate.fillna("").str.lower()
    d["certificate"] = np.select([c.str.startswith("shm"), c.str.startswith("hgb"), c == ""],
                                 ["SHM", "HGB", "Tidak diketahui"], "Lainnya")

    # step(): fungsi kecil untuk membuang baris sesuai aturan lalu mencatatnya ke log
    def step(name, mask, why):
        nonlocal d
        d = d[mask(d)]
        log.append((name, len(d), why))

    # Data hilang: buang baris yang luas tanah/bangunannya kosong (2 baris)
    step("Hapus baris luas kosong", lambda x: x.land_size_m2.notna() & x.building_size_m2.notna(),
         "2 baris tanpa luas tanah/bangunan")
    # Nilai tidak masuk akal: luas < 10 m² atau > 2.000 m² dianggap salah input
    step("Hapus luas tidak masuk akal",
         lambda x: x.land_size_m2.between(10, 2000) & x.building_size_m2.between(10, 2000),
         "Luas < 10 m² atau > 2.000 m² (kemungkinan salah input)")
    # Pencilan harga: aturan IQR pada log10(harga).
    # Batas = Q1 - 1.5*IQR sampai Q3 + 1.5*IQR (aturan boxplot)
    q1, q3 = np.log10(d.price_in_rp).quantile([.25, .75])
    i = q3 - q1
    step("Hapus pencilan harga (IQR pada log10 harga)",
         lambda x: np.log10(x.price_in_rp).between(q1 - 1.5 * i, q3 + 1.5 * i),
         "Harga ekstrem (mis. Rp580 miliar) merusak model")
    # Buat kolom baru: harga dalam miliar (supaya mudah dibaca)
    out = d.reset_index(drop=True).copy()
    out["price_miliar"] = out.price_in_rp / 1e9
    # Transformasi logaritma natural (ln) untuk regresi, karena data miring ke kanan
    out["ln_price"] = np.log(out.price_in_rp)
    out["ln_land"] = np.log(out.land_size_m2)
    out["ln_bld"] = np.log(out.building_size_m2)
    # Kembalikan data bersih, log pembersihan, dan jumlah variabel
    return out[["city", "certificate", "land_size_m2", "building_size_m2", "price_miliar",
                "ln_price", "ln_land", "ln_bld"]], log, n_var


# 4. FUNGSI BANTU
# pf(): format p-value. Jika sangat kecil ditulis "< 0.001"
def pf(p):
    return "< 0.001" if p < .001 else f"= {p:.3f}"


# interp(): menampilkan kotak biru berisi interpretasi di bawah grafik (syarat B.3 & B.4)
def interp(t):
    st.info("**Interpretasi:** " + t)


# desc(): statistik deskriptif satu variabel (setara summary() di R, ditambah std & skewness)
def desc(s):
    return pd.Series({"n": s.count(), "Rata-rata": s.mean(), "Median": s.median(), "Std": s.std(),
                      "Min": s.min(), "Q1": s.quantile(.25), "Q3": s.quantile(.75),
                      "Maks": s.max(), "Skewness": s.skew()})


# 5. MEMUAT DATA
# df = data bersih, log = catatan pembersihan, n_var = jumlah variabel di file
df, log, n_var = load()

# 6. SIDEBAR: FILTER DATA
st.sidebar.header("Filter data")
cities = sorted(df.city.unique())               # daftar kota untuk pilihan
certs = sorted(df.certificate.unique())         # daftar sertifikat untuk pilihan
sc = st.sidebar.multiselect("Kota", cities, cities)          # pilih kota (awal: semua)
ss = st.sidebar.multiselect("Sertifikat", certs, certs)      # pilih sertifikat (awal: semua)
pmin, pmax = float(df.price_miliar.min()), float(df.price_miliar.max())
lo, hi = st.sidebar.slider("Harga (Rp miliar)", pmin, pmax, (pmin, pmax))  # rentang harga
# f = data yang sudah difilter. SEMUA grafik dan model di bawah memakai f
f = df[df.city.isin(sc) & df.certificate.isin(ss) & df.price_miliar.between(lo, hi)]
st.sidebar.metric("Observasi terpilih", f"{len(f):,}")

# 7. JUDUL DAN TAB
st.title("🏠 Eksplorasi & Regresi Harga Rumah Jabodetabek")
st.caption("UTS Eksplorasi dan Visualisasi Data Multivariat · Program Studi Statistika, Universitas Akprind Indonesia")
# Jika filter terlalu ketat (< 30 baris), hentikan supaya model tidak error
if len(f) < 30:
    st.warning("Data terlalu sedikit (< 30). Longgarkan filter di sidebar.")
    st.stop()

# Enam tab = enam bagian soal
tabs = st.tabs(["1. Masalah & Data", "2. Univariat", "3. Bivariat", "4. Multivariat",
                "5. Regresi", "6. Kesimpulan"])

# TAB 1. PERMASALAHAN DAN DATASET (Soal no. 1)
with tabs[0]:
    st.header("Permasalahan dan Dataset")
    # Latar belakang dan tujuan (diminta soal no. 1)
    st.subheader("Latar belakang")
    st.write("Harga rumah di Jabodetabek sangat bervariasi. Pembeli, penjual, dan agen perlu tahu "
             "faktor apa yang paling terkait dengan harga: ukuran rumah, lokasi (kota), atau status sertifikat.")
    st.subheader("Tujuan analisis")
    st.markdown("- Menggambarkan karakteristik harga, luas tanah, luas bangunan, kota, dan sertifikat.\n"
                "- Mengeksplorasi hubungan antarvariabel (bivariat dan multivariat).\n"
                "- Membangun model regresi linear berganda untuk menjelaskan harga rumah.")
    # Sumber data (wajib dicantumkan, syarat A.5)
    st.subheader("Sumber data")
    st.write("Data iklan rumah dijual di Jabodetabek dari **Rumah123.com** (hasil scraping, dataset Kaggle "
             "*Jabodetabek House Price*). Harga adalah **harga penawaran**, bukan harga transaksi.")
    # Jumlah observasi dan variabel (diminta soal no. 1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observasi awal", f"{log[0][1]:,}")
    c2.metric("Observasi setelah dibersihkan", f"{len(df):,}")
    c3.metric("Variabel di data asli", 27)
    c4.metric("Variabel dipakai", n_var)
    # Definisi setiap variabel (diminta soal no. 1)
    st.subheader("Definisi variabel")
    st.dataframe(pd.DataFrame([
        ("price_in_rp → Harga", "Numerik (respons)", "Harga penawaran rumah. Ditampilkan dalam Rp miliar; regresi memakai ln(Rp)"),
        ("land_size_m2 → Luas tanah", "Numerik", "Luas tanah (m²)"),
        ("building_size_m2 → Luas bangunan", "Numerik", "Luas bangunan (m²)"),
        ("city → Kota", "Kategorik (9 level)", "Kota/kabupaten lokasi rumah (Bogor, Depok, Tangerang, Bekasi, 5 wilayah Jakarta)"),
        ("certificate → Sertifikat", "Kategorik (4 level)", "SHM, HGB, Lainnya (PPJB/girik/adat/hak pakai), Tidak diketahui (kosong)")],
        columns=["Variabel", "Tipe", "Definisi"]), hide_index=True)
    # Tabel pembersihan data (syarat B.6: jelaskan data hilang, pencilan, transformasi)
    st.subheader("Pengolahan data (data hilang, pencilan, transformasi)")
    st.dataframe(pd.DataFrame(log, columns=["Langkah", "Baris tersisa", "Alasan"]), hide_index=True)
    st.markdown("- **Data hilang**: `certificate` kosong diberi kategori *Tidak diketahui*; baris dengan luas kosong dibuang. "
                "Dari 27 variabel data asli, `building_age`, `year_built`, `building_orientation` tidak dipilih karena hilang > 40%.\n"
                "- **Duplikat**: baris dengan 5 variabel sama **tidak dihapus** karena sebagian besar adalah unit berbeda dengan spesifikasi sama (mis. rumah kembar di satu perumahan); iklan ganda diperkirakan < 3%.\n"
                "- **Transformasi**: harga, luas tanah, dan luas bangunan di-ln pada regresi karena distribusinya miring ke kanan.\n"
                "- Kolom `kota` dirapikan (spasi dihapus); level sertifikat *hak pakai* (1 baris) digabung ke *Lainnya*.")

# TAB 2. EKSPLORASI UNIVARIAT (Soal no. 2) - 1 variabel per grafik
with tabs[1]:
    st.header("Eksplorasi Univariat")
    # Statistik deskriptif variabel numerik: tiap kolom = satu variabel
    st.subheader("Statistik deskriptif")
    st.dataframe(pd.DataFrame({LBL[c]: desc(f[c]) for c in NUM}).T.round(2))
    # Tabel frekuensi variabel kategorik (kota dan sertifikat)
    a, b = st.columns(2)
    a.dataframe(f.city.value_counts().rename("Jumlah").to_frame())
    b.dataframe(f.certificate.value_counts().rename("Jumlah").to_frame())

    # Visualisasi 1: histogram + boxplot (variabel numerik)
    st.subheader("Visualisasi 1: Histogram + boxplot (variabel numerik)")
    v = st.selectbox("Variabel numerik", NUM, format_func=LBL.get)   # dosen bisa ganti variabel
    logx = st.checkbox("Skala log pada sumbu x")                     # opsi skala log
    # marginal="box" menambah boxplot di atas histogram untuk melihat pencilan
    fig = px.histogram(f, x=v, nbins=50, marginal="box", labels=LBL, title=f"Distribusi {LBL[v]}")
    fig.update_layout(yaxis_title="Frekuensi")
    if logx:
        fig.update_xaxes(type="log")
    st.plotly_chart(fig)
    # Hitung rata-rata, median, skewness, dan jumlah pencilan IQR untuk interpretasi otomatis
    s = f[v]
    q1, q3 = s.quantile([.25, .75])
    nout = int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())
    shape = "miring ke kanan (rata-rata > median)" if s.skew() > .5 else \
            "miring ke kiri" if s.skew() < -.5 else "relatif simetris"
    interp(f"{LBL[v]} berdistribusi {shape}: rata-rata {s.mean():,.2f}, median {s.median():,.2f}, skewness {s.skew():.2f}. "
           f"Terdapat {nout} pencilan menurut aturan IQR ({nout / len(s):.1%} data), yaitu rumah-rumah dengan nilai jauh di atas kebanyakan.")

    # Visualisasi 2: diagram batang kota (variabel kategorik)
    st.subheader("Visualisasi 2: Diagram batang jumlah rumah per kota")
    cc = f.city.value_counts().reset_index()   # hitung jumlah rumah per kota
    cc.columns = ["city", "n"]
    st.plotly_chart(px.bar(cc, x="city", y="n", labels={"city": "Kota", "n": "Jumlah rumah"},
                           title="Jumlah rumah dijual per kota", text="n"))
    interp(f"Kota terbanyak adalah {cc.city[0]} ({cc.n[0]:,} rumah, {cc.n[0] / len(f):.1%}); paling sedikit {cc.city.iloc[-1]} "
           f"({cc.n.iloc[-1]:,} rumah). Sebaran antar kota tidak seimbang sehingga perbandingan antar kota perlu hati-hati.")

    # Visualisasi 3: diagram batang sertifikat (variabel kategorik)
    st.subheader("Visualisasi 3: Diagram batang jenis sertifikat")
    ct = f.certificate.value_counts().reset_index()
    ct.columns = ["certificate", "n"]
    st.plotly_chart(px.bar(ct, x="certificate", y="n", color="certificate",
                           labels={"certificate": "Sertifikat", "n": "Jumlah rumah"},
                           title="Jumlah rumah per jenis sertifikat", text="n"))
    interp(f"Sertifikat dominan: {ct.certificate[0]} ({ct.n[0] / len(f):.1%} data). Kategori lain jauh lebih sedikit.")

# TAB 3. EKSPLORASI BIVARIAT (Soal no. 3) - 2 variabel per grafik
with tabs[2]:
    st.header("Eksplorasi Bivariat")
    # Visualisasi 1: scatter plot (numerik vs numerik)
    st.subheader("Visualisasi 1: Scatter plot harga vs luas (numerik–numerik)")
    xv = st.selectbox("Variabel x", ["building_size_m2", "land_size_m2"], format_func=LBL.get)
    lg = st.checkbox("Skala log-log", value=True)
    # trendline="ols" = garis regresi (setara geom_smooth(method="lm") di R)
    # log_x/log_y membuat sumbu log supaya titik tidak menumpuk di kiri bawah
    st.plotly_chart(px.scatter(f, x=xv, y="price_miliar", opacity=.4, trendline="ols",
                               trendline_options=dict(log_x=lg, log_y=lg), log_x=lg, log_y=lg,
                               labels=LBL, title=f"Harga vs {LBL[xv]}"))
    # Korelasi Pearson (hubungan linear) dan Spearman (hubungan berurutan, tahan pencilan)
    pr = f[[xv, "price_miliar"]].corr().iloc[0, 1]
    sp = f[[xv, "price_miliar"]].corr(method="spearman").iloc[0, 1]
    interp(f"Hubungan positif: korelasi Pearson {pr:.2f}, Spearman {sp:.2f}. Semakin besar {LBL[xv].lower()}, harga cenderung naik, "
           "tetapi sebaran lebar, artinya ukuran saja belum cukup menjelaskan harga.")

    # Visualisasi 2: boxplot (numerik vs kategorik)
    st.subheader("Visualisasi 2: Boxplot harga per kota (numerik–kategorik)")
    ly = st.checkbox("Skala log pada harga", value=True, key="ly2")
    # Urutkan kota berdasarkan median harga supaya mudah dibandingkan
    order = f.groupby("city").price_miliar.median().sort_values().index.tolist()
    st.plotly_chart(px.box(f, x="city", y="price_miliar", color="city", log_y=ly, category_orders={"city": order},
                           labels=LBL, title="Distribusi harga per kota (urut median)"))
    md = f.groupby("city").price_miliar.median().sort_values()
    interp(f"Median harga tertinggi di {md.index[-1]} (Rp{md.iloc[-1]:.2f} miliar) dan terendah di {md.index[0]} "
           f"(Rp{md.iloc[0]:.2f} miliar), selisih {md.iloc[-1] / md.iloc[0]:.1f} kali. Lokasi sangat membedakan harga.")

    # Visualisasi 3: boxplot (numerik vs kategorik)
    st.subheader("Visualisasi 3: Boxplot harga per sertifikat (numerik–kategorik)")
    st.plotly_chart(px.box(f, x="certificate", y="price_miliar", color="certificate", log_y=True,
                           labels=LBL, title="Distribusi harga per jenis sertifikat"))
    mc = f.groupby("certificate").price_miliar.median().sort_values()
    interp("Median harga per sertifikat: " + ", ".join(f"{k} Rp{v:.2f} miliar" for k, v in mc.items()) +
           ". Perbedaan ini belum mengontrol ukuran dan kota, jadi diuji ulang di regresi.")

# TAB 4. EKSPLORASI MULTIVARIAT (Soal no. 4) - minimal 3 variabel per grafik
with tabs[3]:
    st.header("Eksplorasi Multivariat")
    # Visualisasi 1: bubble plot (4 variabel sekaligus)
    st.subheader("Visualisasi 1: Bubble plot luas tanah × luas bangunan × harga × kota")
    lg2 = st.checkbox("Skala log-log", value=True, key="lg2")
    # x = luas tanah, y = luas bangunan, warna = kota, ukuran titik = harga
    st.plotly_chart(px.scatter(f, x="land_size_m2", y="building_size_m2", color="city", size="price_miliar",
                               size_max=18, opacity=.5, log_x=lg2, log_y=lg2, labels=LBL,
                               title="Luas tanah vs luas bangunan (warna = kota, ukuran = harga)"))
    rr = f[["ln_land", "ln_bld"]].corr().iloc[0, 1]    # korelasi dua luas (skala log)
    interp(f"Luas tanah dan luas bangunan berkorelasi kuat (r = {rr:.2f} pada skala log), sehingga titik membentuk pita naik. "
           "Titik besar (harga tinggi) menumpuk di area luas besar, dan warna kota terlihat berkelompok di area tertentu.")

    # Visualisasi 2: scatter per kota (facet) + warna sertifikat
    st.subheader("Visualisasi 2: Harga vs luas bangunan per kota dan sertifikat")
    # facet_col="city" memecah grafik per kota (panel kecil), warna = sertifikat
    st.plotly_chart(px.scatter(f, x="building_size_m2", y="price_miliar", color="certificate", facet_col="city",
                               facet_col_wrap=3, log_x=True, log_y=True, opacity=.5, height=800, labels=LBL,
                               title="Harga vs luas bangunan, dipisah per kota (warna = sertifikat)"))
    # Harga per m² bangunan (Rp juta) = harga(miliar)*1000 / luas bangunan
    ppm = (f.price_miliar * 1000 / f.building_size_m2).groupby(f.city).median().sort_values()
    interp(f"Di setiap kota hubungan harga–luas bangunan tetap positif, tetapi posisi vertikal kelompok berbeda: median harga per m² "
           f"bangunan tertinggi di {ppm.index[-1]} (Rp{ppm.iloc[-1]:.1f} juta) dan terendah di {ppm.index[0]} (Rp{ppm.iloc[0]:.1f} juta). "
           "Artinya pada luas yang sama, lokasi menentukan harga.")

    # Visualisasi 3: heatmap korelasi (setara ggcorrplot di R)
    st.subheader("Visualisasi 3: Heatmap korelasi variabel numerik (skala log)")
    cm = f[["ln_price", "ln_land", "ln_bld"]].corr()          # matriks korelasi Pearson
    cm.index = cm.columns = ["ln(Harga)", "ln(Luas tanah)", "ln(Luas bangunan)"]
    st.plotly_chart(px.imshow(cm, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                              labels=dict(color="Korelasi"), title="Matriks korelasi Pearson"))
    interp(f"ln(Harga) berkorelasi kuat dengan ln(Luas bangunan) (r = {cm.iloc[0, 2]:.2f}) dan ln(Luas tanah) (r = {cm.iloc[0, 1]:.2f}); "
           f"kedua luas saling berkorelasi (r = {cm.iloc[1, 2]:.2f}), sehingga multikolinearitas perlu dicek lewat VIF.")

# TAB 5. REGRESI LINEAR BERGANDA (Soal no. 5)
with tabs[4]:
    st.header("Regresi Linear Berganda")

    # 12. MEMBANGUN MODEL (setara lm(prestige ~ education + income) di R)
    # Kategori acuan dummy: kota terbanyak dan sertifikat SHM.
    # Koefisien dummy dibaca sebagai selisih terhadap kategori acuan ini
    ref = f.city.value_counts().idxmax()
    refc = "SHM" if "SHM" in set(f.certificate) else f.certificate.value_counts().idxmax()
    # Rumus model: ln(harga) ~ ln(luas tanah) + ln(luas bangunan) + kota + sertifikat
    terms = ["ln_land", "ln_bld"]
    if f.city.nunique() > 1:
        terms.append(f"C(city, Treatment(reference='{ref}'))")          # C() = ubah kategori jadi dummy
    if f.certificate.nunique() > 1:
        terms.append(f"C(certificate, Treatment(reference='{refc}'))")
    # smf.ols(...).fit() = menghitung model OLS (setara lm() di R)
    m = smf.ols("ln_price ~ " + " + ".join(terms), f).fit()

    # nice(): ubah nama koefisien teknis jadi nama yang mudah dibaca
    def nice(n):
        if n == "Intercept": return "Intercept"
        if n == "ln_land": return "ln(Luas tanah)"
        if n == "ln_bld": return "ln(Luas bangunan)"
        lv = n.split("[T.")[1].rstrip("]")
        return f"Kota: {lv} (vs {ref})" if n.startswith("C(city") else f"Sertifikat: {lv} (vs {refc})"

    # 13. ALASAN PEMILIHAN VARIABEL (syarat B.5)
    st.subheader("Pemilihan variabel")
    st.markdown("- **Respons**: ln(Harga). Harga miring ke kanan; log menstabilkan ragam dan koefisien dapat dibaca sebagai persentase.\n"
                "- **Prediktor**: ln(Luas tanah), ln(Luas bangunan) (ukuran fisik), Kota dan Sertifikat (dummy). "
                f"Kategori acuan: Kota = {ref}, Sertifikat = {refc}.")

    # 14. HASIL ESTIMASI (setara summary(model) di R)
    names = [nice(k) for k in m.params.index]
    tb = pd.DataFrame({"Koefisien": m.params.values, "Std. error": m.bse.values, "t": m.tvalues.values,
                       "p-value": m.pvalues.values}, index=names)
    # Efek persen untuk dummy: (exp(koef) - 1) * 100. Contoh koef 0.4 ke harga +49%
    tb["Efek harga (%)"] = [np.nan if n in ("Intercept", "ln(Luas tanah)", "ln(Luas bangunan)")
                            else (np.exp(b) - 1) * 100 for n, b in zip(names, m.params.values)]
    st.subheader("Hasil estimasi")
    st.dataframe(tb.round(4))
    # RMSE = akar rata-rata kuadrat residual (ukuran kesalahan prediksi, skala ln)
    rmse = float(np.sqrt(np.mean(m.resid ** 2)))
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("R²", f"{m.rsquared:.3f}"); k2.metric("R² adj.", f"{m.rsquared_adj:.3f}")
    k3.metric("F-statistik", f"{m.fvalue:,.1f}"); k4.metric("p-value F", pf(m.f_pvalue).replace("= ", ""))
    k5.metric("RMSE (skala ln)", f"{rmse:.3f}")

    # 15. PERSAMAAN MODEL (syarat B.5)
    st.subheader("Persamaan model")
    eq = f"ln(Harga) = {m.params['Intercept']:.3f}"
    for k, v in m.params.items():
        if k != "Intercept":
            eq += f"\n   {'+' if v >= 0 else '-'} {abs(v):.3f} × [{nice(k)}]"
    st.code(eq)

    # 16. INTERPRETASI KOEFISIEN DAN UJI HIPOTESIS
    bl, bb = m.params["ln_land"], m.params["ln_bld"]     # koefisien luas tanah dan bangunan
    sig = int((m.pvalues.drop("Intercept") < .05).sum())  # jumlah koefisien signifikan (p < 0.05)
    st.subheader("Interpretasi koefisien dan pengujian")
    # Model log-log: koefisien = elastisitas (naik 1% ke harga naik koef %)
    # Uji F (simultan): apakah minimal satu prediktor berpengaruh
    # Uji t (parsial): apakah tiap prediktor berpengaruh sendiri-sendiri
    st.markdown(f"- **Luas tanah**: naik 1% → harga naik sekitar {bl:.2f}% (kota, sertifikat, luas bangunan tetap), "
                f"{'signifikan' if m.pvalues['ln_land'] < .05 else 'tidak signifikan'} (p {pf(m.pvalues['ln_land'])}).\n"
                f"- **Luas bangunan**: naik 1% → harga naik sekitar {bb:.2f}%, "
                f"{'signifikan' if m.pvalues['ln_bld'] < .05 else 'tidak signifikan'} (p {pf(m.pvalues['ln_bld'])}).\n"
                "- **Dummy kategorik**: kolom *Efek harga (%)* = selisih harga rata-rata (geometrik) dibanding kategori acuan, prediktor lain tetap.\n"
                f"- **Uji simultan F**: p {pf(m.f_pvalue)} → {'minimal satu' if m.f_pvalue < .05 else 'tidak ada'} prediktor berpengaruh pada α = 5%.\n"
                f"- **Uji parsial t**: {sig} dari {len(m.params) - 1} koefisien signifikan pada α = 5%.")

    # 17. UJI ASUMSI RESIDUAL (setara bagian "Uji Asumsi Residual" di contoh R)
    X = m.model.exog                                       # matriks prediktor model
    # VIF: mengukur multikolinearitas (VIF >= 5 berarti kolinearitas tinggi)
    vf = {nm: vif(X, i) for i, nm in enumerate(m.model.exog_names) if nm in ("ln_land", "ln_bld")}
    res = np.asarray(m.resid)                              # residual = nilai aktual - prediksi
    bp = het_breuschpagan(res, X)[1]                       # Breusch-Pagan (heteroskedastisitas)
    jb = jarque_bera(res)[1]                               # Jarque-Bera (normalitas)
    sw = stats.shapiro(res)[1]                             # Shapiro-Wilk (= shapiro.test di R)
    ks = stats.kstest(res, stats.norm(res.mean(), res.std()).cdf)[1]   # Kolmogorov-Smirnov (= ks.test di R)
    # Glejser: regresikan |residual| pada prediktor (sama seperti contoh R: lm(abs(res) ~ ...))
    gl = smf.ols("ar ~ ln_land + ln_bld", f.assign(ar=np.abs(res))).fit().f_pvalue
    cd = m.get_influence().cooks_distance[0]               # Cook's distance (pengamatan berpengaruh)

    st.subheader("Evaluasi model")
    st.markdown(f"- R² = {m.rsquared:.3f}: model menjelaskan {m.rsquared * 100:.1f}% variasi ln(harga); R² adj. = {m.rsquared_adj:.3f}.\n"
                f"- RMSE = {rmse:.3f} (skala ln) ≈ kesalahan prediksi harga sekitar {(np.exp(rmse) - 1) * 100:.0f}%.\n"
                f"- VIF ln(Luas tanah) = {vf['ln_land']:.2f}, ln(Luas bangunan) = {vf['ln_bld']:.2f} "
                f"({'multikolinearitas rendah-sedang' if max(vf.values()) < 5 else 'multikolinearitas tinggi (≥ 5)'}).")

    st.subheader("Uji asumsi residual")
    # Aturan keputusan: p-value >= 0.05 ke asumsi terpenuhi (H0 tidak ditolak)
    ok = lambda p: "Terpenuhi" if p >= .05 else "Tidak terpenuhi"
    st.dataframe(pd.DataFrame([
        ("Normalitas", "Shapiro–Wilk", pf(sw).replace("= ", ""), ok(sw)),
        ("Normalitas", "Kolmogorov–Smirnov", pf(ks).replace("= ", ""), ok(ks)),
        ("Normalitas", "Jarque–Bera", pf(jb).replace("= ", ""), ok(jb)),
        ("Homoskedastisitas", "Breusch–Pagan", pf(bp).replace("= ", ""), ok(bp)),
        ("Homoskedastisitas", "Glejser (|residual| ~ prediktor)", pf(gl).replace("= ", ""), ok(gl))],
        columns=["Asumsi", "Uji", "p-value", "Kesimpulan (α = 5%)"]), hide_index=True)
    # Durbin-Watson/ACF di contoh R tidak dipakai di sini: data rumah bukan data deret waktu
    st.caption("Autokorelasi (Durbin–Watson, ACF) tidak diuji karena data cross-section: baris iklan tidak berurutan waktu.")
    normal_ok, homo_ok = min(sw, ks, jb) >= .05, min(bp, gl) >= .05
    interp(f"Normalitas residual {'terpenuhi' if normal_ok else 'tidak terpenuhi'} dan homoskedastisitas "
           f"{'terpenuhi' if homo_ok else 'tidak terpenuhi'} pada α = 5%. "
           + ("" if normal_ok and homo_ok else "Dengan n besar koefisien tetap konsisten, tetapi uji t dan p-value perlu dibaca hati-hati "
              "(uji juga sangat sensitif pada n besar)."))

    # 18. VISUALISASI REGRESI (syarat: minimal 2; di sini 5)
    # Visualisasi regresi 1: aktual vs prediksi
    st.subheader("Visualisasi regresi 1: Nilai aktual vs prediksi")
    fv = np.asarray(m.fittedvalues)                        # nilai prediksi model
    ac = f.ln_price.values                                 # nilai aktual
    fig = px.scatter(x=fv, y=ac, opacity=.4, labels={"x": "ln(Harga) prediksi", "y": "ln(Harga) aktual"},
                     title="Aktual vs prediksi")
    # Garis y = x: titik di garis berarti prediksi tepat
    fig.add_trace(go.Scatter(x=[ac.min(), ac.max()], y=[ac.min(), ac.max()], mode="lines",
                             name="Garis y = x", line=dict(dash="dash", color="red")))
    st.plotly_chart(fig)
    interp(f"Titik mengikuti garis y = x dengan R² = {m.rsquared:.2f}; makin dekat garis, makin akurat prediksi. "
           "Sebaran di sekitar garis menunjukkan galat yang masih ada.")

    # Visualisasi regresi 2: residual vs prediksi (setara plot Residuals vs Fitted di R)
    st.subheader("Visualisasi regresi 2: Residual vs nilai prediksi")
    fig = px.scatter(x=fv, y=res, opacity=.4,
                     labels={"x": "ln(Harga) prediksi", "y": "Residual"}, title="Residual vs prediksi")
    fig.add_hline(y=0, line_dash="dash", line_color="red")    # garis nol sebagai acuan
    st.plotly_chart(fig)
    interp(f"Residual seharusnya acak di sekitar nol tanpa pola. Uji Breusch–Pagan p {pf(bp)} "
           f"({'ada indikasi sebaran residual tidak merata' if bp < .05 else 'tidak menunjukkan masalah'}).")

    # Visualisasi regresi 3: histogram residual (setara hist(res) di R)
    st.subheader("Visualisasi regresi 3: Histogram residual")
    st.plotly_chart(px.histogram(x=res, nbins=40, labels={"x": "Residual"}, title="Distribusi residual")
                    .update_layout(yaxis_title="Frekuensi"))
    interp(f"Residual berpusat di sekitar nol (rata-rata {res.mean():.3f}), skewness {stats.skew(res):.2f}, kurtosis berlebih {stats.kurtosis(res):.2f}. "
           f"{'Bentuknya mendekati lonceng.' if normal_ok else 'Ekor lebih tebal dari kurva normal (banyak residual ekstrem).' if stats.kurtosis(res) > 1 else 'Bentuknya agak menyimpang dari normal.'}")

    # Visualisasi regresi 4: Q-Q plot (setara qqnorm + qqline di R)
    st.subheader("Visualisasi regresi 4: Q-Q plot residual")
    # probplot menghasilkan kuantil teoretis (normal) dan kuantil residual + garis pembanding
    (osm, osr), (qs, qi, _) = stats.probplot(res, dist="norm")
    fig = px.scatter(x=osm, y=osr, opacity=.5, labels={"x": "Kuantil teoretis (normal)", "y": "Kuantil residual"},
                     title="Q-Q plot residual")
    fig.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[qi + qs * osm.min(), qi + qs * osm.max()], mode="lines",
                             name="Garis normal", line=dict(color="red")))
    st.plotly_chart(fig)
    interp(f"Titik {'mengikuti garis' if normal_ok else 'menjauhi garis di kedua ujung'}: "
           f"{'residual mendekati normal.' if normal_ok else 'ekor distribusi lebih berat dari normal, sejalan dengan uji Shapiro–Wilk.'}")

    # Visualisasi regresi 5: Cook's distance (setara plot Residuals vs Leverage di R)
    st.subheader("Visualisasi regresi 5: Cook's distance (pengamatan berpengaruh)")
    fig = px.scatter(x=np.arange(len(cd)), y=cd, opacity=.5,
                     labels={"x": "Indeks pengamatan", "y": "Cook's distance"}, title="Cook's distance")
    fig.add_hline(y=4 / len(cd), line_dash="dash", line_color="red")   # batas umum 4/n
    st.plotly_chart(fig)
    interp(f"Nilai maksimum Cook's distance {cd.max():.3f} ({'di bawah' if cd.max() < 1 else 'di atas'} 1, batas umum pengamatan sangat berpengaruh); "
           f"{int((cd > 4 / len(cd)).sum())} pengamatan melewati batas 4/n (garis merah) sehingga perlu dicermati{', tetapi tidak ada yang mendominasi model' if cd.max() < 1 else ', dan ada pengamatan yang sangat berpengaruh'}.")

    # 19. KETERBATASAN MODEL (diminta soal no. 5)
    st.subheader("Keterbatasan model")
    st.markdown("- Harga adalah harga **penawaran** iklan, bukan transaksi.\n"
                "- Variabel penting tidak ada di model (kondisi rumah, jumlah kamar, akses, kecamatan), sehingga ada bias variabel yang terlewat.\n"
                "- Luas tanah dan luas bangunan berkorelasi, sehingga koefisien masing-masing kurang stabil.\n"
                "- Kota terlalu agregat (misalnya Bogor mencakup area mahal dan murah).\n"
                "- Model hanya menunjukkan **asosiasi**, bukan sebab-akibat.\n"
                "- Pencilan dan data tidak valid dibuang, sehingga model tidak berlaku untuk rumah super mewah atau sangat murah.\n"
                + ("" if normal_ok and homo_ok else "- Asumsi normalitas dan/atau homoskedastisitas tidak terpenuhi (lihat uji asumsi), sehingga p-value dan selang kepercayaan perlu dibaca hati-hati.\n"))

# TAB 6. KESIMPULAN DAN REKOMENDASI (Soal: Kesimpulan)
with tabs[5]:
    st.header("Kesimpulan dan Rekomendasi")
    # Efek persen tiap kota terhadap kota acuan (untuk rentang di kesimpulan no. 4)
    cs = tb.loc[[n for n in tb.index if n.startswith("Kota")], "Efek harga (%)"] if any(n.startswith("Kota") for n in tb.index) else pd.Series(dtype=float)
    st.subheader("Kesimpulan")
    # Angka di kesimpulan dihitung otomatis dari data/model, jadi selalu cocok dengan hasil
    st.markdown(f"1. Harga rumah sangat miring ke kanan (median Rp{f.price_miliar.median():.2f} miliar vs rata-rata Rp{f.price_miliar.mean():.2f} miliar).\n"
                f"2. Harga berhubungan positif dengan luas bangunan (r = {f[['building_size_m2', 'price_miliar']].corr().iloc[0, 1]:.2f}) dan luas tanah; lokasi (kota) membedakan harga secara jelas.\n"
                f"3. Model regresi menjelaskan {m.rsquared * 100:.1f}% variasi ln(harga). Elastisitas luas bangunan {bb:.2f} dan luas tanah {bl:.2f}.\n"
                + (f"4. Dibanding {ref}, efek kota berkisar dari {cs.min():.0f}% sampai {cs.max():.0f}%.\n" if len(cs) else ""))
    st.subheader("Rekomendasi")
    st.markdown("- **Pembeli**: bandingkan harga per m² dalam kota yang sama, bukan antar kota.\n"
                "- **Penjual/agen**: tetapkan harga dengan acuan luas bangunan dan lokasi, lalu sesuaikan sertifikat.\n"
                "- **Penelitian lanjutan**: tambahkan jumlah kamar, kondisi, dan kecamatan; coba model regresi robust atau interaksi kota × luas.")
