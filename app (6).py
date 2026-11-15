# DASHBOARD UTS EKSPLORASI & VISUALISASI DATA MULTIVARIAT
# Topik : Harga Rumah Jabodetabek (data iklan Rumah123)
# Alur  : data, univariat, bivariat, multivariat, regresi, kesimpulan

# 0. IMPORT LIBRARY
import base64, gzip, io, os                 # untuk membaca data tertanam (opsional)
import numpy as np                          # hitungan numerik (log, exp, akar, dll)
import pandas as pd                         # mengolah tabel data (DataFrame)
import plotly.express as px                 # grafik interaktif cepat
import plotly.graph_objects as go           # menambah garis/elemen khusus pada grafik
import streamlit as st                      # membuat tampilan aplikasi web
from scipy import stats                     # uji normalitas dan Q-Q plot
import statsmodels.formula.api as smf       # regresi: smf.ols setara lm() di R
from statsmodels.stats.diagnostic import het_breuschpagan                          # uji heteroskedastisitas
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif  # VIF
from statsmodels.stats.stattools import jarque_bera                                # uji normalitas Jarque-Bera

# 0b. PALET WARNA
# Sage (utama), terakota (aksen), krem (latar): lembut dan hangat, tidak mencolok
PALET = ["#4F7C74", "#D98B6A", "#6B8CAE", "#E0B35A", "#8E6C8A", "#3E8E9A", "#B5654A", "#8A9A5B", "#7A8199"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALET

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="Harga Rumah Jabodetabek", page_icon="🏠", layout="wide")

# 2. KAMUS LABEL
# LBL: nama kolom di data ke nama yang tampil di sumbu/judul grafik
LBL = {"price_miliar": "Harga (Rp miliar)", "land_size_m2": "Luas tanah (m²)",
       "building_size_m2": "Luas bangunan (m²)", "city": "Kota", "certificate": "Sertifikat"}
# NUM: daftar variabel numerik
NUM = ["price_miliar", "land_size_m2", "building_size_m2"]

# 2b. SUMBER DATA
# Data sudah tertanam (terkompres) di DATA_B64, jadi cukup upload app.py dan requirements.txt ke GitHub.
# Kalau jabodetabek_house_price.csv ada di folder yang sama, file itu yang dibaca.
DATA_B64 = """
H4sIALdv+GoC/+19S6+kyXHdvn8F4ZUNlIF8P7aCV14ZsLwmmuSYbA81IobDhfzrnflV1a26GZFxMrIe0xpLYFNS942bX74i43Hi
xN9+/vbHH37/7aff//y30x+//fJvp79+/elPv//7t//7w+//xZ3+8I9vf/3Tt5/+/PEXf/zh51++/e9vf/z6yw9fXK3m/D+nf/rh
x69//3Zyvp5cdqe//+Vffvdff/f388/++PWX3/3l64+/+5dvf/324xfr8iAU4ynV01/+/Icm0n/uz//46evv/vD1p/6/f/piaxx+
3tp6st5OJbw3o0QxJ5uD9FkhjkLelZOtSRL6mMptJJ9Orv2/kpAnEwrh+DObEP02Z9sw0UwlXCUS7f+00T1/NpGdTRHXrUZHjkCb
zX/669dvP/30b19/95//9rf/84fTn7/9/O3H09c/ff3l9Ke//vW//KcvbRuZo1DczmrPD5wjq90l+jJwK1T9sUqzXxZ9IhLu5PN8
6yZrM/1ax16PKG/allBgV3Jl367D3UYzxx9JTdTCfGItkkxO4bNILqfixTkZsqGpzyk+eRjPrnfAS7d1vUKhGtaD1R7XodqTdfj7
iKpN5uTFj4tE/x83wuT5EafXvh06pJeyJcosHvf+C7nf7cEKbq59faVTnH8t+dg2O1lFlcjpaiefQTJKEwnyqSAPqQ/lFMDBIBtc
2vEzUX7hyX61k1Tl7UrknLdldvLzu/Vxb5IJo85r80+ihBtvbXskgrhotdZxnXO7TaJMzONCN2V28rJitfRJ7EdUvuiJ3ITchYJ8
bz3RyN3OAUL2OtbdFW0KVrpzmd6fYpauz8cZSKcgb2iOSt3o6Yr1/0gzD4Gxps1zn6+dC60fRb/AW8dS/2L5pN7HyIxRZAnLSUxd
nOq1n1QdefQEu5Ucq5TgnFUzCMbSAWQlRLzN44mw8hNhybvi0vFHmgo57z6nQ3t9IW9o++ySlYe0jV7mFr0lxnl7bWJasLKpmeSQ
mTSf6tS+iVRzygrHkpfN+nqoEElBj9ogFqBxLPVPvD05p3za++7IA6WqXYHKrBmYv/WcDFho6s20KwUGciU+41o19S5HF3wYFFBT
J8nLZzNTCXmdMzW68pMlrCf+S7MErXm+kLO8w52/DyHOKmhXZx7GstSXtaFdUFnr01CFKWK8kLEFXZUVBxfwyeLk82h6tx+3srWe
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
"""


def baca_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    if DATA_B64.strip():
        teks = gzip.decompress(base64.b64decode(DATA_B64)).decode("utf-8")
        return pd.read_csv(io.StringIO(teks))
    st.error("Data tidak ditemukan. Taruh `jabodetabek_house_price.csv` satu folder dengan app.py, "
             "atau tempel data base64 di variabel DATA_B64.")
    st.stop()


# 3. MEMUAT DAN MEMBERSIHKAN DATA
# @st.cache_data: hasil fungsi disimpan, jadi data tidak dibaca ulang tiap klik
@st.cache_data
def load(path="jabodetabek_house_price.csv"):
    d = baca_data(path)
    n_var = d.shape[1]                  # jumlah kolom (variabel) di file
    # log: catatan jumlah baris di setiap langkah, ditampilkan di Tab 1
    log = [("Data awal", len(d), "Hasil scraping Rumah123, 5 variabel terpilih")]
    # Rapikan nama kota: hapus spasi di depan/belakang
    d["city"] = d["city"].str.strip()
    # Sederhanakan sertifikat jadi 4 kategori: SHM, HGB, "Tidak diketahui" (kosong), "Lainnya"
    c = d.certificate.fillna("").str.lower()
    d["certificate"] = np.select([c.str.startswith("shm"), c.str.startswith("hgb"), c == ""],
                                 ["SHM", "HGB", "Tidak diketahui"], "Lainnya")

    # step(): buang baris sesuai aturan lalu catat ke log
    def step(name, mask, why):
        nonlocal d
        d = d[mask(d)]
        log.append((name, len(d), why))

    # Data hilang: buang baris yang luas tanah/bangunannya kosong
    step("Hapus baris luas kosong", lambda x: x.land_size_m2.notna() & x.building_size_m2.notna(),
         "2 baris tanpa luas tanah/bangunan")
    # Nilai tidak masuk akal: luas < 10 m² atau > 2.000 m² dianggap salah input
    step("Hapus luas tidak masuk akal",
         lambda x: x.land_size_m2.between(10, 2000) & x.building_size_m2.between(10, 2000),
         "Luas < 10 m² atau > 2.000 m² (kemungkinan salah input)")
    # Pencilan harga: aturan IQR pada log10(harga), batas Q1 - 1.5*IQR sampai Q3 + 1.5*IQR
    q1, q3 = np.log10(d.price_in_rp).quantile([.25, .75])
    i = q3 - q1
    step("Hapus pencilan harga (IQR pada log10 harga)",
         lambda x: np.log10(x.price_in_rp).between(q1 - 1.5 * i, q3 + 1.5 * i),
         "Harga ekstrem (mis. Rp580 miliar) merusak model")
    out = d.reset_index(drop=True).copy()
    out["price_miliar"] = out.price_in_rp / 1e9            # harga dalam miliar (mudah dibaca)
    # Transformasi logaritma natural untuk regresi, karena data miring ke kanan
    out["ln_price"] = np.log(out.price_in_rp)
    out["ln_land"] = np.log(out.land_size_m2)
    out["ln_bld"] = np.log(out.building_size_m2)
    return out[["city", "certificate", "land_size_m2", "building_size_m2", "price_miliar",
                "ln_price", "ln_land", "ln_bld"]], log, n_var


# 4. FUNGSI BANTU
# pf(): format p-value. Jika sangat kecil ditulis "< 0.001"
def pf(p):
    return "< 0.001" if p < .001 else f"= {p:.3f}"


# interp(): kotak biru di bawah tiap grafik/uji. Selalu diawali "Berdasarkan output ...",
# lalu: untuk melihat apa (tujuan), hasil data (t), dan artinya (arti)
def interp(t, tujuan="", arti="", out="grafik di atas"):
    teks = f"Berdasarkan output {out}"
    teks += f", yang dipakai untuk melihat {tujuan.rstrip('.')}." if tujuan else ":"
    teks += f"\n\n**Hasil:** {t}"
    if arti:
        teks += f"\n\n**Artinya:** {arti}"
    st.info(teks)


# desc(): statistik deskriptif satu variabel (setara summary() di R, ditambah std & skewness)
def desc(s):
    return pd.Series({"n": s.count(), "Rata-rata": s.mean(), "Median": s.median(), "Std": s.std(),
                      "Koef. variasi (%)": s.std() / s.mean() * 100,
                      "Min": s.min(), "Q1": s.quantile(.25), "Q3": s.quantile(.75),
                      "Maks": s.max(), "Skewness": s.skew(), "Kurtosis": s.kurt()})


# 5. MEMUAT DATA
df, log, n_var = load()

# 6. SIDEBAR: FILTER DATA
st.sidebar.header("Filter data")
cities = sorted(df.city.unique())
certs = sorted(df.certificate.unique())
sc = st.sidebar.multiselect("Kota", cities, cities)
ss = st.sidebar.multiselect("Sertifikat", certs, certs)
pmin, pmax = float(df.price_miliar.min()), float(df.price_miliar.max())
lo, hi = st.sidebar.slider("Harga (Rp miliar)", pmin, pmax, (pmin, pmax))
# f = data yang sudah difilter. SEMUA grafik dan model di bawah memakai f
f = df[df.city.isin(sc) & df.certificate.isin(ss) & df.price_miliar.between(lo, hi)]
st.sidebar.metric("Observasi terpilih", f"{len(f):,}")

# 7. GAYA TAMPILAN (CSS): kalem, ringkas, kartu putih berlatar krem
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp p, .stApp li, .stApp label, [data-testid="stMarkdownContainer"] {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif; color: #26363A; }
.stApp { background: #F7F5F0; }
.block-container { padding-top: 4.2rem; max-width: 1200px; }
header[data-testid="stHeader"] { background: rgba(247,245,240,.92); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #2F4F4A !important; font-weight: 700; letter-spacing: -0.01em; }
.stApp h2 { font-size: 1.5rem; }
.stApp h3 { font-size: 1.15rem; }
section[data-testid="stSidebar"] { background: #EFEBE2; }
span[data-baseweb="tag"] { background-color: #4F7C74 !important; }
div[data-testid="stSlider"] div[role="slider"] { background-color: #4F7C74 !important; }
div[data-testid="stSlider"] div[style*="linear-gradient"] { background: #4F7C74 !important; }
div[data-testid="stMetric"] { background: #FFFFFF; border-left: 4px solid #4F7C74; border-radius: 10px;
    padding: 8px 12px; box-shadow: 0 1px 3px rgba(38,54,58,.08); }
div[data-testid="stPlotlyChart"] { background: #FFFFFF; border-radius: 12px; padding: 6px;
    box-shadow: 0 1px 3px rgba(38,54,58,.08); }
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
button[data-baseweb="tab"] { font-weight: 600; color: #7A8886; }
button[data-baseweb="tab"][aria-selected="true"] { color: #2F4F4A; }
[data-baseweb="tab-highlight"] { background-color: #D98B6A !important; height: 3px; }
[data-baseweb="tab-border"] { background-color: #E3DED3 !important; }
div[data-testid="stAlert"] { background: #EAF1EE !important; border-left: 4px solid #4F7C74; border-radius: 10px; color: #26363A; }
.top { background: linear-gradient(135deg, #4F7C74, #3B625B); border-radius: 14px; padding: 14px 20px;
       display: flex; flex-wrap: wrap; gap: 10px 20px; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.top .ttl { color: #FFFFFF; font-size: 1.3rem; font-weight: 700; line-height: 1.2; }
.top .sub { color: #CFE0DB; font-size: 0.74rem; margin-top: 3px; }
.top .chips { display: flex; flex-wrap: wrap; gap: 8px; }
.top .chip { background: rgba(255,255,255,.16); color: #FFFFFF; border-radius: 999px; padding: 4px 12px; font-size: 0.78rem; }
.top .chip b { color: #F6D9C6; }
</style>
""", unsafe_allow_html=True)

# Header ringkas satu baris. Angka diambil dari data asli (tidak ikut filter)
st.markdown(
    f'<div class="top"><div><div class="ttl">🏠 Harga Rumah Jabodetabek</div>'
    f'<div class="sub">Eksplorasi data dan regresi linear berganda · UTS Eksplorasi dan Visualisasi Data Multivariat · '
    f'Statistika, Universitas Akprind Indonesia</div></div>'
    f'<div class="chips"><span class="chip"><b>{len(df):,}</b> iklan</span>'
    f'<span class="chip"><b>{df.city.nunique()}</b> kota</span>'
    f'<span class="chip">median <b>Rp{df.price_miliar.median():.2f} M</b></span></div></div>',
    unsafe_allow_html=True)

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
    st.subheader("Latar belakang")
    st.write("Harga rumah di Jabodetabek sangat bervariasi. Pembeli, penjual, dan agen perlu tahu "
             "faktor apa yang paling terkait dengan harga: ukuran rumah, lokasi (kota), atau status sertifikat.")
    st.subheader("Tujuan analisis")
    st.markdown("- Menggambarkan karakteristik harga, luas tanah, luas bangunan, kota, dan sertifikat.\n"
                "- Mengeksplorasi hubungan antarvariabel (bivariat dan multivariat).\n"
                "- Membangun model regresi linear berganda untuk menjelaskan harga rumah.")
    st.subheader("Sumber data")
    st.write("Data iklan rumah dijual di Jabodetabek dari **Rumah123.com** (hasil scraping, dataset Kaggle "
             "*Jabodetabek House Price*). Harga adalah **harga penawaran**, bukan harga transaksi.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observasi awal", f"{log[0][1]:,}")
    c2.metric("Observasi setelah dibersihkan", f"{len(df):,}")
    c3.metric("Variabel di data asli", 27)
    c4.metric("Variabel dipakai", n_var)
    st.subheader("Definisi variabel")
    st.dataframe(pd.DataFrame([
        ("price_in_rp → Harga", "Numerik (respons)", "Harga penawaran rumah. Ditampilkan dalam Rp miliar; regresi memakai ln(Rp)"),
        ("land_size_m2 → Luas tanah", "Numerik", "Luas tanah (m²)"),
        ("building_size_m2 → Luas bangunan", "Numerik", "Luas bangunan (m²)"),
        ("city → Kota", "Kategorik (9 level)", "Kota/kabupaten lokasi rumah (Bogor, Depok, Tangerang, Bekasi, 5 wilayah Jakarta)"),
        ("certificate → Sertifikat", "Kategorik (4 level)", "SHM, HGB, Lainnya (PPJB/girik/adat/hak pakai), Tidak diketahui (kosong)")],
        columns=["Variabel", "Tipe", "Definisi"]), hide_index=True)
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
    st.subheader("Karakteristik variabel")
    st.dataframe(pd.DataFrame([
        ("Harga (Rp miliar)", "Numerik (rasio)", "Rata-rata, median, std, kuartil, skewness, kurtosis", "Histogram + boxplot"),
        ("Luas tanah (m²)", "Numerik (rasio)", "Rata-rata, median, std, kuartil, skewness, kurtosis", "Histogram + boxplot"),
        ("Luas bangunan (m²)", "Numerik (rasio)", "Rata-rata, median, std, kuartil, skewness, kurtosis", "Histogram + boxplot"),
        ("Kota", "Kategorik (nominal)", "Frekuensi, persentase, modus", "Diagram batang"),
        ("Sertifikat", "Kategorik (nominal)", "Frekuensi, persentase, modus", "Diagram lingkaran")],
        columns=["Variabel", "Tipe", "Ukuran statistik", "Visualisasi"]), hide_index=True)

    st.subheader("Statistik deskriptif variabel numerik")
    st.dataframe(pd.DataFrame({LBL[c]: desc(f[c]) for c in NUM}).T.round(2))

    st.subheader("Ringkasan variabel kategorik")

    def freq(col):
        t = f[col].value_counts().rename("Jumlah").to_frame()
        t["Persen (%)"] = (t["Jumlah"] / len(f) * 100).round(1)
        return t
    a, b = st.columns(2)
    a.dataframe(freq("city"))
    b.dataframe(freq("certificate"))
    st.caption(f"Modus kota: {f.city.mode()[0]}. Modus sertifikat: {f.certificate.mode()[0]}.")

    st.subheader("Visualisasi variabel numerik")
    logx = st.checkbox("Skala log pada sumbu x (semua grafik numerik)")
    for i, v in enumerate(NUM, 1):
        st.markdown(f"**Visualisasi {i}: Histogram + boxplot {LBL[v]}**")
        # marginal="box" menambah boxplot di atas histogram untuk melihat pencilan
        fig = px.histogram(f, x=v, nbins=50, marginal="box", labels=LBL, title=f"Distribusi {LBL[v]}")
        fig.update_layout(yaxis_title="Frekuensi")
        if logx:
            fig.update_xaxes(type="log")
        st.plotly_chart(fig, key=f"hist_{v}")
        s = f[v]
        q1, q3 = s.quantile([.25, .75])
        nout = int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())
        shape = "miring ke kanan (rata-rata > median)" if s.skew() > .5 else \
                "miring ke kiri" if s.skew() < -.5 else "relatif simetris"
        arti_s = ("sebagian besar rumah berada di nilai kecil sampai menengah, sedangkan sedikit rumah bernilai sangat besar menarik "
                  "rata-rata ke atas. Jadi median lebih mewakili rumah 'biasa' daripada rata-rata. Karena miring, regresi memakai ln "
                  "agar sebarannya lebih simetris." if s.skew() > .5 else
                  "sebagian besar rumah berada di nilai besar dan hanya sedikit di nilai kecil." if s.skew() < -.5 else
                  "rata-rata dan median hampir sama, jadi rata-rata cukup mewakili rumah pada umumnya.")
        interp(f"{LBL[v]} berdistribusi {shape}: rata-rata {s.mean():,.2f}, median {s.median():,.2f}, skewness {s.skew():.2f}. "
               f"Terdapat {nout} pencilan menurut aturan IQR ({nout / len(s):.1%} data), yaitu nilai yang jauh di atas kebanyakan rumah.",
               tujuan=f"bentuk sebaran {LBL[v]}: simetris atau miring, di mana kebanyakan rumah berada, dan adakah nilai ekstrem (pencilan). "
                      "Histogram menunjukkan bentuk sebaran, boxplot di atasnya menunjukkan median, kuartil, dan pencilan",
               arti=arti_s,
               out=f"histogram dan boxplot {LBL[v]}")

    st.subheader("Visualisasi kategorik")
    st.markdown("**Visualisasi 4: Diagram batang jumlah rumah per kota**")
    cc = f.city.value_counts().reset_index()
    cc.columns = ["city", "n"]
    st.plotly_chart(px.bar(cc, x="city", y="n", labels={"city": "Kota", "n": "Jumlah rumah"},
                           title="Jumlah rumah dijual per kota", text="n"))
    interp(f"Kota terbanyak adalah {cc.city[0]} ({cc.n[0]:,} rumah, {cc.n[0] / len(f):.1%}); paling sedikit {cc.city.iloc[-1]} "
           f"({cc.n.iloc[-1]:,} rumah). Sebaran antar kota tidak seimbang sehingga perbandingan antar kota perlu hati-hati.",
           tujuan="kota mana yang paling banyak dan paling sedikit datanya",
           arti="kota dengan data banyak hasilnya lebih dapat dipercaya, sedangkan kota dengan data sedikit estimasinya kurang stabil. "
                "Karena data adalah iklan yang tayang di situs, komposisi ini belum tentu mewakili semua rumah di Jabodetabek.",
           out="diagram batang jumlah rumah per kota")

    st.markdown("**Visualisasi 5: Diagram lingkaran komposisi sertifikat**")
    ct = f.certificate.value_counts().reset_index()
    ct.columns = ["certificate", "n"]
    st.plotly_chart(px.pie(ct, names="certificate", values="n",
                           labels={"certificate": "Sertifikat", "n": "Jumlah rumah"},
                           title="Komposisi rumah menurut jenis sertifikat"))
    interp(f"Sertifikat dominan: {ct.certificate[0]} ({ct.n[0] / len(f):.1%} data). Kategori lain jauh lebih kecil, "
           "jadi perbandingan harga antar sertifikat didasarkan pada jumlah rumah yang tidak seimbang.",
           tujuan="komposisi jenis sertifikat: berapa bagian rumah yang SHM, HGB, lainnya, atau tidak diketahui",
           arti="kalau satu jenis mendominasi, perbandingan harga antar sertifikat jadi tidak seimbang, "
                "dan kategori yang kecil kurang kuat dipakai untuk menarik kesimpulan.",
           out="diagram lingkaran komposisi sertifikat")

# TAB 3. EKSPLORASI BIVARIAT (Soal no. 3) - 2 variabel per grafik
with tabs[2]:
    st.header("Eksplorasi Bivariat")
    st.subheader("Visualisasi 1: Scatter plot harga vs luas (numerik–numerik)")
    xv = st.selectbox("Variabel x", ["building_size_m2", "land_size_m2"], format_func=LBL.get)
    lg = st.checkbox("Skala log-log", value=True)
    # trendline="ols" = garis regresi; sumbu log supaya titik tidak menumpuk di kiri bawah
    st.plotly_chart(px.scatter(f, x=xv, y="price_miliar", opacity=.4, trendline="ols",
                               trendline_options=dict(log_x=lg, log_y=lg), log_x=lg, log_y=lg,
                               labels=LBL, title=f"Harga vs {LBL[xv]}"))
    # Korelasi Pearson (hubungan linear) dan Spearman (hubungan berurutan, tahan pencilan)
    pr = f[[xv, "price_miliar"]].corr().iloc[0, 1]
    sp = f[[xv, "price_miliar"]].corr(method="spearman").iloc[0, 1]
    kuat_r = "kuat" if abs(pr) >= .7 else "sedang" if abs(pr) >= .4 else "lemah"
    arah = "positif" if pr > 0 else "negatif"
    interp(f"Hubungan {arah}: korelasi Pearson {pr:.2f}, Spearman {sp:.2f}. Semakin besar {LBL[xv].lower()}, harga cenderung "
           f"{'naik' if pr > 0 else 'turun'}, tetapi sebaran lebar, artinya ukuran saja belum cukup menjelaskan harga.",
           tujuan="apakah ada hubungan antara ukuran rumah dan harga, serta arah dan kekuatannya. Garis tren menunjukkan pola rata-rata; "
                  "Pearson mengukur hubungan garis lurus, Spearman mengukur hubungan urutan (tahan pencilan)",
           arti=f"hubungannya {kuat_r} dan {'searah' if pr > 0 else 'berlawanan arah'}: rumah yang lebih luas cenderung "
                f"{'lebih mahal' if pr > 0 else 'lebih murah'}. Titik yang menyebar jauh dari garis berarti ukuran bukan satu-satunya penentu harga "
                "(lokasi dan kondisi rumah juga berperan). Korelasi hanya menunjukkan keterkaitan, bukan bahwa luas menyebabkan harga naik.",
           out="scatter plot harga vs luas beserta garis tren dan korelasi Pearson/Spearman")

    st.subheader("Visualisasi 2: Boxplot harga per kota (numerik–kategorik)")
    ly = st.checkbox("Skala log pada harga", value=True, key="ly2")
    order = f.groupby("city").price_miliar.median().sort_values().index.tolist()
    st.plotly_chart(px.box(f, x="city", y="price_miliar", color="city", log_y=ly, category_orders={"city": order},
                           labels=LBL, title="Distribusi harga per kota (urut median)"))
    md = f.groupby("city").price_miliar.median().sort_values()
    interp(f"Median harga tertinggi di {md.index[-1]} (Rp{md.iloc[-1]:.2f} miliar) dan terendah di {md.index[0]} "
           f"(Rp{md.iloc[0]:.2f} miliar), selisih {md.iloc[-1] / md.iloc[0]:.1f} kali. Lokasi sangat membedakan harga.",
           tujuan="apakah harga berbeda antar kota: kotak = 50% harga tengah, garis dalam kotak = median, titik di luar = pencilan",
           arti="kalau median dan posisi kotak antar kota berbeda jauh, lokasi berkaitan kuat dengan harga. "
                "Kotak yang tinggi atau lebar berarti harga di kota itu sangat bervariasi. Inilah alasan kota dimasukkan sebagai prediktor di regresi.",
           out="boxplot harga per kota")

    st.subheader("Visualisasi 3: Boxplot harga per sertifikat (numerik–kategorik)")
    st.plotly_chart(px.box(f, x="certificate", y="price_miliar", color="certificate", log_y=True,
                           labels=LBL, title="Distribusi harga per jenis sertifikat"))
    mc = f.groupby("certificate").price_miliar.median().sort_values()
    interp("Median harga per sertifikat: " + ", ".join(f"{k} Rp{v:.2f} miliar" for k, v in mc.items()) +
           ". Perbedaan ini belum mengontrol ukuran dan kota, jadi diuji ulang di regresi.",
           tujuan="apakah harga berbeda menurut jenis sertifikat",
           arti="kalau median antar sertifikat berbeda, sertifikat tampak berkaitan dengan harga. Tetapi bisa jadi itu karena jenis sertifikat "
                "tertentu lebih banyak di kota mahal atau rumah besar, jadi belum bisa disimpulkan sebagai pengaruh sertifikat itu sendiri. "
                "Hal ini dicek di regresi.",
           out="boxplot harga per jenis sertifikat")

    st.subheader("Visualisasi 4: Komposisi sertifikat per kota (kategorik–kategorik)")
    # barnorm="percent": tiap batang jadi 100% supaya proporsi antar kota bisa dibandingkan
    st.plotly_chart(px.histogram(f, x="city", color="certificate", barnorm="percent", text_auto=".1f",
                                 category_orders={"city": order},
                                 labels={"city": "Kota", "certificate": "Sertifikat"},
                                 title="Komposisi jenis sertifikat per kota (%)")
                    .update_layout(yaxis_title="Persentase (%)"))
    tab = pd.crosstab(f.city, f.certificate)               # tabel silang kota x sertifikat
    if tab.shape[0] > 1 and tab.shape[1] > 1:
        # Chi-square: apakah jenis sertifikat berhubungan dengan kota. Cramer's V = kekuatan hubungan (0 sampai 1)
        chi2, pchi, dof, _ = stats.chi2_contingency(tab)
        cv = float(np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1))))
        kuat = "lemah" if cv < .1 else "sedang" if cv < .3 else "kuat"
        if "SHM" in tab.columns:
            shm = (tab["SHM"] / tab.sum(axis=1) * 100).sort_values()
            txt = f"Proporsi SHM tertinggi di {shm.index[-1]} ({shm.iloc[-1]:.1f}%) dan terendah di {shm.index[0]} ({shm.iloc[0]:.1f}%). "
        else:
            txt = ""
        arti_c = (f"komposisi sertifikat berbeda antar kota, jadi kota dan jenis sertifikat tidak sepenuhnya terpisah. "
                  f"Karena V = {cv:.2f} (hubungan {kuat}), perbedaannya {'cukup nyata' if cv >= .3 else 'ada tetapi kecil'}."
                  if pchi < .05 else
                  "tidak cukup bukti bahwa komposisi sertifikat berbeda antar kota, jadi sebaran sertifikatnya relatif mirip di semua kota.")
        interp(txt + f"Uji chi-square p {pf(pchi)} ({'ada' if pchi < .05 else 'tidak ada'} hubungan kota dengan jenis sertifikat); "
               f"Cramér's V = {cv:.2f} (hubungan {kuat}). Catatan: sebagian sel berisi sedikit rumah, jadi hasil uji perlu hati-hati.",
               tujuan="apakah jenis sertifikat berhubungan dengan kota (kategorik vs kategorik). H0: tidak ada hubungan. "
                      "Cramér's V mengukur kekuatan hubungan dari 0 (tidak ada) sampai 1 (sangat kuat)",
               arti=arti_c,
               out="diagram batang bertumpuk dan uji chi-square")
    else:
        st.info("Pilih minimal 2 kota dan 2 jenis sertifikat di sidebar untuk melihat hubungan kategorik–kategorik.")

# TAB 4. EKSPLORASI MULTIVARIAT (Soal no. 4) - minimal 3 variabel per grafik
with tabs[3]:
    st.header("Eksplorasi Multivariat")
    st.subheader("Visualisasi 1: Bubble plot luas tanah × luas bangunan × harga × kota")
    lg2 = st.checkbox("Skala log-log", value=True, key="lg2")
    # x = luas tanah, y = luas bangunan, warna = kota, ukuran titik = harga
    st.plotly_chart(px.scatter(f, x="land_size_m2", y="building_size_m2", color="city", size="price_miliar",
                               size_max=18, opacity=.5, log_x=lg2, log_y=lg2, labels=LBL,
                               title="Luas tanah vs luas bangunan (warna = kota, ukuran = harga)"))
    rr = f[["ln_land", "ln_bld"]].corr().iloc[0, 1]    # korelasi dua luas (skala log)
    interp(f"Luas tanah dan luas bangunan berkorelasi kuat (r = {rr:.2f} pada skala log), sehingga titik membentuk pita naik. "
           "Titik besar (harga tinggi) menumpuk di area luas besar, dan warna kota terlihat berkelompok di area tertentu.",
           tujuan="hubungan empat variabel sekaligus: luas tanah (sumbu x), luas bangunan (sumbu y), kota (warna), dan harga (ukuran titik)",
           arti="korelasi yang tinggi antara dua luas berarti keduanya membawa informasi yang mirip, sehingga di regresi bisa terjadi "
                "multikolinearitas (dicek lewat VIF). Titik besar yang menumpuk di area luas besar menunjukkan harga naik bersama ukuran, "
                "dan warna yang berkelompok menunjukkan lokasi punya pola ukuran dan harga sendiri.",
           out="bubble plot luas tanah, luas bangunan, harga, dan kota")

    st.subheader("Visualisasi 2: Harga vs luas bangunan per kota dan sertifikat")
    # facet_col="city" memecah grafik per kota (panel kecil), warna = sertifikat
    st.plotly_chart(px.scatter(f, x="building_size_m2", y="price_miliar", color="certificate", facet_col="city",
                               facet_col_wrap=3, log_x=True, log_y=True, opacity=.5, height=800, labels=LBL,
                               title="Harga vs luas bangunan, dipisah per kota (warna = sertifikat)"))
    # Harga per m² bangunan (Rp juta) = harga(miliar)*1000 / luas bangunan
    ppm = (f.price_miliar * 1000 / f.building_size_m2).groupby(f.city).median().sort_values()
    interp(f"Di setiap kota hubungan harga–luas bangunan tetap positif, tetapi posisi vertikal kelompok berbeda: median harga per m² "
           f"bangunan tertinggi di {ppm.index[-1]} (Rp{ppm.iloc[-1]:.1f} juta) dan terendah di {ppm.index[0]} (Rp{ppm.iloc[0]:.1f} juta). "
           "Artinya pada luas yang sama, lokasi menentukan harga.",
           tujuan="apakah hubungan harga dan luas bangunan sama di setiap kota, sambil membandingkan tingkat harga antar kota pada luas yang sama",
           arti="kalau garis tiap kota sejajar tetapi posisinya beda tinggi, lokasi menggeser harga ke atas atau bawah tanpa mengubah efek luas, "
                "dan itulah yang ditangkap dummy kota di regresi. Kalau kemiringannya berbeda-beda, efek luas bergantung pada kota sehingga perlu suku interaksi.",
           out="scatter plot harga vs luas bangunan per kota")

    st.subheader("Visualisasi 3: Heatmap korelasi variabel numerik (skala log)")
    cm = f[["ln_price", "ln_land", "ln_bld"]].corr()          # matriks korelasi Pearson
    cm.index = cm.columns = ["ln(Harga)", "ln(Luas tanah)", "ln(Luas bangunan)"]
    st.plotly_chart(px.imshow(cm, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                              labels=dict(color="Korelasi"), title="Matriks korelasi Pearson"))
    interp(f"ln(Harga) berkorelasi kuat dengan ln(Luas bangunan) (r = {cm.iloc[0, 2]:.2f}) dan ln(Luas tanah) (r = {cm.iloc[0, 1]:.2f}); "
           f"kedua luas saling berkorelasi (r = {cm.iloc[1, 2]:.2f}), sehingga multikolinearitas perlu dicek lewat VIF.",
           tujuan="korelasi semua pasangan variabel numerik sekaligus. Merah berarti searah, biru berlawanan arah, makin pekat makin kuat",
           arti="ln(Harga) berkorelasi tinggi dengan kedua luas, berarti keduanya layak jadi prediktor. Korelasi tinggi antara luas tanah dan "
                "luas bangunan menjadi peringatan bahwa koefisien keduanya bisa kurang stabil.",
           out="heatmap korelasi variabel numerik")

# TAB 5. REGRESI LINEAR BERGANDA (Soal no. 5)
with tabs[4]:
    st.header("Regresi Linear Berganda")

    # MEMBANGUN MODEL (setara lm() di R). Kategori acuan dummy: kota terbanyak dan sertifikat SHM
    ref = f.city.value_counts().idxmax()
    refc = "SHM" if "SHM" in set(f.certificate) else f.certificate.value_counts().idxmax()
    terms = ["ln_land", "ln_bld"]
    if f.city.nunique() > 1:
        terms.append(f"C(city, Treatment(reference='{ref}'))")          # C() = ubah kategori jadi dummy
    if f.certificate.nunique() > 1:
        terms.append(f"C(certificate, Treatment(reference='{refc}'))")
    m = smf.ols("ln_price ~ " + " + ".join(terms), f).fit()

    # nice(): ubah nama koefisien teknis jadi nama yang mudah dibaca
    def nice(n):
        if n == "Intercept": return "Intercept"
        if n == "ln_land": return "ln(Luas tanah)"
        if n == "ln_bld": return "ln(Luas bangunan)"
        lv = n.split("[T.")[1].rstrip("]")
        return f"Kota: {lv} (vs {ref})" if n.startswith("C(city") else f"Sertifikat: {lv} (vs {refc})"

    st.subheader("Pemilihan variabel")
    st.markdown("- **Respons**: ln(Harga). Harga miring ke kanan; log menstabilkan ragam dan koefisien dapat dibaca sebagai persentase.\n"
                "- **Prediktor**: ln(Luas tanah), ln(Luas bangunan) (ukuran fisik), Kota dan Sertifikat (dummy). "
                f"Kategori acuan: Kota = {ref}, Sertifikat = {refc}.")

    # HASIL ESTIMASI (setara summary(model) di R)
    names = [nice(k) for k in m.params.index]
    tb = pd.DataFrame({"Koefisien": m.params.values, "Std. error": m.bse.values, "t": m.tvalues.values,
                       "p-value": m.pvalues.values}, index=names)
    # Efek persen untuk dummy: (exp(koef) - 1) * 100
    tb["Efek harga (%)"] = [np.nan if n in ("Intercept", "ln(Luas tanah)", "ln(Luas bangunan)")
                            else (np.exp(b) - 1) * 100 for n, b in zip(names, m.params.values)]
    st.subheader("Hasil estimasi")
    st.dataframe(tb.round(4))
    rmse = float(np.sqrt(np.mean(m.resid ** 2)))     # RMSE: kesalahan prediksi (skala ln)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("R²", f"{m.rsquared:.3f}"); k2.metric("R² adj.", f"{m.rsquared_adj:.3f}")
    k3.metric("F-statistik", f"{m.fvalue:,.1f}"); k4.metric("p-value F", pf(m.f_pvalue).replace("= ", ""))
    k5.metric("RMSE (skala ln)", f"{rmse:.3f}")

    st.subheader("Persamaan model")
    eq = f"ln(Harga) = {m.params['Intercept']:.3f}"
    for k, v in m.params.items():
        if k != "Intercept":
            eq += f"\n   {'+' if v >= 0 else '-'} {abs(v):.3f} × [{nice(k)}]"
    st.code(eq)

    # INTERPRETASI KOEFISIEN DAN UJI HIPOTESIS
    bl, bb = m.params["ln_land"], m.params["ln_bld"]
    sig = int((m.pvalues.drop("Intercept") < .05).sum())
    st.subheader("Interpretasi koefisien dan pengujian")
    st.markdown(f"- **Luas tanah**: naik 1% → harga naik sekitar {bl:.2f}% (kota, sertifikat, luas bangunan tetap), "
                f"{'signifikan' if m.pvalues['ln_land'] < .05 else 'tidak signifikan'} (p {pf(m.pvalues['ln_land'])}). "
                f"*Artinya*: model log-log, jadi koefisien dibaca sebagai elastisitas; "
                f"{'tanah yang lebih luas berkaitan nyata dengan harga yang lebih tinggi' if m.pvalues['ln_land'] < .05 else 'luas tanah tidak berkaitan nyata dengan harga setelah ukuran bangunan dan lokasi diperhitungkan'}.\n"
                f"- **Luas bangunan**: naik 1% → harga naik sekitar {bb:.2f}%, "
                f"{'signifikan' if m.pvalues['ln_bld'] < .05 else 'tidak signifikan'} (p {pf(m.pvalues['ln_bld'])}). "
                f"*Artinya*: {'bangunan yang lebih luas berkaitan nyata dengan harga yang lebih tinggi' if m.pvalues['ln_bld'] < .05 else 'luas bangunan tidak berkaitan nyata dengan harga pada model ini'}"
                f"{'; karena koefisiennya lebih besar dari luas tanah, harga lebih responsif terhadap luas bangunan.' if (m.pvalues['ln_bld'] < .05 and bb > bl) else '.'}\n"
                "- **Dummy kategorik**: kolom *Efek harga (%)* = selisih harga rata-rata (geometrik) dibanding kategori acuan, prediktor lain tetap. "
                "*Artinya*: angka +30% berarti rumah di kota itu rata-rata 30% lebih mahal daripada kota acuan pada ukuran dan sertifikat yang sama; angka negatif berarti lebih murah.\n"
                f"- **Uji simultan F**: p {pf(m.f_pvalue)} → {'minimal satu' if m.f_pvalue < .05 else 'tidak ada'} prediktor berpengaruh pada α = 5%. "
                f"*Artinya*: {'model secara keseluruhan bermakna dan lebih baik daripada tanpa prediktor' if m.f_pvalue < .05 else 'model tidak lebih baik daripada tanpa prediktor'}.\n"
                f"- **Uji parsial t**: {sig} dari {len(m.params) - 1} koefisien signifikan pada α = 5%. "
                "*Artinya*: tiap variabel yang signifikan memberi sumbangan sendiri pada harga, bukan hanya ikut-ikutan variabel lain.")

    # UJI ASUMSI RESIDUAL
    X = m.model.exog
    # VIF: multikolinearitas (VIF >= 5 berarti kolinearitas tinggi)
    vf = {nm: vif(X, i) for i, nm in enumerate(m.model.exog_names) if nm in ("ln_land", "ln_bld")}
    res = np.asarray(m.resid)                              # residual = aktual - prediksi
    bp = het_breuschpagan(res, X)[1]                       # Breusch-Pagan (heteroskedastisitas)
    jb = jarque_bera(res)[1]                               # Jarque-Bera (normalitas)
    sw = stats.shapiro(res)[1]                             # Shapiro-Wilk
    ks = stats.kstest(res, stats.norm(res.mean(), res.std()).cdf)[1]   # Kolmogorov-Smirnov
    # Glejser: regresikan |residual| pada prediktor
    gl = smf.ols("ar ~ ln_land + ln_bld", f.assign(ar=np.abs(res))).fit().f_pvalue
    cd = m.get_influence().cooks_distance[0]               # Cook's distance

    st.subheader("Evaluasi model")
    st.markdown(f"- R² = {m.rsquared:.3f}: model menjelaskan {m.rsquared * 100:.1f}% variasi ln(harga); R² adj. = {m.rsquared_adj:.3f}. "
                "*Artinya*: sisanya dijelaskan faktor yang tidak ada di data (kondisi rumah, jumlah kamar, kecamatan).\n"
                f"- RMSE = {rmse:.3f} (skala ln) ≈ kesalahan prediksi harga sekitar {(np.exp(rmse) - 1) * 100:.0f}%. "
                "*Artinya*: prediksi model rata-rata meleset sebesar persentase itu dari harga sebenarnya.\n"
                f"- VIF ln(Luas tanah) = {vf['ln_land']:.2f}, ln(Luas bangunan) = {vf['ln_bld']:.2f} "
                f"({'multikolinearitas rendah-sedang' if max(vf.values()) < 5 else 'multikolinearitas tinggi (≥ 5)'}). "
                f"*Artinya*: {'kedua luas tidak terlalu tumpang tindih, jadi koefisiennya masih stabil.' if max(vf.values()) < 5 else 'kedua luas terlalu tumpang tindih, jadi koefisiennya tidak stabil.'}")

    st.subheader("Uji asumsi residual")
    # Aturan keputusan: p-value >= 0.05 berarti asumsi terpenuhi (H0 tidak ditolak)
    ok = lambda p: "Terpenuhi" if p >= .05 else "Tidak terpenuhi"
    st.dataframe(pd.DataFrame([
        ("Normalitas", "Shapiro–Wilk", pf(sw).replace("= ", ""), ok(sw)),
        ("Normalitas", "Kolmogorov–Smirnov", pf(ks).replace("= ", ""), ok(ks)),
        ("Normalitas", "Jarque–Bera", pf(jb).replace("= ", ""), ok(jb)),
        ("Homoskedastisitas", "Breusch–Pagan", pf(bp).replace("= ", ""), ok(bp)),
        ("Homoskedastisitas", "Glejser (|residual| ~ prediktor)", pf(gl).replace("= ", ""), ok(gl))],
        columns=["Asumsi", "Uji", "p-value", "Kesimpulan (α = 5%)"]), hide_index=True)
    st.caption("Autokorelasi (Durbin–Watson, ACF) tidak diuji karena data cross-section: baris iklan tidak berurutan waktu.")
    st.markdown("**Cara membaca tabel ini**\n"
                "- *Residual* = selisih antara harga sebenarnya dan harga prediksi model (dalam skala ln).\n"
                "- *Normalitas*: apakah residual berbentuk lonceng (normal). *Homoskedastisitas*: apakah sebaran residual sama lebar di semua tingkat harga.\n"
                "- Aturannya sama untuk semua uji: **p-value ≥ 0,05 berarti asumsi terpenuhi**, **p-value < 0,05 berarti tidak terpenuhi**.")
    normal_ok, homo_ok = min(sw, ks, jb) >= .05, min(bp, gl) >= .05
    interp(f"Normalitas residual {'terpenuhi' if normal_ok else 'tidak terpenuhi'} dan homoskedastisitas "
           f"{'terpenuhi' if homo_ok else 'tidak terpenuhi'} pada α = 5%. "
           + ("" if normal_ok and homo_ok else "Dengan n besar koefisien tetap konsisten, tetapi uji t dan p-value perlu dibaca hati-hati "
              "(uji juga sangat sensitif pada n besar)."),
           tujuan="apakah model OLS memenuhi asumsinya (residual normal dan ragamnya konstan). Kalau terpenuhi, uji t, uji F, dan selang kepercayaan sahih",
           arti=("kedua asumsi terpenuhi, jadi p-value dan selang kepercayaan bisa dipercaya apa adanya."
                 if normal_ok and homo_ok else
                 "ada asumsi yang tidak terpenuhi (p < 0,05). Koefisien umumnya masih tidak bias, tetapi p-value dan selang kepercayaan bisa kurang tepat. "
                 "Karena itu hasilnya dicek ulang dengan HC3 di bawah. Dengan data yang banyak, uji juga sangat sensitif sehingga penyimpangan kecil pun ditolak."),
           out="uji asumsi residual (Shapiro–Wilk, Kolmogorov–Smirnov, Jarque–Bera, Breusch–Pagan, Glejser)")

    # Upaya memperbaiki asumsi: coba beberapa transformasi lalu bandingkan
    st.subheader("Upaya memperbaiki asumsi: perbandingan transformasi")
    st.markdown("Karena sebagian asumsi tidak terpenuhi, kita mencoba **mengubah bentuk data (transformasi)** untuk melihat "
                "apakah residual menjadi lebih rapi. Setiap baris di tabel adalah satu cara yang dicoba.\n\n"
                "**Cara membaca kolom**\n"
                "- **R²**: seberapa baik model menjelaskan harga (0 sampai 1, makin besar makin baik).\n"
                "- **Skewness residual**: kemiringan residual. Makin dekat 0 makin simetris.\n"
                "- **Kurtosis residual**: ketebalan ekor residual. Makin dekat 0 makin sedikit nilai ekstrem.\n"
                "- **p Shapiro–Wilk**: uji normalitas. Lolos jika p ≥ 0,05.\n"
                "- **p Breusch–Pagan**: uji kesamaan sebaran residual. Lolos jika p ≥ 0,05.")
    g = f.copy()
    g["bc_y"], lam = stats.boxcox(g.price_miliar)       # Box-Cox: pangkat terbaik untuk harga
    g["sq_y"] = np.sqrt(g.price_miliar)
    g["ln_bld2"] = g.ln_bld ** 2
    g["ln_land2"] = g.ln_land ** 2
    kat = [t for t in terms if t.startswith("C(")]       # bagian kota dan sertifikat tetap sama
    kat = (" + " + " + ".join(kat)) if kat else ""
    kandidat = [
        ("ln(harga) ~ ln(luas)  [dipakai]", "ln_price ~ ln_land + ln_bld"),
        ("Box-Cox(harga) ~ ln(luas)", "bc_y ~ ln_land + ln_bld"),
        ("akar(harga) ~ ln(luas)", "sq_y ~ ln_land + ln_bld"),
        ("ln(harga) ~ luas (tanpa log)", "ln_price ~ land_size_m2 + building_size_m2"),
        ("ln(harga) ~ ln(luas) + kuadratnya", "ln_price ~ ln_land + ln_bld + ln_land2 + ln_bld2")]
    baris, lolos = [], False
    for nama, rumus in kandidat:
        mm = smf.ols(rumus + kat, g).fit()
        rr_ = np.asarray(mm.resid)
        sw_, bp_ = stats.shapiro(rr_)[1], het_breuschpagan(rr_, mm.model.exog)[1]
        lolos = lolos or (sw_ >= .05 and bp_ >= .05)
        baris.append((nama, round(mm.rsquared, 3), round(float(stats.skew(rr_)), 2), round(float(stats.kurtosis(rr_)), 2),
                      pf(sw_).replace("= ", ""), pf(bp_).replace("= ", "")))
    st.dataframe(pd.DataFrame(baris, columns=["Model", "R²", "Skewness residual", "Kurtosis residual",
                                              "p Shapiro–Wilk", "p Breusch–Pagan"]), hide_index=True)
    st.caption("R² antar model dengan skala respons berbeda tidak sebanding langsung; bandingkan terutama kondisi residualnya.")
    st.markdown("**Artinya untuk analisis ini**\n"
                + ("- Ada cara yang membuat kedua uji lolos (p ≥ 0,05); pertimbangkan memakai model tersebut.\n" if lolos else
                   "- Semua cara masih gagal pada kedua uji (p < 0,05), jadi tidak ada transformasi yang memperbaiki asumsi secara formal.\n")
                + "- Model **ln(harga) ~ ln(luas)** tetap dipakai karena sederhana dan koefisiennya bisa dibaca sebagai persentase perubahan harga.\n"
                "- Penyebab yang paling mungkin: ada faktor penentu harga yang tidak ada di data (kecamatan, kondisi rumah, jumlah kamar). "
                "Selain itu, dengan data yang banyak, uji statistik sangat sensitif sehingga penyimpangan kecil pun ditolak.\n"
                "- Hal ini dicatat sebagai keterbatasan model, dan kesimpulan diperiksa lagi dengan robust standard error di bawah.")
    interp(f"Box-Cox menyarankan λ = {lam:.2f}; "
           f"{'nilai mendekati 0 berarti transformasi log memang pilihan yang tepat untuk harga. ' if abs(lam) < .3 else 'nilai ini berbeda dari 0 sehingga transformasi lain patut dipertimbangkan. '}"
           + ("Ada model kandidat yang memenuhi kedua asumsi." if lolos else
              "Tidak ada kandidat yang membuat normalitas dan homoskedastisitas sama-sama terpenuhi, "
              "sehingga model ln(harga) tetap dipakai dan keterbatasannya dijelaskan."),
           tujuan="pangkat transformasi terbaik untuk harga supaya residual lebih rapi. λ ≈ 0 setara log, λ ≈ 0,5 setara akar, λ ≈ 1 berarti tanpa transformasi",
           arti=("λ dekat 0 berarti memakai ln(harga) sudah tepat." if abs(lam) < .3 else
                 "λ jauh dari 0 berarti ada transformasi lain yang mungkin lebih cocok daripada log."),
           out="transformasi Box-Cox dan tabel perbandingan transformasi")

    # Robust standard error (HC3): p-value tetap sahih walau ragam residual tidak konstan
    st.subheader("Pemeriksaan ketahanan: robust standard error (HC3)")
    st.markdown("HC3 adalah cara menghitung p-value yang tetap sahih walau sebaran residual tidak merata. "
                "**Cara membaca**: bandingkan kolom kiri (cara biasa) dengan kolom kanan (HC3). "
                "Jika keduanya sama-sama di bawah 0,05 atau sama-sama di atas 0,05, kesimpulan tidak berubah dan hasil regresi dapat dipercaya.")
    h = smf.ols("ln_price ~ " + " + ".join(terms), f).fit(cov_type="HC3")
    st.dataframe(pd.DataFrame({"p-value biasa": m.pvalues.values, "p-value robust (HC3)": h.pvalues.values},
                              index=names).round(4))
    sama = bool(((m.pvalues < .05) == (h.pvalues < .05)).all())
    sig_h = int((h.pvalues.drop("Intercept") < .05).sum())
    interp(f"Dengan HC3, {sig_h} dari {len(m.params) - 1} koefisien signifikan (sebelumnya {sig}). "
           + ("Keputusan uji tidak berubah pada koefisien mana pun, jadi kesimpulan regresi tahan terhadap ragam residual yang tidak konstan."
              if sama else "Ada koefisien yang keputusannya berubah, jadi p-value biasa perlu dibaca hati-hati."),
           tujuan="apakah kesimpulan signifikan atau tidak berubah jika ragam residual tidak konstan",
           arti=("keputusan sama di semua koefisien, jadi kesimpulan regresi kuat dan tidak bergantung pada asumsi ragam konstan."
                 if sama else
                 "ada koefisien yang keputusannya berubah, jadi koefisien itu belum bisa dianggap pasti signifikan dan harus dibaca hati-hati."),
           out="perbandingan p-value biasa dan robust (HC3)")

    # VISUALISASI REGRESI (6 grafik)
    st.subheader("Visualisasi regresi 1: Nilai aktual vs prediksi")
    fv = np.asarray(m.fittedvalues)
    ac = f.ln_price.values
    fig = px.scatter(x=fv, y=ac, opacity=.4, labels={"x": "ln(Harga) prediksi", "y": "ln(Harga) aktual"},
                     title="Aktual vs prediksi")
    fig.add_trace(go.Scatter(x=[ac.min(), ac.max()], y=[ac.min(), ac.max()], mode="lines",
                             name="Garis y = x", line=dict(dash="dash", color="#D98B6A")))
    st.plotly_chart(fig)
    interp(f"Titik mengikuti garis y = x dengan R² = {m.rsquared:.2f}; makin dekat garis, makin akurat prediksi. "
           "Sebaran di sekitar garis menunjukkan galat yang masih ada.",
           tujuan="seberapa dekat prediksi model dengan harga sebenarnya. Titik tepat di garis putus-putus berarti prediksi akurat",
           arti=f"R² = {m.rsquared:.2f} berarti sekitar {m.rsquared * 100:.0f}% variasi harga terjelaskan model, sisanya karena faktor yang tidak ada di data. "
                "Titik di atas garis = harga sebenarnya lebih mahal dari prediksi; titik di bawah garis = lebih murah dari prediksi.",
           out="grafik nilai aktual vs prediksi")

    st.subheader("Visualisasi regresi 2: Residual vs nilai prediksi")
    fig = px.scatter(x=fv, y=res, opacity=.4,
                     labels={"x": "ln(Harga) prediksi", "y": "Residual"}, title="Residual vs prediksi")
    fig.add_hline(y=0, line_dash="dash", line_color="#D98B6A")
    st.plotly_chart(fig)
    interp(f"Residual seharusnya acak di sekitar nol tanpa pola. Uji Breusch–Pagan p {pf(bp)} "
           f"({'ada indikasi sebaran residual tidak merata' if bp < .05 else 'tidak menunjukkan masalah'}).",
           tujuan="linearitas dan kekonstanan ragam. Pola yang diharapkan adalah awan titik acak, rata, di sekitar garis nol",
           arti=("tidak ada pola jelas, jadi hubungan linear dan ragam residual cukup konstan."
                 if bp >= .05 else
                 "pola melebar atau menyempit seperti corong menandakan ragam residual tidak konstan (heteroskedastisitas), artinya ketepatan "
                 "prediksi berbeda di tiap tingkat harga. Pola melengkung berarti hubungannya tidak sepenuhnya linear."),
           out="grafik residual vs nilai prediksi")

    st.subheader("Visualisasi regresi 3: Histogram residual")
    st.plotly_chart(px.histogram(x=res, nbins=40, labels={"x": "Residual"}, title="Distribusi residual")
                    .update_layout(yaxis_title="Frekuensi"))
    interp(f"Residual berpusat di sekitar nol (rata-rata {res.mean():.3f}), skewness {stats.skew(res):.2f}, kurtosis berlebih {stats.kurtosis(res):.2f}. "
           f"{'Bentuknya mendekati lonceng.' if normal_ok else 'Ekor lebih tebal dari kurva normal (banyak residual ekstrem).' if stats.kurtosis(res) > 1 else 'Bentuknya agak menyimpang dari normal.'}",
           tujuan="apakah residual berbentuk lonceng simetris di sekitar nol (asumsi normalitas)",
           arti=("bentuknya mendekati lonceng, jadi residual relatif normal."
                 if normal_ok else
                 "bentuk menyimpang dari lonceng. Ekor yang tebal berarti ada rumah yang harganya meleset jauh dari prediksi, "
                 "kemungkinan karena faktor yang tidak terukur (kondisi rumah, kecamatan)."),
           out="histogram residual")

    st.subheader("Visualisasi regresi 4: Q-Q plot residual")
    (osm, osr), (qs, qi, _) = stats.probplot(res, dist="norm")
    fig = px.scatter(x=osm, y=osr, opacity=.5, labels={"x": "Kuantil teoretis (normal)", "y": "Kuantil residual"},
                     title="Q-Q plot residual")
    fig.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[qi + qs * osm.min(), qi + qs * osm.max()], mode="lines",
                             name="Garis normal", line=dict(color="#D98B6A")))
    st.plotly_chart(fig)
    interp(f"Titik {'mengikuti garis' if normal_ok else 'menjauhi garis di kedua ujung'}: "
           f"{'residual mendekati normal.' if normal_ok else 'ekor distribusi lebih berat dari normal, sejalan dengan uji Shapiro–Wilk.'}",
           tujuan="apakah residual berdistribusi normal (cara lain memeriksa normalitas): kalau normal, titik menempel pada garis",
           arti=("titik mengikuti garis, jadi residual mendekati normal."
                 if normal_ok else
                 "titik menjauh di ujung kiri dan kanan, artinya ada lebih banyak nilai ekstrem daripada yang diharapkan pada distribusi normal."),
           out="Q-Q plot residual")

    st.subheader("Visualisasi regresi 5: Cook's distance (pengamatan berpengaruh)")
    fig = px.scatter(x=np.arange(len(cd)), y=cd, opacity=.5,
                     labels={"x": "Indeks pengamatan", "y": "Cook's distance"}, title="Cook's distance")
    fig.add_hline(y=4 / len(cd), line_dash="dash", line_color="#D98B6A")   # batas umum 4/n
    st.plotly_chart(fig)
    interp(f"Nilai maksimum Cook's distance {cd.max():.3f} ({'di bawah' if cd.max() < 1 else 'di atas'} 1, batas umum pengamatan sangat berpengaruh); "
           f"{int((cd > 4 / len(cd)).sum())} pengamatan melewati batas 4/n (garis putus-putus) sehingga perlu dicermati"
           f"{', tetapi tidak ada yang mendominasi model' if cd.max() < 1 else ', dan ada pengamatan yang sangat berpengaruh'}.",
           tujuan="pengamatan yang terlalu berpengaruh pada hasil model (kalau dibuang, koefisien bisa berubah banyak)",
           arti=("tidak ada satu pun pengamatan yang mendominasi model. Titik yang melewati garis hanya perlu dicermati, belum tentu harus dibuang."
                 if cd.max() < 1 else
                 "ada pengamatan yang sangat berpengaruh, jadi perlu dicek apakah itu data salah input atau rumah yang memang unik, "
                 "lalu model dibandingkan dengan dan tanpa titik itu."),
           out="grafik Cook's distance")

    st.subheader("Visualisasi regresi 6: Forest plot koefisien regresi (CI 95%)")
    ci = m.conf_int()
    fp = pd.DataFrame({
        "Variabel": [nice(k) for k in ci.index],
        "Koef": m.params.values,
        "err_plus": ci[1].values - m.params.values,
        "err_minus": m.params.values - ci[0].values
    }).sort_values("Koef")
    fig_fp = px.scatter(fp, x="Koef", y="Variabel", error_x="err_plus", error_x_minus="err_minus",
                        labels={"Koef": "Estimasi koefisien (beta)", "Variabel": "Variabel prediktor"},
                        title="Forest plot koefisien regresi (interval kepercayaan 95%)")
    fig_fp.add_vline(x=0, line_dash="dash", line_color="#D98B6A", line_width=1.5)
    st.plotly_chart(fig_fp)
    nsig = int(((ci[0] > 0) | (ci[1] < 0)).drop("Intercept").sum())
    interp(f"Titik adalah estimasi koefisien dan garis horizontal adalah interval kepercayaan 95%. "
           f"{nsig} dari {len(m.params) - 1} koefisien tidak memotong garis nol, artinya signifikan pada α = 5%; "
           "yang memotong nol tidak berbeda nyata dari nol.",
           tujuan="besar, arah, dan ketelitian tiap koefisien. Titik = estimasi, garis = selang kepercayaan 95%, garis putus-putus = nol (tidak berpengaruh)",
           arti="garis yang tidak memotong nol berarti variabel itu berpengaruh nyata. Titik di kanan nol menaikkan ln(harga), di kiri nol menurunkan, "
                "dan garis yang pendek berarti estimasinya presisi. Garis yang memotong nol berarti pengaruhnya tidak bisa dibedakan dari nol.",
           out="forest plot koefisien regresi")

    st.subheader("Keterbatasan model")
    st.markdown("- Harga adalah harga **penawaran** iklan, bukan transaksi.\n"
                "- Variabel penting tidak ada di model (kondisi rumah, jumlah kamar, akses, kecamatan), sehingga ada bias variabel yang terlewat.\n"
                "- Luas tanah dan luas bangunan berkorelasi, sehingga koefisien masing-masing kurang stabil.\n"
                "- Kota terlalu agregat (misalnya Bogor mencakup area mahal dan murah).\n"
                "- Model hanya menunjukkan **asosiasi**, bukan sebab-akibat.\n"
                "- Pencilan dan data tidak valid dibuang, sehingga model tidak berlaku untuk rumah super mewah atau sangat murah.\n"
                + ("" if normal_ok and homo_ok else "- Asumsi normalitas dan/atau homoskedastisitas tidak terpenuhi (lihat uji asumsi), sehingga p-value dan selang kepercayaan perlu dibaca hati-hati.\n"))

# TAB 6. KESIMPULAN DAN REKOMENDASI
with tabs[5]:
    st.header("Kesimpulan dan Rekomendasi")
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
