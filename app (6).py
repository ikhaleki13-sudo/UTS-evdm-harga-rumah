# DASHBOARD UTS EKSPLORASI & VISUALISASI DATA MULTIVARIAT
# Topik : Harga Rumah Jabodetabek (data iklan Rumah123)
# Alur  : data, univariat, bivariat, multivariat, regresi, kesimpulan

# 0. IMPORT LIBRARY
import base64, gzip, io, os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.stats.stattools import jarque_bera

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="Harga Rumah Jabodetabek", page_icon="🏠", layout="wide")

# 2. KAMUS LABEL
LBL = {
    "price_miliar": "Harga (Rp miliar)",
    "land_size_m2": "Luas tanah (m²)",
    "building_size_m2": "Luas bangunan (m²)",
    "city": "Kota",
    "certificate": "Sertifikat"
}
NUM = ["price_miliar", "land_size_m2", "building_size_m2"]

# 2b. DATA TERTANAM
DATA_B64 = """
H4sIALdv+GoC/+19S6+kyXHdvn8F4ZUNlIF8P7aCV14ZsLwmmuSYbA81IobDhfzrnflV1a26GZFxMrIe0xpLYFNS942bX74i43Hi
xN9+/vbHH37/7aff//y30x+//fJvp79+/elPv//7t//7w+//xZ3+8I9vf/3Tt5/+/PEXf/zh51++/e9vf/z6yw9fXK3m/D+nf/rh
x69///Zyvp5cdqe//+Vffvdff/f388/++PWX3/3l64+/+5dvf/324xfr8iAU4ynV01/+/Icm0n/uz//46evv/vD1p/6/f/piaxx+
3tp6st5OJvw3o0QxJ5uD9FkhjkLelZOtSRL6mMptJJ9Orv2/kpAnEwrh+DObEP02Z9sw0UwlXCUS7f+00T1/NpGdTRHXrUZHjkCb
zX/669dvP/30b19/95//9rf/84fTn7/9/O3H09c/ff3l9Ke//vW//KcvbRuZo1DczmrPD5wjq90l+jJwK1T9sUqzXxZ9IhLu5PN8
6yZrM/1ax16PKG/allBgV3Jl367D3UYzxx9JTdTCfGItkkxO4bNILqfixTkZsqGpzyk+eRjPrnfAS7d1vUKhGtaD1R7XodqTdfj7
iKpN5uTFj4tE/x83wuT5EafXvh06pJeyJcosHvf+C7nf7cEKbq59faVTnH8t+dg2O1lFlcjpaiefQTJKEwnyqSAPqQ/lFMDBIBtc
2vEzUX7hyX61k1Tl7UrknLdldvLzu/Vxb5IJo85r80+ihBtvbXskgrhotdZxnXO7TaJMzONCN2V28rJitfRJ7EdUvuiJ3ITchYJ8
bz3RyN3OAUL2OtbdFW0KVrpzmd6fYpauz8cZSKcgb2iOSt3o6Yr1/0gzD4Gxps1zn6+dC60fRb/AW8dS/2L5pn7HyIxRZAnLSUxd
nOq1n1QdefQEu5Ucq5TgnFUzCMbSAWQlRLzN44mw8hNhybvi0vFHmgo57z6nQ3t9IW9o++ySlYe0jV7mFr0lxnl7bWJasLKpmeSQ
mTSf6tS+iVRzygrHkpfN+nqoEElBj9ogFqBxLPVPvD05p3za++7IA6WqXYHKrBmYv/WcDFho6s20KwUGciU+41o19S5FH3wYFFBT
J8nLZzNTCXmdMzW68pMlrCf+S7MErXm+kLO8w52/DyHOKmhXZx7GstSXtaFdUFnr01CFKWK8kLEFXZUVBxfwyeLk82h6tx+3srWe
8yASA7gAmRiQ6AJsbST5sPZVUTYIiRNhc+1vlDQK0WdtjaNy65syB+YdNewDutCzBxo+uIz5GaFf6GmcOaJXmr5SFdnfOyI0zuL9
ykJcI5bripr1qBdsnFQi2V0QSLxayp82ST4RG6M4y+YchMAtWbS2YrK7o5dgVHRzi5N/toyzmVM7zj7fXnPMsQbKamsc+iacnx3p
2Iz7UxP03ll1bbXX9FhsECExnJWXlfLGYQ6g8X4j5PAc/b2shaAqn/U17vp5+GipJc3pwiyITL5IzEZWJZMtfRd/UBFMxcQxlnuPl
koSlcSzZuSfte/nVwJ+nz2yP84PIwceZ46GqVw4dMl8y5R+G58bKNCJuj74cQhyCHvtu4co4wMsG1UFRuYUJu4TtWSj8P6hFDTzIS
H8IhsykVJlVUhT2MJP3cHmWU7rCFGmeyO7QhQqJ5FQWuC2MzVirtHEyLZQzQUM00p3Ps192mcE00tqWgtEqaxahhyVc8D3qRKMrdi0
KYie/Wa+a2cU7njJo4QYaVpQNqyNZfxmOflIn7gEFF4tRam9QtLaNwTIACUSZ6sUZcKxmRAemBEbKtIUxhEFHrlRPypbw9j3DPOmT
3uPiLNveCS35qJ/iseQaakgmL9lRm98WR5d6oIUwBa+w5LMorFd7smbST8tAxHiFsP5RwZCAIJEJHRV3AlheAsFOVkZSeQzB2L1
IPBpMrUSi6jPQ06D343ecXL8M8zOlRKISFIa1TWjdEEwgQQVPQqAWz4SKW8oAwhtDq6L2viL7Q64Vxnxtd0BHPitjh7rIi84CcB0
SCPYJAoxaPY8CCl4kt3N3agDa0cXPAboXTP4oHzyADZhSTAmAqiTLRtLF8nS2bZ0ck7IkpDpghD1zcMp5OdGY4jebZOXo8wkIdjP
tfm1JWqsxKwXoCc5ByX+rBLQFswy0Viqbe+N1dnBTZ1VXWK/J9tB2sfRQaJueZuEsLyF+6aUVAcRflMxA2oiOBR35rwseaWYjHFQ
R6qllQqVHMQKXmSq5cDSjke33/GidWBBNjoymyG+xIk56cB+C5E+3cCyoA9dN8Qc2vUxRl/BvfUMWk7GIuglXK4MYNihUqd3Cenf
nsBZ8E6HFTuOgPdPF9LPJjL5JoCTMerXujilOnKeq4/rd2ArqseAOpoPBUxAWtWXrDvigLn7sclnalpKWGWOjLq5pQ0bpKl91cpa
fsLd2aX6/h6nI3K7v/X5z2mI+wL4u92a8o4eZ0b8yGq8Gv1Qd2rF4Mup7d/s39O5G8263M5eN+j+0e632K2uQJb+Z31lZqL1c29v84FlDpv1M7R45bX68P9w+WeeqN/0qzzsNyN6nLlw2r91PWMlD4L+psSxB1rZ7fxsTWRBQFPVmCerFqWdeB5NWY/rd8fSw9ZNs+cw
Bv27Q2zVl+t0dqtkAh5sq6vuw8etbNVr6t+lV4voLo/3+NhfZqK0S/ehkLUX61CrTx5BZPPn/cIVeTm4enNBQUZxsgdmnXwxOz3
USXg4VaMwZsIOjmMGEvIiUUyRh08ZSgiW5s7nN93n4K/eFf/h80b8K8Deu8LD3+m56mF979sUrm8ZtDJGHVJADLD2/lVv7CFDz
puF6ftWpOH4vJQ7/scK/lkTUv9PJUViEl23CqMMa0m71PQ0tg4LU4X/y7iRlzPP5gV7PdqBY7QbczRER+Exc/nz5G8v7Knoq
H2fXFk2cM4iR0yWqR8kX4r96p08t4pLeBwIeYJ0v6n4Z6C2h7L167p59gJc59j7uB7Xm80tJ/eQ+w96+37vT8kQ0Y1365w5eN2t0v78l33d/k4m6fL7tJ0yB5X11s2Q93y+K49+7+rL3z2V1279ZfL7L98b9s14N9/P7Z3z+9vn1H+7Pj2/f/u3fv/3973/7
+ff/9//929//+vffv//89/8P3e7HqgQAAA==
"""

def baca_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    teks = gzip.decompress(base64.b64decode(DATA_B64)).decode("utf-8")
    return pd.read_csv(io.StringIO(teks))

# 3. MEMUAT DAN MEMBERSIHKAN DATA
@st.cache_data
def load(path="jabodetabek_house_price.csv"):
    d = baca_data(path)
    n_var = d.shape[1]
    log = [("Data awal", len(d), "Hasil scraping Rumah123, 5 variabel terpilih")]
    d["city"] = d["city"].str.strip()
    c = d.certificate.fillna("").str.lower()
    d["certificate"] = np.select([c.str.startswith("shm"), c.str.startswith("hgb"), c == ""],
                                 ["SHM", "HGB", "Tidak diketahui"], "Lainnya")

    def step(name, mask, why):
        nonlocal d
        d = d[mask(d)]
        log.append((name, len(d), why))

    step("Hapus baris luas kosong", lambda x: x.land_size_m2.notna() & x.building_size_m2.notna(),
         "2 baris tanpa luas tanah/bangunan")
    step("Hapus luas tidak masuk akal",
         lambda x: x.land_size_m2.between(10, 2000) & x.building_size_m2.between(10, 2000),
         "Luas < 10 m² atau > 2.000 m² (kemungkinan salah input)")
    q1, q3 = np.log10(d.price_in_rp).quantile([.25, .75])
    i = q3 - q1
    step("Hapus pencilan harga (IQR pada log10 harga)",
         lambda x: np.log10(x.price_in_rp).between(q1 - 1.5 * i, q3 + 1.5 * i),
         "Harga ekstrem (mis. Rp580 miliar) merusak model")
    
    out = d.reset_index(drop=True).copy()
    out["price_miliar"] = out.price_in_rp / 1e9
    out["ln_price"] = np.log(out.price_in_rp)
    out["ln_land"] = np.log(out.land_size_m2)
    out["ln_bld"] = np.log(out.building_size_m2)
    return out[["city", "certificate", "land_size_m2", "building_size_m2", "price_miliar",
                "ln_price", "ln_land", "ln_bld"]], log, n_var

# 4. FUNGSI BANTU
def pf(p):
    return "< 0.001" if p < .001 else f"= {p:.3f}"

def interp(t):
    st.info("**Interpretasi:** " + t)

def desc(s):
    return pd.Series({"n": s.count(), "Rata-rata": s.mean(), "Median": s.median(), "Std": s.std(),
                      "Min": s.min(), "Q1": s.quantile(.25), "Q3": s.quantile(.75),
                      "Maks": s.max(), "Skewness": s.skew()})

df, log, n_var = load()

# 6. SIDEBAR: FILTER DATA
st.sidebar.header("Filter data")
cities = sorted(df.city.unique())
certs = sorted(df.certificate.unique())
sc = st.sidebar.multiselect("Kota", cities, cities)
ss = st.sidebar.multiselect("Sertifikat", certs, certs)
pmin, pmax = float(df.price_miliar.min()), float(df.price_miliar.max())
lo, hi = st.sidebar.slider("Harga (Rp miliar)", pmin, pmax, (pmin, pmax))
f = df[df.city.isin(sc) & df.certificate.isin(ss) & df.price_miliar.between(lo, hi)]
st.sidebar.metric("Observasi terpilih", f"{len(f):,}")

# 7. JUDUL DAN TAB
st.title("🏠 Eksplorasi & Regresi Harga Rumah Jabodetabek")
st.caption("UTS Eksplorasi dan Visualisasi Data Multivariat · Program Studi Statistika, Universitas Akprind Indonesia")

if len(f) < 30:
    st.warning("Data terlalu sedikit (< 30). Longgarkan filter di sidebar.")
    st.stop()

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

# TAB 2. EKSPLORASI UNIVARIAT (Soal no. 2)
with tabs[1]:
    st.header("Eksplorasi Univariat")
    st.subheader("Statistik deskriptif")
    st.dataframe(pd.DataFrame({LBL[c]: desc(f[c]) for c in NUM}).T.round(2))
    
    a, b = st.columns(2)
    a.dataframe(f.city.value_counts().rename("Jumlah").to_frame())
    b.dataframe(f.certificate.value_counts().rename("Jumlah").to_frame())

    # Visualisasi 1: Histogram + Boxplot numerik
    st.subheader("Visualisasi 1: Histogram + boxplot (variabel numerik)")
    v = st.selectbox("Variabel numerik", NUM, format_func=LBL.get)
    logx = st.checkbox("Skala log pada sumbu x", key="uni_logx")
    fig = px.histogram(f, x=v, nbins=50, marginal="boxplot", labels=LBL, title=f"Distribusi {LBL[v]}")
    fig.update_layout(yaxis_title="Frekuensi")
    if logx:
        fig.update_xaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)
    
    s = f[v]
    q1, q3 = s.quantile([.25, .75])
    nout = int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())
    shape = "miring ke kanan (rata-rata > median)" if s.skew() > .5 else \
            "miring ke kiri" if s.skew() < -.5 else "relatif simetris"
    interp(f"{LBL[v]} berdistribusi {shape}: rata-rata {s.mean():,.2f}, median {s.median():,.2f}, skewness {s.skew():.2f}. "
           f"Terdapat {nout} pencilan menurut aturan IQR ({nout / len(s):.1%} data), yaitu rumah-rumah dengan nilai jauh di atas kebanyakan.")

    # Visualisasi 2: Diagram batang kota
    st.subheader("Visualisasi 2: Diagram batang jumlah rumah per kota")
    cc = f.city.value_counts().reset_index()
    cc.columns = ["city", "n"]
    st.plotly_chart(px.bar(cc, x="city", y="n", labels={"city": "Kota", "n": "Jumlah rumah"},
                           title="Jumlah rumah dijual per kota", text="n"), use_container_width=True)
    interp(f"Kota terbanyak adalah {cc.city.iloc[0]} ({cc.n.iloc[0]:,} rumah, {cc.n.iloc[0] / len(f):.1%}); paling sedikit {cc.city.iloc[-1]} "
           f"({cc.n.iloc[-1]:,} rumah). Sebaran antar kota tidak seimbang sehingga perbandingan antar kota perlu hati-hati.")

    # Visualisasi 3: Diagram batang sertifikat
    st.subheader("Visualisasi 3: Diagram batang jenis sertifikat")
    ct = f.certificate.value_counts().reset_index()
    ct.columns = ["certificate", "n"]
    st.plotly_chart(px.bar(ct, x="certificate", y="n", color="certificate",
                           labels={"certificate": "Sertifikat", "n": "Jumlah rumah"},
                           title="Jumlah rumah per jenis sertifikat", text="n"), use_container_width=True)
    interp(f"Sertifikat dominan: {ct.certificate.iloc[0]} ({ct.n.iloc[0] / len(f):.1%} data). Kategori lain jauh lebih sedikit.")

# TAB 3. EKSPLORASI BIVARIAT (Soal no. 3)
with tabs[2]:
    st.header("Eksplorasi Bivariat")
    
    # Visualisasi 1: Scatter plot numerik vs numerik
    st.subheader("Visualisasi 1: Scatter plot harga vs luas (numerik–numerik)")
    xv = st.selectbox("Variabel x", ["building_size_m2", "land_size_m2"], format_func=LBL.get)
    lg = st.checkbox("Skala log-log", value=True, key="biv_lg")
    st.plotly_chart(px.scatter(f, x=xv, y="price_miliar", opacity=.4, trendline="ols",
                               trendline_options=dict(log_x=lg, log_y=lg), log_x=lg, log_y=lg,
                               labels=LBL, title=f"Harga vs {LBL[xv]}"), use_container_width=True)
    pr = f[[xv, "price_miliar"]].corr().iloc[0, 1]
    sp = f[[xv, "price_miliar"]].corr(method="spearman").iloc[0, 1]
    interp(f"Hubungan positif: korelasi Pearson {pr:.2f}, Spearman {sp:.2f}. Semakin besar {LBL[xv].lower()}, harga cenderung naik, "
           "tetapi sebaran lebar, artinya ukuran saja belum cukup menjelaskan harga.")

    # Visualisasi 2: Boxplot harga per kota
    st.subheader("Visualisasi 2: Boxplot harga per kota (numerik–kategorik)")
    ly = st.checkbox("Skala log pada harga", value=True, key="ly2")
    order = f.groupby("city").price_miliar.median().sort_values().index.tolist()
    st.plotly_chart(px.box(f, x="city", y="price_miliar", color="city", log_y=ly, category_orders={"city": order},
                           labels=LBL, title="Distribusi harga per kota (urut median)"), use_container_width=True)
    md = f.groupby("city").price_miliar.median().sort_values()
    interp(f"Median harga tertinggi di {md.index[-1]} (Rp{md.iloc[-1]:.2f} miliar) dan terendah di {md.index[0]} "
           f"(Rp{md.iloc[0]:.2f} miliar), selisih {md.iloc[-1] / md.iloc[0]:.1f} kali. Lokasi sangat membedakan harga.")

    # Visualisasi 3: Boxplot harga per sertifikat
    st.subheader("Visualisasi 3: Boxplot harga per jenis sertifikat (numerik–kategorik)")
    st.plotly_chart(px.box(f, x="certificate", y="price_miliar", color="certificate", log_y=True,
                           labels=LBL, title="Distribusi harga per jenis sertifikat"), use_container_width=True)
    mc = f.groupby("certificate").price_miliar.median().sort_values()
    interp("Median harga per sertifikat: " + ", ".join(f"{k} Rp{v:.2f} miliar" for k, v in mc.items()) +
           ". Perbedaan ini belum mengontrol ukuran dan kota, jadi diuji ulang di regresi.")

# TAB 4. EKSPLORASI MULTIVARIAT (Soal no. 4)
with tabs[3]:
    st.header("Eksplorasi Multivariat")
    
    # Visualisasi 1: Bubble plot
    st.subheader("Visualisasi 1: Bubble plot luas tanah × luas bangunan × harga × kota")
    lg2 = st.checkbox("Skala log-log", value=True, key="lg2")
    st.plotly_chart(px.scatter(f, x="land_size_m2", y="building_size_m2", color="city", size="price_miliar",
                               size_max=18, opacity=.5, log_x=lg2, log_y=lg2, labels=LBL,
                               title="Luas tanah vs luas bangunan (warna = kota, ukuran = harga)"), use_container_width=True)
    rr = f[["ln_land", "ln_bld"]].corr().iloc[0, 1]
    interp(f"Luas tanah dan luas bangunan berkorelasi kuat (r = {rr:.2f} pada skala log), sehingga titik membentuk pita naik. "
           "Titik besar (harga tinggi) menumpuk di area luas besar, dan warna kota terlihat berkelompok di area tertentu.")

    # Visualisasi 2: Facet scatter plot per kota
    st.subheader("Visualisasi 2: Harga vs luas bangunan per kota dan sertifikat")
    st.plotly_chart(px.scatter(f, x="building_size_m2", y="price_miliar", color="certificate", facet_col="city",
                               facet_col_wrap=3, log_x=True, log_y=True, opacity=.5, height=800, labels=LBL,
                               title="Harga vs luas bangunan, dipisah per kota (warna = sertifikat)"), use_container_width=True)
    ppm = (f.price_miliar * 1000 / f.building_size_m2).groupby(f.city).median().sort_values()
    interp(f"Di setiap kota hubungan harga–luas bangunan tetap positif, tetapi posisi vertikal kelompok berbeda: median harga per m² "
           f"bangunan tertinggi di {ppm.index[-1]} (Rp{ppm.iloc[-1]:.1f} juta) dan terendah di {ppm.index[0]} (Rp{ppm.iloc[0]:.1f} juta). "
           "Artinya pada luas yang sama, lokasi menentukan harga.")

    # Visualisasi 3: Heatmap korelasi Pearson
    st.subheader("Visualisasi 3: Heatmap korelasi variabel numerik (skala log)")
    cm = f[["ln_price", "ln_land", "ln_bld"]].corr()
    cm.index = cm.columns = ["ln(Harga)", "ln(Luas tanah)", "ln(Luas bangunan)"]
    st.plotly_chart(px.imshow(cm, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                              labels=dict(color="Korelasi"), title="Matriks korelasi Pearson"), use_container_width=True)
    interp(f"ln(Harga) berkorelasi kuat dengan ln(Luas bangunan) (r = {cm.iloc[0, 2]:.2f}) dan ln(Luas tanah) (r = {cm.iloc[0, 1]:.2f}); "
           f"kedua luas saling berkorelasi (r = {cm.iloc[1, 2]:.2f}), sehingga multikolinearitas perlu dicek lewat VIF.")

# TAB 5. REGRESI LINEAR BERGANDA (Soal no. 5)
with tabs[4]:
    st.header("Regresi Linear Berganda")

    ref = f.city.value_counts().idxmax()
    refc = "SHM" if "SHM" in set(f.certificate) else f.certificate.value_counts().idxmax()
    terms = ["ln_land", "ln_bld"]
    if f.city.nunique() > 1:
        terms.append(f"C(city, Treatment(reference='{ref}'))")
    if f.certificate.nunique() > 1:
        terms.append(f"C(certificate, Treatment(reference='{refc}'))")
    
    m = smf.ols("ln_price ~ " + " + ".join(terms), f).fit()

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

    names = [nice(k) for k in m.params.index]
    tb = pd.DataFrame({"Koefisien": m.params.values, "Std. error": m.bse.values, "t": m.tvalues.values,
                       "p-value": m.pvalues.values}, index=names)
    tb["Efek harga (%)"] = [np.nan if n in ("Intercept", "ln(Luas tanah)", "ln(Luas bangunan)")
                            else (np.exp(b) - 1) * 100 for n, b in zip(names, m.params.values)]
    st.subheader("Hasil estimasi")
    st.dataframe(tb.round(4))
    
    rmse = float(np.sqrt(np.mean(m.resid ** 2)))
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

    bl, bb = m.params["ln_land"], m.params["ln_bld"]
    sig = int((m.pvalues.drop("Intercept") < .05).sum())
    st.subheader("Interpretasi koefisien dan pengujian")
    st.markdown(f"- **Luas tanah**: naik 1% → harga naik sekitar {bl:.2f}% (kota, sertifikat, luas bangunan tetap), "
                f"{'signifikan' if m.pvalues['ln_land'] < .05 else 'tidak signifikan'} (p {pf(m.pvalues['ln_land'])}).\n"
                f"- **Luas bangunan**: naik 1% → harga naik sekitar {bb:.2f}%, "
                f"{'signifikan' if m.pvalues['ln_bld'] < .05 else 'tidak signifikan'} (p {pf(m.pvalues['ln_bld'])}).\n"
                "- **Dummy kategorik**: kolom *Efek harga (%)* = selisih harga rata-rata (geometrik) dibanding kategori acuan, prediktor lain tetap.\n"
                f"- **Uji simultan F**: p {pf(m.f_pvalue)} → {'minimal satu' if m.f_pvalue < .05 else 'tidak ada'} prediktor berpengaruh pada α = 5%.\n"
                f"- **Uji parsial t**: {sig} dari {len(m.params) - 1} koefisien signifikan pada α = 5%.")

    X = m.model.exog
    vf = {nm: vif(X, i) for i, nm in enumerate(m.model.exog_names) if nm in ("ln_land", "ln_bld")}
    res = np.asarray(m.resid)
    bp = het_breuschpagan(res, X)[1]
    jb = jarque_bera(res)[1]
    sw = stats.shapiro(res)[1]
    ks = stats.kstest(res, stats.norm(res.mean(), res.std()).cdf)[1]
    gl = smf.ols("ar ~ ln_land + ln_bld", f.assign(ar=np.abs(res))).fit().f_pvalue
    cd = m.get_influence().cooks_distance[0]

    st.subheader("Evaluasi model")
    st.markdown(f"- R² = {m.rsquared:.3f}: model menjelaskan {m.rsquared * 100:.1f}% variasi ln(harga); R² adj. = {m.rsquared_adj:.3f}.\n"
                f"- RMSE = {rmse:.3f} (skala ln) ≈ kesalahan prediksi harga sekitar {(np.exp(rmse) - 1) * 100:.0f}%.\n"
                f"- VIF ln(Luas tanah) = {vf['ln_land']:.2f}, ln(Luas bangunan) = {vf['ln_bld']:.2f} "
                f"({'multikolinearitas rendah-sedang' if max(vf.values()) < 5 else 'multikolinearitas tinggi (≥ 5)'}).")

    st.subheader("Uji asumsi residual")
    ok = lambda p: "Terpenuhi" if p >= .05 else "Tidak terpenuhi"
    st.dataframe(pd.DataFrame([
        ("Normalitas", "Shapiro–Wilk", pf(sw).replace("= ", ""), ok(sw)),
        ("Normalitas", "Kolmogorov–Smirnov", pf(ks).replace("= ", ""), ok(ks)),
        ("Normalitas", "Jarque–Bera", pf(jb).replace("= ", ""), ok(jb)),
        ("Homoskedastisitas", "Breusch–Pagan", pf(bp).replace("= ", ""), ok(bp)),
        ("Homoskedastisitas", "Glejser (|residual| ~ prediktor)", pf(gl).replace("= ", ""), ok(gl))],
        columns=["Asumsi", "Uji", "p-value", "Kesimpulan (α = 5%)"]), hide_index=True)
    st.caption("Autokorelasi (Durbin–Watson, ACF) tidak diuji karena data cross-section: baris iklan tidak berurutan waktu.")
    
    normal_ok, homo_ok = min(sw, ks, jb) >= .05, min(bp, gl) >= .05
    interp(f"Normalitas residual {'terpenuhi' if normal_ok else 'tidak terpenuhi'} dan homoskedastisitas "
           f"{'terpenuhi' if homo_ok else 'tidak terpenuhi'} pada α = 5%. "
           + ("" if normal_ok and homo_ok else "Dengan n besar koefisien tetap konsisten, tetapi uji t dan p-value perlu dibaca hati-hati."))

    # VISUALISASI REGRESI (6 Visualisasi Lengkap)
    st.subheader("Visualisasi regresi 1: Nilai aktual vs prediksi")
    fv = np.asarray(m.fittedvalues)
    ac = f.ln_price.values
    fig = px.scatter(x=fv, y=ac, opacity=.4, labels={"x": "ln(Harga) prediksi", "y": "ln(Harga) aktual"},
                     title="Aktual vs prediksi")
    fig.add_trace(go.Scatter(x=[ac.min(), ac.max()], y=[ac.min(), ac.max()], mode="lines",
                             name="Garis y = x", line=dict(dash="dash", color="red")))
    st.plotly_chart(fig, use_container_width=True)
    interp(f"Titik mengikuti garis y = x dengan R² = {m.rsquared:.2f}; makin dekat garis, makin akurat prediksi.")

    st.subheader("Visualisasi regresi 2: Residual vs nilai prediksi")
    fig = px.scatter(x=fv, y=res, opacity=.4,
                     labels={"x": "ln(Harga) prediksi", "y": "Residual"}, title="Residual vs prediksi")
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)
    interp(f"Residual seharusnya acak di sekitar nol tanpa pola. Uji Breusch–Pagan p {pf(bp)}.")

    st.subheader("Visualisasi regresi 3: Histogram residual")
    st.plotly_chart(px.histogram(x=res, nbins=40, labels={"x": "Residual"}, title="Distribusi residual")
                    .update_layout(yaxis_title="Frekuensi"), use_container_width=True)
    interp(f"Residual berpusat di sekitar nol (rata-rata {res.mean():.3f}), skewness {stats.skew(res):.2f}.")

    st.subheader("Visualisasi regresi 4: Q-Q plot residual")
    (osm, osr), (qs, qi, _) = stats.probplot(res, dist="norm")
    fig = px.scatter(x=osm, y=osr, opacity=.5, labels={"x": "Kuantil teoretis (normal)", "y": "Kuantil residual"},
                     title="Q-Q plot residual")
    fig.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[qi + qs * osm.min(), qi + qs * osm.max()], mode="lines",
                             name="Garis normal", line=dict(color="red")))
    st.plotly_chart(fig, use_container_width=True)
    interp("Q-Q plot mengevaluasi apakah residual berdistribusi normal di sepanjang garis referensi.")

    st.subheader("Visualisasi regresi 5: Cook's distance (pengamatan berpengaruh)")
    fig = px.scatter(x=np.arange(len(cd)), y=cd, opacity=.5,
                     labels={"x": "Indeks pengamatan", "y": "Cook's distance"}, title="Cook's distance")
    fig.add_hline(y=4 / len(cd), line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)
    interp(f"Nilai maksimum Cook's distance {cd.max():.3f}; {int((cd > 4 / len(cd)).sum())} pengamatan melewati batas 4/n.")

    st.subheader("Visualisasi regresi 6: Forest plot koefisien regresi (CI 95%)")
    ci = m.conf_int()
    params_idx = ci.index
    fp = pd.DataFrame({
        "Variabel": [nice(k) for k in params_idx],
        "Koef": m.params.values,
        "err_plus": ci[1].values - m.params.values,
        "err_minus": m.params.values - ci[0].values
    }).sort_values("Koef")

    fig_fp = px.scatter(
        fp, x="Koef", y="Variabel",
        error_x="err_plus", error_x_minus="err_minus",
        labels={"Koef": "Estimasi Nilai Koefisien (Beta)", "Variabel": "Variabel Prediktor"},
        title="Forest Plot Koefisien Regresi (Interval Kepercayaan 95%)"
    )
    fig_fp.update_traces(marker=dict(size=10, color="#1B365D"), error_x=dict(color="#1B365D", thickness=2))
    fig_fp.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5)
    st.plotly_chart(fig_fp, use_container_width=True)
    interp("Forest plot menampilkan koefisien beserta interval kepercayaan 95%. Variabel yang intervalnya tidak melewati garis merah berpengaruh signifikan.")

    st.subheader("Keterbatasan model")
    st.markdown("- Harga adalah harga **penawaran** iklan, bukan transaksi.\n"
                "- Variabel penting tidak ada di model (kondisi rumah, jumlah kamar, akses, kecamatan).\n"
                "- Luas tanah dan luas bangunan berkorelasi, sehingga koefisien masing-masing kurang stabil.\n"
                "- Model hanya menunjukkan **asosiasi**, bukan sebab-akibat.")

# TAB 6. KESIMPULAN DAN REKOMENDASI
with tabs[5]:
    st.header("Kesimpulan dan Rekomendasi")
    cs = tb.loc[[n for n in tb.index if n.startswith("Kota")], "Efek harga (%)"] if any(n.startswith("Kota") for n in tb.index) else pd.Series(dtype=float)
    st.subheader("Kesimpulan")
    st.markdown(f"1. Harga rumah sangat miring ke kanan (median Rp{f.price_miliar.median():.2f} miliar vs rata-rata Rp{f.price_miliar.mean():.2f} miliar).\n"
                f"2. Harga berhubungan positif dengan luas bangunan (r = {f[['building_size_m2', 'price_miliar']].corr().iloc[0, 1]:.2f}) dan luas tanah; lokasi (kota) membedakan harga secara jelas.\n"
                f"3. Model regresi menjelaskan {m.rsquared * 100:.1f}% variasi ln(harga). Elastisitas luas bangunan {bb:.2f} dan luas tanah {bl:.2f}.\n"
                + (f"4. Dibanding {ref}, efek kota berkisar dari {cs.min():.0f}% sampai {cs.max():.0f}%.\n" if len(cs) else ""))
    st.subheader("Rekomendasi")
    st.markdown("- **Pembeli**: bandingkan harga per m² dalam kota yang sama, bukan antar kota.\n"
                "- **Penjual/agen**: tetapkan harga dengan acuan luas bangunan dan lokasi, lalu sesuaikan sertifikat.\n"
                "- **Penelitian lanjutan**: tambahkan jumlah kamar, kondisi, dan kecamatan.")
