"""Aplikasi web segmentasi pelanggan mall (Streamlit) - 4 FITUR.

Jalankan lokal:  streamlit run streamlit_app.py
Model dibuat oleh train_model.py (model/mall_model.joblib).

Pembagian fitur (1 fitur = 1 anggota):
  FITUR 1 - Penentuan Segmen Pelanggan   (fungsi fitur_1)
  FITUR 2 - Saran Strategi Pemasaran     (fungsi fitur_2)
  FITUR 3 - Dashboard Visualisasi Segmen (fungsi fitur_3)
  FITUR 4 - Segmentasi Massal via CSV    (fungsi fitur_4)
"""
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Segmentasi Pelanggan Mall", page_icon="🛍️", layout="centered")


@st.cache_resource
def load_artifacts():
    return joblib.load("model/mall_model.joblib")


try:
    ART = load_artifacts()
except Exception as e:  # biasanya karena versi scikit-learn berbeda
    st.error("Model gagal dimuat. Biasanya penyebabnya beda versi scikit-learn. "
             "Jalankan ulang `python train_model.py Mall_Customers.csv` di lingkungan yang sama.")
    st.exception(e)
    st.stop()

CLF, KM, SCALER = ART["classifier"], ART["kmeans"], ART["cl_scaler"]
NAMES, PROFILE, ADVICE = ART["segment_names"], ART["segment_profile"], ART["segment_advice"]
OPT, DEF, RNG, MET = ART["options"], ART["defaults"], ART["ranges"], ART["metrics"]
DATA = ART["data"]

# langkah konkret tambahan untuk tiap segmen (dipakai di Fitur 2)
LANGKAH = {
    "Prioritas": ["Program member VIP dengan poin berlipat", "Akses awal ke koleksi dan event baru", "Layanan personal shopper"],
    "Reguler": ["Promo musiman (gajian, lebaran, akhir tahun)", "Program poin per transaksi", "Voucher untuk kunjungan berikutnya"],
    "Antusias": ["Diskon dan flash sale", "Cicilan ringan 0%", "Paket bundling hemat"],
    "Potensial": ["Kampanye personal lewat email atau WhatsApp", "Undangan event premium", "Penawaran eksklusif yang membuat penasaran"],
    "Hemat": ["Kupon dan promo harga murah", "Notifikasi cuci gudang", "Biaya pemasaran dijaga rendah"],
}


# ------------------------------------------------------------ fungsi bantu
def segmen_dari_skor(income, score):
    """Skor belanja diketahui: tentukan segmen langsung dengan K-Means."""
    X = pd.DataFrame([[income, score]], columns=ART["cl_cols"])
    return int(KM.predict(SCALER.transform(X))[0])


def segmen_dari_profil(age, income, gender):
    """Skor belanja belum diketahui: perkirakan segmen dengan Random Forest."""
    row = pd.DataFrame([{"age": age, "annual_income_k": income, "gender": gender}])[ART["NUM"] + ART["CAT"]]
    proba = CLF.predict_proba(row)[0]
    return {int(c): float(p) for c, p in zip(CLF.classes_, proba)}


def tampilkan_profil(seg_id):
    p = PROFILE[seg_id]
    st.write(f"**Profil segmen:** {p['jumlah']:.0f} pelanggan, rata-rata usia {p['usia_rata2']:.0f} tahun, "
             f"pendapatan {p['pendapatan_rata2']:.0f} ribu $, spending score {p['spending_rata2']:.0f}.")


# =========================================================== FITUR 1
def fitur_1():
    st.subheader("Penentuan Segmen Pelanggan")
    st.write("Masukkan data satu pelanggan, lalu sistem menentukan segmennya.")

    left, right = st.columns(2)
    with left:
        gender = st.selectbox("Jenis kelamin", OPT["gender"],
                              format_func=lambda g: "Perempuan" if g == "Female" else "Laki-laki")
        age = st.number_input("Usia (tahun)", min_value=1, max_value=100, value=int(DEF["age"]), step=1)
    with right:
        income = st.number_input("Pendapatan tahunan (ribu $)", min_value=1, max_value=500,
                                 value=int(DEF["annual_income_k"]), step=1)
        known = st.checkbox("Spending score sudah diketahui")
        score = st.number_input("Spending score (1-100)", min_value=1, max_value=100,
                                value=int(DEF["spending_score"]), step=1, disabled=not known)

    if st.button("Tentukan segmen", type="primary"):
        st.divider()
        if known:
            seg_id = segmen_dari_skor(income, score)
            st.success(f"Segmen: **{NAMES[seg_id]}**")
            st.caption("Ditentukan langsung oleh K-Means dari pendapatan dan spending score.")
        else:
            proba = segmen_dari_profil(age, income, gender)
            seg_id = max(proba, key=proba.get)
            st.success(f"Perkiraan segmen: **{NAMES[seg_id]}** (keyakinan {proba[seg_id] * 100:.0f}%)")
            st.bar_chart(pd.Series({NAMES[c]: p for c, p in proba.items()}, name="Peluang"))
            st.caption(f"Diperkirakan oleh Random Forest dari usia, gender, dan pendapatan. Akurasi cross-validation "
                       f"{MET['cv_accuracy']:.0%}, jadi hasilnya hanya perkiraan. Isi spending score bila sudah diketahui.")

        tampilkan_profil(seg_id)
        st.write(f"**Saran strategi:** {ADVICE[seg_id]}")

        if not (RNG["age"][0] <= age <= RNG["age"][1]) or \
           not (RNG["annual_income_k"][0] <= income <= RNG["annual_income_k"][1]):
            st.warning("Usia atau pendapatan di luar rentang data latih, hasil kurang dapat diandalkan.")


# =========================================================== FITUR 2
def arah(delta):
    """Ubah selisih terhadap rata-rata menjadi kata yang mudah dibaca."""
    a = abs(delta)
    if a < 5:
        return "mendekati rata-rata"
    return ("jauh di atas" if a >= 20 else "di atas") + " rata-rata" if delta > 0 else \
           ("jauh di bawah" if a >= 20 else "di bawah") + " rata-rata"


def fitur_2():
    st.subheader("Saran Strategi Pemasaran")
    st.write("Pilih segmen untuk melihat profil, strategi, dan langkah konkret pemasarannya.")

    pilihan = st.selectbox("Segmen", [NAMES[c] for c in sorted(NAMES)])
    seg_id = [c for c in NAMES if NAMES[c] == pilihan][0]
    p = PROFILE[seg_id]

    # rata-rata seluruh pelanggan dan selisih segmen terhadapnya (Δ = rata-rata segmen - rata-rata semua)
    semua = {"usia": DATA["age"].mean(), "pendapatan": DATA["annual_income_k"].mean(),
             "skor": DATA["spending_score"].mean()}
    d = {"usia": p["usia_rata2"] - semua["usia"],
         "pendapatan": p["pendapatan_rata2"] - semua["pendapatan"],
         "skor": p["spending_rata2"] - semua["skor"]}

    # ringkasan angka profil
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pelanggan", f"{p['jumlah']:.0f}", f"{p['jumlah'] / len(DATA):.1%} dari total", delta_color="off")
    c2.metric("Usia rata-rata", f"{p['usia_rata2']:.0f} th", f"{d['usia']:+.1f}", delta_color="off")
    c3.metric("Pendapatan (ribu $)", f"{p['pendapatan_rata2']:.0f}", f"{d['pendapatan']:+.1f}", delta_color="off")
    c4.metric("Spending score", f"{p['spending_rata2']:.0f}", f"{d['skor']:+.1f}", delta_color="off")
    st.caption("Angka kecil di bawah tiap nilai adalah selisih terhadap rata-rata semua pelanggan.")

    # strategi + dasar profil agar saran dapat ditelusuri
    st.info(ADVICE[seg_id])
    st.write(f"**Dasar profil:** pendapatan {arah(d['pendapatan'])}, spending score {arah(d['skor'])}, "
             f"usia {arah(d['usia'])}. Komposisi laki-laki {p.get('persen_laki', float('nan')):.0f}%.")

    st.write("**Langkah konkret:**")
    for langkah in LANGKAH.get(pilihan, ["Pantau perilaku belanja segmen ini."]):
        st.write(f"- {langkah}")

    # posisi segmen pada peta pelanggan
    peta = DATA[["annual_income_k", "spending_score"]].copy()
    peta["Kelompok"] = ["Segmen terpilih" if s == pilihan else "Segmen lain" for s in DATA["segmen"]]
    st.write("**Posisi segmen pada peta pelanggan**")
    st.scatter_chart(peta, x="annual_income_k", y="spending_score", color="Kelompok")

    # perbandingan dengan rata-rata seluruh pelanggan
    banding = pd.DataFrame({
        "Segmen ini": [p["usia_rata2"], p["pendapatan_rata2"], p["spending_rata2"]],
        "Semua pelanggan": [semua["usia"], semua["pendapatan"], semua["skor"]],
        "Selisih": [d["usia"], d["pendapatan"], d["skor"]],
    }, index=["Usia", "Pendapatan (ribu $)", "Spending score"]).round(1)
    st.write("**Perbandingan dengan rata-rata semua pelanggan:**")
    st.dataframe(banding)


# =========================================================== FITUR 3
def fitur_3():
    st.subheader("Dashboard Visualisasi Segmen")
    st.write(f"Gambaran {len(DATA)} pelanggan dalam {len(NAMES)} segmen hasil K-Means.")

    st.write("**Sebaran pelanggan** (pendapatan vs spending score)")
    st.scatter_chart(DATA, x="annual_income_k", y="spending_score", color="segmen")

    st.write("**Jumlah pelanggan per segmen**")
    st.bar_chart(DATA["segmen"].value_counts())

    st.write("**Ringkasan semua segmen**")
    st.dataframe(pd.DataFrame([{"Segmen": NAMES[c], "Jumlah": int(PROFILE[c]["jumlah"]),
                                "Usia rata-rata": round(PROFILE[c]["usia_rata2"], 1),
                                "Pendapatan (ribu $)": round(PROFILE[c]["pendapatan_rata2"], 1),
                                "Spending score": round(PROFILE[c]["spending_rata2"], 1)} for c in sorted(NAMES)]),
                 hide_index=True)

    with st.expander("Evaluasi model"):
        st.write(f"- Silhouette K-Means (K=5): **{MET['silhouette']:.3f}**")
        st.write(f"- Random Forest, akurasi cross-validation: **{MET['cv_accuracy']:.1%}**, "
                 f"akurasi data uji: **{MET['test_accuracy']:.1%}**, F1 macro uji: **{MET['test_f1_macro']:.3f}**")
        st.write("Silhouette untuk beberapa nilai K:")
        st.line_chart(pd.Series(ART["k_scores"], name="Silhouette"))


# =========================================================== FITUR 4
def fitur_4():
    st.subheader("Segmentasi Massal via CSV")
    st.write("Unggah file CSV berisi banyak pelanggan, lalu unduh hasil segmentasinya.")
    st.caption("Kolom wajib: `gender` (Male/Female), `age`, `annual_income_k`. "
               "Kolom `spending_score` boleh kosong atau tidak ada (jika ada, segmen ditentukan K-Means).")

    contoh = pd.DataFrame({"gender": ["Female", "Male", "Male"], "age": [25, 40, 33],
                           "annual_income_k": [30, 85, 60], "spending_score": [80, 20, None]})
    st.download_button("Unduh contoh CSV", contoh.to_csv(index=False), "contoh_pelanggan.csv", "text/csv")

    file = st.file_uploader("Unggah CSV", type="csv")
    if file is None:
        return

    df = pd.read_csv(file)
    wajib = ["gender", "age", "annual_income_k"]
    kurang = [c for c in wajib if c not in df.columns]
    if kurang:
        st.error(f"Kolom berikut tidak ditemukan: {', '.join(kurang)}")
        return
    if "spending_score" not in df.columns:
        df["spending_score"] = None

    hasil, metode = [], []
    for _, r in df.iterrows():
        if pd.notna(r["spending_score"]):
            hasil.append(NAMES[segmen_dari_skor(r["annual_income_k"], r["spending_score"])])
            metode.append("K-Means")
        else:
            proba = segmen_dari_profil(r["age"], r["annual_income_k"], r["gender"])
            hasil.append(NAMES[max(proba, key=proba.get)])
            metode.append("Random Forest (perkiraan)")
    df["segmen"] = hasil
    df["metode"] = metode

    st.success(f"{len(df)} pelanggan berhasil diproses.")
    st.dataframe(df, hide_index=True)
    st.bar_chart(df["segmen"].value_counts())
    st.download_button("Unduh hasil (CSV)", df.to_csv(index=False), "hasil_segmentasi.csv", "text/csv")


# ------------------------------------------------------------ halaman utama
st.title("🛍️ Segmentasi Pelanggan Mall")

tab1, tab2, tab3, tab4 = st.tabs(["1. Tentukan Segmen", "2. Strategi", "3. Dashboard", "4. Unggah CSV"])
with tab1:
    fitur_1()
with tab2:
    fitur_2()
with tab3:
    fitur_3()
with tab4:
    fitur_4()

st.caption("Data: Mall Customer Segmentation Data (200 pelanggan). Segmen dibuat dengan K-Means (K=5), "
           "bukan label dari lapangan. Hasil adalah alat bantu analisis, bukan keputusan akhir.")
