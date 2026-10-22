"""Aplikasi web segmentasi pelanggan mall (Streamlit).

Jalankan lokal:  streamlit run streamlit_app.py
Model dibuat oleh train_model.py (model/mall_model.joblib).
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


def segmen_dari_skor(income: float, score: float) -> int:
    """Skor belanja diketahui: tentukan segmen langsung dengan K-Means."""
    X = pd.DataFrame([[income, score]], columns=ART["cl_cols"])
    return int(KM.predict(SCALER.transform(X))[0])


def segmen_dari_profil(age: float, income: float, gender: str):
    """Skor belanja belum diketahui: perkirakan segmen dengan Random Forest."""
    row = pd.DataFrame([{"age": age, "annual_income_k": income, "gender": gender}])[ART["NUM"] + ART["CAT"]]
    proba = CLF.predict_proba(row)[0]
    return {int(c): float(p) for c, p in zip(CLF.classes_, proba)}


st.title("🛍️ Segmentasi Pelanggan Mall")
st.write("Masukkan data pelanggan, lalu sistem menentukan segmen pelanggannya beserta saran strategi pemasaran.")

left, right = st.columns(2)
with left:
    gender = st.selectbox("Jenis kelamin", OPT["gender"], format_func=lambda g: "Perempuan" if g == "Female" else "Laki-laki")
    age = st.number_input("Usia (tahun)", min_value=1, max_value=100, value=int(DEF["age"]), step=1)
with right:
    income = st.number_input("Pendapatan tahunan (ribu $)", min_value=1, max_value=500, value=int(DEF["annual_income_k"]), step=1)
    known = st.checkbox("Spending score sudah diketahui")
    score = st.number_input("Spending score (1-100)", min_value=1, max_value=100, value=int(DEF["spending_score"]),
                            step=1, disabled=not known)

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

    p = PROFILE[seg_id]
    st.write(f"**Profil segmen:** {p['jumlah']:.0f} pelanggan, rata-rata usia {p['usia_rata2']:.0f} tahun, "
             f"pendapatan {p['pendapatan_rata2']:.0f} ribu $, spending score {p['spending_rata2']:.0f}.")
    st.write(f"**Saran strategi:** {ADVICE[seg_id]}")

    if not (RNG["age"][0] <= age <= RNG["age"][1]) or not (RNG["annual_income_k"][0] <= income <= RNG["annual_income_k"][1]):
        st.warning("Usia atau pendapatan di luar rentang data latih, hasil kurang dapat diandalkan.")

with st.expander("Ringkasan semua segmen"):
    st.dataframe(pd.DataFrame([{"Segmen": NAMES[c], "Jumlah": int(PROFILE[c]["jumlah"]),
                                "Usia rata-rata": round(PROFILE[c]["usia_rata2"], 1),
                                "Pendapatan (ribu $)": round(PROFILE[c]["pendapatan_rata2"], 1),
                                "Spending score": round(PROFILE[c]["spending_rata2"], 1)} for c in sorted(NAMES)]),
                 hide_index=True)

st.caption("Data: Mall Customer Segmentation Data (200 pelanggan). Segmen dibuat dengan K-Means (K=5), "
           "bukan label dari lapangan. Hasil adalah alat bantu analisis, bukan keputusan akhir.")
