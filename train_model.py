"""
Melatih model segmentasi pelanggan mall untuk aplikasi web.

Pemakaian:
    python train_model.py Mall_Customers.csv

Hasil: model/mall_model.joblib (dipakai oleh streamlit_app.py)

Tahap cleaning, pemilihan fitur, K-Means (K=5), dan penamaan segmen sama dengan
notebook analisis_segmentasi_pelanggan_mall.ipynb. Dua model disimpan:
  1. K-Means pada pendapatan + spending score  -> segmen jika skor sudah diketahui
  2. Random Forest pada usia, pendapatan, gender -> perkiraan segmen pelanggan baru
     (skor belanja belum diketahui)
"""
import sys
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, silhouette_score

RANDOM_STATE = 42
K = 5
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "Mall_Customers.csv"

# ---------------------------------------------------------------- cleaning
df = pd.read_csv(CSV_PATH)
n0 = len(df)
df = df.rename(columns={"CustomerID": "customer_id", "Gender": "gender", "Age": "age",
                        "Annual Income (k$)": "annual_income_k",
                        "Spending Score (1-100)": "spending_score"})
df["gender"] = df["gender"].str.strip().str.title()
df = df.drop_duplicates("customer_id")
df = df[df.gender.isin(["Male", "Female"])]
df = df[df.age.between(18, 100)]
df = df[df.annual_income_k > 0]
df = df[df.spending_score.between(1, 100)].reset_index(drop=True)
print(f"Data: {n0} -> {len(df)} baris setelah cleaning")

# ------------------------------------------------- segmentasi (K-Means)
cl_cols = ["annual_income_k", "spending_score"]
cl_scaler = StandardScaler().fit(df[cl_cols])
Xc = cl_scaler.transform(df[cl_cols])
km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(Xc)
df["cluster"] = km.labels_
sil = float(silhouette_score(Xc, km.labels_))

df["gender_male"] = (df["gender"] == "Male").astype(int)
prof = df.groupby("cluster").agg(jumlah=("customer_id", "size"), usia_rata2=("age", "mean"),
                                 pendapatan_rata2=("annual_income_k", "mean"),
                                 spending_rata2=("spending_score", "mean"),
                                 persen_laki=("gender_male", "mean"))
prof["persen_laki"] = prof["persen_laki"] * 100

def level(v, lo, hi):
    return "rendah" if v < lo else ("tinggi" if v >= hi else "sedang")

NAMA_SEGMEN = {("tinggi", "tinggi"): "Prioritas", ("sedang", "sedang"): "Reguler",
               ("rendah", "tinggi"): "Antusias", ("tinggi", "rendah"): "Potensial",
               ("rendah", "rendah"): "Hemat"}
SARAN = {
    "Prioritas": "Pelanggan bernilai tertinggi: program loyalitas VIP, akses awal produk baru, layanan personal.",
    "Reguler": "Kelompok terbesar: promo musiman dan program poin untuk meningkatkan frekuensi belanja.",
    "Antusias": "Pendapatan terbatas tetapi gemar belanja: diskon, cicilan ringan, dan paket hemat.",
    "Potensial": "Mampu tetapi jarang belanja: kampanye personal, pengalaman premium, dan penawaran eksklusif untuk menarik minat.",
    "Hemat": "Sensitif harga dan jarang belanja: promo harga murah dan kupon, biaya pemasaran dijaga rendah.",
}

def beri_nama(pend, skor):
    return NAMA_SEGMEN.get((level(pend, 40, 70), level(skor, 40, 60)),
                           f"Pendapatan {level(pend, 40, 70)}, belanja {level(skor, 40, 60)}")

segment_names = {}
for c, r in prof.iterrows():
    nm = beri_nama(r["pendapatan_rata2"], r["spending_rata2"])
    segment_names[int(c)] = nm if nm not in segment_names.values() else f"{nm} {c}"
segment_profile = {int(c): prof.loc[c].to_dict() for c in prof.index}
segment_advice = {int(c): SARAN.get(segment_names[int(c)], "Pantau perilaku belanja segmen ini.") for c in prof.index}
print(f"K-Means K={K} | silhouette {sil:.3f}")

# ------------------------------------- klasifikasi segmen (tanpa spending score)
NUM = ["age", "annual_income_k"]
CAT = ["gender"]

def make_pipe():
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("model", RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE))])

X, y = df[NUM + CAT], df["cluster"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
cv = cross_validate(make_pipe(), X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                    scoring=["accuracy", "f1_macro"])
pipe = make_pipe().fit(X_tr, y_tr)
pred = pipe.predict(X_te)
metrics = {"cv_accuracy": float(cv["test_accuracy"].mean()), "cv_f1_macro": float(cv["test_f1_macro"].mean()),
           "test_accuracy": float(accuracy_score(y_te, pred)),
           "test_f1_macro": float(f1_score(y_te, pred, average="macro")),
           "silhouette": sil, "n_train": len(X_tr), "n_test": len(X_te)}
print("Evaluasi klasifikasi:", {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()})

# model akhir dilatih ulang pada seluruh data
pipe = make_pipe().fit(X, y)

# ----------------------------------------------------- info untuk formulir
options = {"gender": ["Female", "Male"]}
defaults = {"age": float(df.age.median()), "annual_income_k": float(df.annual_income_k.median()),
            "spending_score": float(df.spending_score.median())}
ranges = {"age": [float(df.age.min()), float(df.age.max())],
          "annual_income_k": [float(df.annual_income_k.min()), float(df.annual_income_k.max())]}

joblib.dump({"classifier": pipe, "NUM": NUM, "CAT": CAT, "metrics": metrics,
             "kmeans": km, "cl_scaler": cl_scaler, "cl_cols": cl_cols,
             "segment_names": segment_names, "segment_profile": segment_profile,
             "segment_advice": segment_advice, "options": options, "defaults": defaults,
             "ranges": ranges, "sklearn_version": sklearn.__version__},
            "model/mall_model.joblib", compress=3)
print("Tersimpan: model/mall_model.joblib | scikit-learn", sklearn.__version__)
print("Segmen:", {segment_names[c]: int(prof.loc[c, "jumlah"]) for c in prof.index})
