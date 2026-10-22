# Segmentasi Pelanggan Mall (K-Means + Random Forest + Streamlit)

Tugas Akhir mata kuliah **Data Mining (DM)**.

Di proyek ini aku belajar mengelompokkan pelanggan mall ke dalam beberapa segmen supaya strategi promosinya bisa dibedakan per kelompok, nggak disamaratakan. Hasilnya aku pasang jadi aplikasi web pakai Streamlit: isi data pelanggan, lalu muncul segmen dan saran strateginya.

**Demo aplikasi:** _(tempel link `.streamlit.app` di sini setelah deploy berhasil)_

---

## Latar belakang

Kalau mall memberi promo yang sama ke semua pelanggan, hasilnya kurang tepat. Pelanggan berpenghasilan tinggi tapi jarang belanja jelas butuh pendekatan beda dengan anak muda berpenghasilan rendah yang hobi belanja. Karena itu aku coba memakai **clustering** untuk menemukan kelompok-kelompok pelanggan secara otomatis dari data.

## Dataset

**Mall Customer Segmentation Data** dari Kaggle, berisi **200 pelanggan** dengan 4 kolom:

| Kolom | Keterangan |
|---|---|
| `Gender` | Perempuan / Laki-laki |
| `Age` | Usia (tahun) |
| `Annual Income (k$)` | Pendapatan tahunan (ribu dolar) |
| `Spending Score (1-100)` | Skor belanja yang diberikan mall |

## Alur pengerjaan

1. **Pembersihan data**: cek missing value, duplikat, nilai tidak valid, dan outlier (outlier hanya ditandai, tidak dibuang, karena bisa jadi itu justru pelanggan yang ingin ditemukan).
2. **Feature engineering**: cek skewness, ternyata semua variabel numerik cukup simetris jadi tidak perlu transformasi log.
3. **EDA**: lihat distribusi, perbandingan gender, korelasi, dan hubungan pendapatan dengan spending score. Di grafik terlihat titik-titik membentuk kelompok, dan itu alasan utama pakai clustering.
4. **Uji hipotesis**: uji normalitas, Mann-Whitney U (gender), Kruskal-Wallis (kelompok usia), dan korelasi Spearman.
5. **Clustering K-Means**: dipilih **K = 5** lewat elbow dan silhouette (silhouette = **0,555**).
6. **Klasifikasi Random Forest**: untuk memperkirakan segmen pelanggan baru yang spending score-nya belum diketahui (akurasi cross-validation sekitar **80%**).
7. **Deploy** ke Streamlit Community Cloud.

## Lima segmen yang ditemukan

| Segmen | Jumlah | Usia rata-rata | Pendapatan (ribu $) | Spending score |
|---|---|---|---|---|
| Reguler | 81 | 43 | 55 | 50 |
| Prioritas | 39 | 33 | 87 | 82 |
| Antusias | 22 | 25 | 26 | 79 |
| Potensial | 35 | 41 | 88 | 17 |
| Hemat | 23 | 45 | 26 | 21 |

Nama segmen diturunkan dari tingkat pendapatan dan spending score rata-rata tiap klaster.

## Hal yang aku pelajari

- **Pilih K tidak boleh asal.** Elbow dan silhouette membantu, tapi tetap harus dicek apakah klasternya masuk akal secara bisnis.
- **Korelasi mendekati nol bukan berarti tidak ada hubungan.** Pendapatan dan spending score hampir tidak berkorelasi, tapi ternyata membentuk kelompok yang jelas.
- **Hati-hati kebocoran data (data leakage).** Label segmen dihitung dari spending score, jadi tidak boleh dipakai sebagai fitur untuk memprediksi segmen itu sendiri.
- **Data uji kecil, hasil gampang goyah.** Dengan 40 data uji, satu prediksi salah mengubah akurasi 2,5 poin. Makanya aku lebih percaya hasil cross-validation.
- **Memprediksi angka spending score langsung itu sulit** (R² rendah), tapi mengklasifikasikan ke segmen jauh lebih berguna.

## Keterbatasan

- Datanya cuma 200 baris, jadi hasilnya sebaiknya dianggap alat bantu analisis, bukan keputusan akhir.
- Segmen adalah hasil clustering, **bukan label asli dari lapangan**.
- Untuk pelanggan baru tanpa spending score, hasilnya hanya perkiraan.

## Struktur repo

```
├── model/
│   └── mall_model.joblib      # model hasil latih
├── streamlit_app.py           # aplikasi web
├── train_model.py             # melatih ulang model dari CSV
├── requirements.txt           # library dan versinya
└── README.md
```

> Notebook analisis lengkap (`analisis_segmentasi_pelanggan_mall.ipynb`) dan dataset (`Mall_Customers.csv`) bisa ditaruh di repo ini juga, atau di folder Google Drive `DM/`.

## Cara menjalankan di komputer sendiri

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Kalau mau melatih ulang model (butuh `Mall_Customers.csv`):

```bash
python train_model.py Mall_Customers.csv
```

> Jika muncul error model gagal dimuat, biasanya karena beda versi scikit-learn. Latih ulang model di lingkungan yang sama dengan perintah di atas.

## Teknologi

Python 3.12 · pandas · NumPy · scikit-learn · Streamlit · joblib

## Penulis

**IkyyArt** — Universitas Dian Nuswantoro
