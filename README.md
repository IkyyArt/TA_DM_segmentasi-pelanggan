# Segmentasi Pelanggan Mall (Streamlit)

Aplikasi web yang menentukan segmen pelanggan mall dan memberi saran strategi pemasaran dari data yang diisi pengguna. Proyek Tugas Akhir Data Mining (DM).

- **Data:** Mall Customer Segmentation Data (Kaggle), 200 pelanggan: gender, usia, pendapatan tahunan, spending score
- **Model:** K-Means (K=5) pada pendapatan + spending score; Random Forest untuk memperkirakan segmen pelanggan baru yang spending score-nya belum diketahui
- **Catatan:** segmen adalah hasil clustering, bukan label dari lapangan. Dataset kecil, jadi hasil hanya alat bantu analisis.
- **Notebook dan data** ada di Google Drive (folder `DM/`), bukan di repo ini.

## 4 Fitur Aplikasi
| Fitur | Isi | Fungsi di `streamlit_app.py` |
|---|---|---|
| 1. Penentuan Segmen Pelanggan | Input satu pelanggan, hasil segmen (K-Means atau Random Forest) | `fitur_1()` |
| 2. Saran Strategi Pemasaran | Profil, strategi, dan langkah konkret per segmen | `fitur_2()` |
| 3. Dashboard Visualisasi Segmen | Grafik sebaran, jumlah, ringkasan segmen, evaluasi model | `fitur_3()` |
| 4. Segmentasi Massal via CSV | Unggah CSV banyak pelanggan, unduh hasil segmen | `fitur_4()` |

## File
| File | Fungsi |
|---|---|
| `streamlit_app.py` | Aplikasi web |
| `model/mall_model.joblib` | Model hasil latih |
| `requirements.txt` | Library dan versinya (Python 3.11 ke atas) |
| `train_model.py` | Melatih ulang model dari CSV: `python train_model.py Mall_Customers.csv` |

## Menjalankan di komputer sendiri
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Hosting gratis di Streamlit Community Cloud
1. Buat repo **publik** baru di GitHub, lalu unggah `streamlit_app.py`, `requirements.txt`, `train_model.py`, `README.md`, dan folder `model/` ke folder utama repo (bukan di dalam subfolder).
2. Buka share.streamlit.io dan masuk dengan akun GitHub.
3. Klik **Create app**, pilih repo, branch `main`, dan file utama `streamlit_app.py`.
4. Di **Advanced settings**, pilih Python **3.12**.
5. Klik **Deploy**, tunggu beberapa menit, lalu salin link `.streamlit.app`.

Jika model gagal dimuat, penyebabnya hampir selalu beda versi scikit-learn: jalankan ulang `train_model.py` di lingkungan yang sama (butuh `Mall_Customers.csv` dari Drive).
