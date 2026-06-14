# 📊 Prediksi Pengeluaran Bulanan Mahasiswa UNTAD Berdasarkan Uang Saku dan Gaya Hidup
Tugas Statistika dan Probabilitas

Analisis **Regresi Berganda** untuk memprediksi total pengeluaran bulanan mahasiswa Universitas Tadulako (UNTAD) berdasarkan uang saku dan gaya hidup.

```
regresi_untad/
├── Dataset_Pengeluaran_Mahasiswa_Untad_100.csv   # Dataset lengkap (100 responden)
├── analisis_regresi_berganda.py                   # Analisis utama (scikit-learn)
├── requirements.txt
├── output/
│   ├── 01_actual_vs_predicted.png
│   ├── 02_residual_plot.png
│   ├── 03_koefisien.png
│   ├── 04_distribusi_y.png
│   ├── 05_correlation_heatmap.png
│   └── 06_boxplot_tempat_tinggal.png
└── README.md
```

## 📌 Variabel Penelitian
| Simbol | Variabel | Tipe |
|--------|----------|------|
| **Y**  | Total Pengeluaran Bulanan (Rp) | Numerik (target) |
| X1 | Uang Saku per Bulan (Rp) | Numerik |
| X2 | Tempat Tinggal | Kategorikal |
| X3 | Frekuensi Makan di Luar per Minggu | Numerik |
| X4 | Kepemilikan Kendaraan (1=Ya, 0=Tidak) | Biner |
| X5 | Pengeluaran Hiburan per Bulan (Rp) | Numerik |
| X6 | Semester | Numerik |

## 🚀 Cara Menjalankan
### 1. Clone repository
```bash
git clone https://github.com/username/regresi-untad.git
cd regresi-untad
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan analisis regresi
```bash
python analisis_regresi_berganda.py
```
Grafik hasil analisis akan tersimpan otomatis di folder `output/`.

## 📐 Model
Model **Regresi Berganda (Multiple Linear Regression)** dibangun dengan:

- **Preprocessing:** `StandardScaler` (normalisasi fitur numerik) + One-Hot Encoding untuk `X2_Tempat_Tinggal`
- **Model:** `LinearRegression` dari [scikit-learn](https://scikit-learn.org/)
- **Evaluasi:** R², RMSE, MAE, dan 5-Fold Cross Validation
- **Split data:** 80% latih / 20% uji

## 🔬 Referensi
- [scikit-learn: LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- Dataset: Survey Pengeluaran Mahasiswa UNTAD (Palu, Sulawesi Tengah)
