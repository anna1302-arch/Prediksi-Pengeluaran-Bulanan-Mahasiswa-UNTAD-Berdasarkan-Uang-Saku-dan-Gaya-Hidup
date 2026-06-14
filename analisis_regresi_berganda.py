"""
analisis_regresi_berganda.py
==============================
Analisis Regresi Berganda – Pengeluaran Bulanan Mahasiswa UNTAD
Menggunakan scikit-learn (https://scikit-learn.org/)

Variabel:
  Y  : Total Pengeluaran Bulanan (Rp)
  X1 : Uang Saku per Bulan (Rp)
  X2 : Tempat Tinggal (kategorikal → one-hot)
  X3 : Frekuensi Makan di Luar per Minggu
  X4 : Kepemilikan Kendaraan (1=Ya, 0=Tidak)
  X5 : Pengeluaran Hiburan (Rp)
  X6 : Semester
"""

# ── 0. Import ─────────────────────────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("Dataset_Pengeluaran_Mahasiswa_Untad_100.csv")
print("=" * 60)
print("DATASET PENGELUARAN MAHASISWA UNTAD (n =", len(df), ")")
print("=" * 60)
print(df.head())
print("\nInfo:\n")
print(df.info())
print("\nStatistik deskriptif:")
print(df.describe().to_string())

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
# One-hot encode X2_Tempat_Tinggal
df_enc = pd.get_dummies(df, columns=["X2_Tempat_Tinggal"], drop_first=True)

fitur = [c for c in df_enc.columns
         if c not in ["No", "Y_Total_Pengeluaran_Rp"]]
X = df_enc[fitur].astype(float)
y = df_enc["Y_Total_Pengeluaran_Rp"].astype(float)

print("\nFitur yang digunakan:")
for f in fitur:
    print(" -", f)

# ── 3. Split data ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"\nData latih : {len(X_train)} sampel")
print(f"Data uji   : {len(X_test)} sampel")

# ── 4. Model pipeline (StandardScaler + LinearRegression) ────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])
pipeline.fit(X_train, y_train)

y_pred_train = pipeline.predict(X_train)
y_pred_test  = pipeline.predict(X_test)

# ── 5. Evaluasi ───────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n── {label} ──")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : Rp {rmse:,.0f}")
    print(f"  MAE  : Rp {mae:,.0f}")
    return r2, rmse, mae

print("\n" + "=" * 60)
print("HASIL EVALUASI MODEL")
print("=" * 60)
r2_tr, rmse_tr, mae_tr = evaluate(y_train, y_pred_train, "Data Latih")
r2_te, rmse_te, mae_te = evaluate(y_test,  y_pred_test,  "Data Uji")

# Cross-validation (5-fold)
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
print(f"\n── 5-Fold Cross Validation ──")
print(f"  R² tiap fold : {np.round(cv_scores, 4)}")
print(f"  R² rata-rata : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 6. Koefisien regresi ──────────────────────────────────────────────────────
reg   = pipeline.named_steps["regressor"]
coefs = pd.Series(reg.coef_, index=fitur).sort_values(key=abs, ascending=False)
intercept = reg.intercept_

print("\n" + "=" * 60)
print("KOEFISIEN REGRESI")
print("=" * 60)
print(f"Intercept (β₀) : Rp {intercept:,.0f}")
for i, (feat, val) in enumerate(coefs.items(), 1):
    print(f"  β{i:02d} ({feat:45s}) : {val:+,.2f}")

# ── 7. Persamaan regresi ringkas ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("PERSAMAAN REGRESI BERGANDA (skala terstandarisasi):")
print("=" * 60)
terms = [f"({v:+.4f})·{k}" for k, v in coefs.items()]
print(f"Ŷ = {intercept:.0f}\n  " + "\n  ".join(terms))

# ── 8. Visualisasi ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
RUPIAH = lambda x, _: f"Rp {x/1e6:.1f}jt"

# -- 8a. Actual vs Predicted ---------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test / 1e6, y_pred_test / 1e6,
           color="#2196F3", alpha=0.7, edgecolors="white", s=70, label="Data Uji")
mn, mx = y.min() / 1e6, y.max() / 1e6
ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Ideal (y = ŷ)")
ax.set_xlabel("Pengeluaran Aktual (juta Rp)")
ax.set_ylabel("Pengeluaran Prediksi (juta Rp)")
ax.set_title(f"Actual vs Predicted  (R² uji = {r2_te:.3f})")
ax.legend()
fig.tight_layout()
fig.savefig("output/01_actual_vs_predicted.png", dpi=150)
plt.close(fig)

# -- 8b. Residual plot ---------------------------------------------------------
residuals = y_test - y_pred_test
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_pred_test / 1e6, residuals / 1e6,
           color="#FF5722", alpha=0.7, edgecolors="white", s=60)
ax.axhline(0, color="black", lw=1.2, ls="--")
ax.set_xlabel("Prediksi (juta Rp)")
ax.set_ylabel("Residual (juta Rp)")
ax.set_title("Residual Plot")
fig.tight_layout()
fig.savefig("output/02_residual_plot.png", dpi=150)
plt.close(fig)

# -- 8c. Koefisien bar chart ---------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#4CAF50" if v > 0 else "#F44336" for v in coefs.values]
coefs.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Koefisien (skala baku)")
ax.set_title("Koefisien Regresi per Variabel")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig("output/03_koefisien.png", dpi=150)
plt.close(fig)

# -- 8d. Distribusi Y ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(y / 1e6, bins=20, color="#9C27B0", edgecolor="white", alpha=0.8)
ax.set_xlabel("Total Pengeluaran (juta Rp)")
ax.set_ylabel("Jumlah Mahasiswa")
ax.set_title("Distribusi Total Pengeluaran Bulanan")
fig.tight_layout()
fig.savefig("output/04_distribusi_y.png", dpi=150)
plt.close(fig)

# -- 8e. Correlation heatmap ---------------------------------------------------
num_cols = ["Y_Total_Pengeluaran_Rp", "X1_Uang_Saku_Rp",
            "X3_Makan_Luar_per_Minggu", "X4_Kendaraan_1Ya_0Tidak",
            "X5_Hiburan_Rp", "X6_Semester"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Korelasi Antar Variabel Numerik")
fig.tight_layout()
fig.savefig("output/05_correlation_heatmap.png", dpi=150)
plt.close(fig)

# -- 8f. Boxplot pengeluaran by tempat tinggal ---------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
order = (df.groupby("X2_Tempat_Tinggal")["Y_Total_Pengeluaran_Rp"]
         .median().sort_values(ascending=False).index)
df_plot = df.copy()
df_plot["Pengeluaran (juta Rp)"] = df_plot["Y_Total_Pengeluaran_Rp"] / 1e6
sns.boxplot(data=df_plot, x="X2_Tempat_Tinggal", y="Pengeluaran (juta Rp)",
            order=order, palette="Set2", ax=ax)
ax.set_xlabel("Tempat Tinggal")
ax.set_title("Pengeluaran Bulanan berdasarkan Tempat Tinggal")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig("output/06_boxplot_tempat_tinggal.png", dpi=150)
plt.close(fig)

# ── 9. Ringkasan akhir ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RINGKASAN AKHIR")
print("=" * 60)
print(f"  Jumlah data          : {len(df)} responden")
print(f"  Jumlah fitur         : {len(fitur)}")
print(f"  R² Data Latih        : {r2_tr:.4f}")
print(f"  R² Data Uji          : {r2_te:.4f}")
print(f"  RMSE Data Uji        : Rp {rmse_te:,.0f}")
print(f"  MAE  Data Uji        : Rp {mae_te:,.0f}")
print(f"  CV R² (5-fold) mean  : {cv_scores.mean():.4f}")
print("\n✅ Semua grafik tersimpan di folder output/")
