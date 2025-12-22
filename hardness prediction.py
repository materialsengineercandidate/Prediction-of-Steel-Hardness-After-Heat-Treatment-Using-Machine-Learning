import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score

# =================================================
# 0) DOSYA YOLU
# =================================================

BASE_PATH = r"C:\Users\Ayberk\Desktop\isilislem data"
input_file = os.path.join(BASE_PATH, "raw_tempering_data.csv")

# =================================================
# 1) DATA OKU
# =================================================

df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()

# =================================================
# 2) SICAKLIK KOLONUNU GÜVENLİ YAKALA
# =================================================

for col in df.columns:
    if "Tempering temperature" in col:
        df = df.rename(columns={col: "Tempering_Temperature_C"})

df = df.rename(columns={
    "Steel type": "Steel_Type",
    "Initial hardness (HRC) - post quenching": "Initial_Hardness_HRC",
    "Tempering time (s)": "Tempering_Time_s",
    "C (%wt)": "C",
    "Mn (%wt)": "Mn",
    "P (%wt)": "P",
    "S (%wt)": "S",
    "Si (%wt)": "Si",
    "Ni (%wt)": "Ni",
    "Cr (%wt)": "Cr",
    "Mo (%wt)": "Mo",
    "V (%wt)": "V",
    "Al (%wt)": "Al",
    "Cu (%wt)": "Cu",
    "Final hardness (HRC) - post tempering": "Final_Hardness_HRC"
})

# =================================================
# 3) SAYISAL DÖNÜŞÜMLER
# =================================================

numeric_cols = [
    "Initial_Hardness_HRC",
    "Tempering_Time_s",
    "Tempering_Temperature_C",
    "C","Mn","P","S","Si","Ni","Cr","Mo","V","Al","Cu",
    "Final_Hardness_HRC"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Initial hardness yoksa ortalama ile doldur
df["Initial_Hardness_HRC"] = df["Initial_Hardness_HRC"].fillna(
    df["Initial_Hardness_HRC"].mean()
)

# =================================================
# 4) GİRİŞ / ÇIKIŞ
# =================================================

X = df[
    [
        "Initial_Hardness_HRC",
        "Tempering_Time_s",
        "Tempering_Temperature_C",
        "C","Mn","P","S","Si","Ni","Cr","Mo","V","Al","Cu"
    ]
]

y = df["Final_Hardness_HRC"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# =================================================
# 5) MODELLER
# =================================================

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    
    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=100, gamma="scale"))
    ]),
    
    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        min_samples_leaf=5,
        random_state=42
    ),
    
    "Random Forest (FINAL)": RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42
    )
}

# =================================================
# 6) EĞİTİM & KARŞILAŞTIRMA
# =================================================

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = (mae, r2)

print("\n📊 MODEL KARŞILAŞTIRMA SONUÇLARI\n")
for m, (mae, r2) in results.items():
    print(f"{m:25s} | MAE: {mae:.2f} | R2: {r2:.3f}")

# =================================================
# 7) FINAL MODEL (RANDOM FOREST)
# =================================================

final_model = models["Random Forest (FINAL)"]

# =================================================
# 8) KULLANICIDAN GİRDİ AL
# =================================================

print("\n--- Yeni Isıl İşlem İçin Sertlik Tahmini ---")

user_input = {
    "Initial_Hardness_HRC": float(input("Quench sonrası sertlik (HRC): ")),
    "Tempering_Time_s": float(input("Temperleme süresi (s): ")),
    "Tempering_Temperature_C": float(input("Temperleme sıcaklığı (°C): ")),
    "C": float(input("C (%): ")),
    "Mn": float(input("Mn (%): ")),
    "P": float(input("P (%): ")),
    "S": float(input("S (%): ")),
    "Si": float(input("Si (%): ")),
    "Ni": float(input("Ni (%): ")),
    "Cr": float(input("Cr (%): ")),
    "Mo": float(input("Mo (%): ")),
    "V": float(input("V (%): ")),
    "Al": float(input("Al (%): ")),
    "Cu": float(input("Cu (%): "))
}

input_df = pd.DataFrame([user_input])

predicted_hrc = final_model.predict(input_df)

print(f"\n🔩 Tahmini Nihai Sertlik: {predicted_hrc[0]:.2f} HRC")
