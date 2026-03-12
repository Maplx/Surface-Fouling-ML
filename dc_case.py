import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CSV_PATH = "12-02-2025 raw sensor data.csv"   
RANDOM_SEED = 42

# -----------------------------
# 1) Load raw CSV
# -----------------------------
df = pd.read_csv(CSV_PATH, low_memory=False)

# DC mode NOx @ 600C appears in the first ~209 rows with valid values.
# Extract the exact columns for that experiment.
cols = [
    "Time-sensor (s)",
    "NO-600-Resistance (ohm)-Right",
    "NO-600-Resistance (ohm)-left",
    "Time-gas (s)",
    "NO concentration (ppm)",
    "NO2 concentration (ppm)",
]
dc = df[cols].copy()

# Keep only complete rows
dc = dc.dropna().reset_index(drop=True)

print(f"[INFO] DC NOx dataset shape: {dc.shape}")
print(dc.head())

# -----------------------------
# 2) Build features (X) and target (y)
# -----------------------------
# Target: average resistance across left/right channels
dc["R_avg"] = (dc["NO-600-Resistance (ohm)-Right"] + dc["NO-600-Resistance (ohm)-left"]) / 2.0

# Features: NO, NO2, and total NOx
dc["NOx_total"] = dc["NO concentration (ppm)"] + dc["NO2 concentration (ppm)"]

X = dc[["NO concentration (ppm)", "NO2 concentration (ppm)", "NOx_total"]].to_numpy()
y = dc["R_avg"].to_numpy()

# -----------------------------
# 3) 50/50 Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.5,
    random_state=RANDOM_SEED,
    shuffle=True
)

print(f"[INFO] Train size = {len(y_train)}, Test size = {len(y_test)}")

# -----------------------------
# 4) Model (start simple): Ridge Regression
# -----------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# 5) Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== RESULTS: DC NOx @ 600C =====")
print(f"MAE  = {mae:.6f} ohm")
print(f"RMSE = {rmse:.6f} ohm")
print(f"R^2  = {r2:.4f}")

# -----------------------------
# 6) Plot: Prediction vs Ground Truth
# -----------------------------
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("True Resistance (ohm)")
plt.ylabel("Predicted Resistance (ohm)")
plt.title("DC NOx @ 600C: True vs Predicted Resistance")
plt.grid(True)
plt.show()

# Optional: Plot residuals
residuals = y_test - y_pred
plt.figure()
plt.hist(residuals, bins=20)
plt.xlabel("Residual (True - Predicted) [ohm]")
plt.ylabel("Count")
plt.title("Residual Distribution")
plt.grid(True)
plt.show()
