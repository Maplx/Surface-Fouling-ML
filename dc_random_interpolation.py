import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

CSV_PATH = "12-02-2025 raw sensor data.csv"
RANDOM_SEED = 0


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}")


# -----------------------------
# 1) Load + extract DC series and Gas series
# -----------------------------

df = pd.read_csv(CSV_PATH, low_memory=False)

c_t_dc = pick_col(df, ["Time-sensor (s)", "Time-sensor (s).1"])
c_right = pick_col(df, ["NO-600-Resistance (ohm)-Right", "NO-600-Resistance (ohm)-Right.1"])
c_left = pick_col(df, ["NO-600-Resistance (ohm)-left", "NO-600-Resistance (ohm)-left.1"])

c_t_gas = pick_col(df, ["Time-gas (s)", "Time-gas (s).1"])
c_no = pick_col(df, ["NO concentration (ppm)", "NO concentration (ppm).1"])
c_no2 = pick_col(df, ["NO2 concentration (ppm)", "NO2 concentration (ppm).1"])

# DC series
dc = df[[c_t_dc, c_right, c_left]].dropna().copy()
dc = dc.rename(columns={c_t_dc: "t_dc", c_right: "R_right", c_left: "R_left"})
dc = (dc.groupby("t_dc", as_index=False)[["R_right", "R_left"]]
      .mean()
      .sort_values("t_dc")
      .reset_index(drop=True))

# Gas series
gas = df[[c_t_gas, c_no, c_no2]].dropna().copy()
gas = gas.rename(columns={c_t_gas: "t_gas", c_no: "NO", c_no2: "NO2"})
gas = (gas.groupby("t_gas", as_index=False)[["NO", "NO2"]]
       .mean()
       .sort_values("t_gas")
       .reset_index(drop=True))

print(f"[INFO] DC points:  {len(dc)}  time range [{dc.t_dc.min()}, {dc.t_dc.max()}]")
print(f"[INFO] Gas points: {len(gas)} time range [{gas.t_gas.min()}, {gas.t_gas.max()}]")


# -----------------------------
# 2) Build dense time grid on DC clock (per-second), NO LAG
# -----------------------------

t_min = int(np.floor(dc["t_dc"].min()))
t_max = int(np.ceil(dc["t_dc"].max()))
t_grid = np.arange(t_min, t_max + 1, 1, dtype=float)

# Interpolate gas onto this grid (no lag)
gmin, gmax = float(gas["t_gas"].min()), float(gas["t_gas"].max())
t_gas_query = np.clip(t_grid, gmin, gmax)

NO_dense = np.interp(t_gas_query, gas["t_gas"].to_numpy(), gas["NO"].to_numpy())
NO2_dense = np.interp(t_gas_query, gas["t_gas"].to_numpy(), gas["NO2"].to_numpy())

# Interpolate DC onto same grid
R_right_dense = np.interp(t_grid, dc["t_dc"].to_numpy(), dc["R_right"].to_numpy())
R_left_dense = np.interp(t_grid, dc["t_dc"].to_numpy(), dc["R_left"].to_numpy())

dense = pd.DataFrame({
    "t": t_grid,
    "NO_interp": NO_dense,
    "NO2_interp": NO2_dense,
    "R_right": R_right_dense,
    "R_left": R_left_dense,
})

print(f"[INFO] Dense dataset rows: {len(dense)}")
print("\n[Dense preview]")
print(dense.head(8))


# -----------------------------
# 3) Random 50/50 split (shuffle)
# -----------------------------

X = dense[["R_right", "R_left"]].to_numpy()
y = dense[["NO_interp", "NO2_interp"]].to_numpy()
t = dense["t"].to_numpy()

X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
    X, y, t, test_size=0.5, shuffle=True, random_state=RANDOM_SEED
)

print(f"\n[INFO] Train size = {len(y_train)}, Test size = {len(y_test)}")


# -----------------------------
# 4) Train model + evaluate
# -----------------------------

model = MultiOutputRegressor(HistGradientBoostingRegressor(
    max_depth=4,
    learning_rate=0.05,
    max_iter=800,
    random_state=RANDOM_SEED
))
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_test_pred, multioutput="raw_values")
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred, multioutput="raw_values"))
r2 = r2_score(y_test, y_test_pred, multioutput="raw_values")

print("\n===== RESULTS: Dense Interpolation (NO LAG) + DC RANDOM SPLIT =====")
print(f"MAE  = {float(np.mean(mae)):.6f} (NO={mae[0]:.6f}, NO2={mae[1]:.6f})")
print(f"RMSE = {float(np.mean(rmse)):.6f} (NO={rmse[0]:.6f}, NO2={rmse[1]:.6f})")
print(f"R^2  = {float(np.mean(r2)):.6f} (NO={r2[0]:.6f}, NO2={r2[1]:.6f})")


# -----------------------------
# 5) Plots (test)
# -----------------------------

order_test = np.argsort(t_test)

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(t_test[order_test], y_test[order_test, 0], label="NO True (test)", linewidth=2)
axes[0].plot(t_test[order_test], y_test_pred[order_test, 0], label="NO Predicted (test)", linewidth=2)
axes[0].set_ylabel("NO (ppm)")
axes[0].set_title("DC Random Split: NO")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(t_test[order_test], y_test[order_test, 1], label="NO2 True (test)", linewidth=2)
axes[1].plot(t_test[order_test], y_test_pred[order_test, 1], label="NO2 Predicted (test)", linewidth=2)
axes[1].set_xlabel("Time (s) on DC grid")
axes[1].set_ylabel("NO2 (ppm)")
axes[1].set_title("DC Random Split: NO2")
axes[1].grid(True)
axes[1].legend()

plt.show()
