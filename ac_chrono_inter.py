import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_PATH = "12-02-2025 raw sensor data.csv"
RANDOM_SEED = 0


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}")


# -----------------------------
# 1) Load + extract AC series and Gas series
# -----------------------------
df = pd.read_csv(CSV_PATH, low_memory=False)

c_t_ac = pick_col(df, ["AC-time/s", "AC-time/s.1"])
c_right = pick_col(df, ["AC-NO-600C-Right", "AC-NO-600C-Right.1"])
c_left = pick_col(df, ["AC-NO-600C-Left", "AC-NO-600C-Left.1"])

c_t_gas = pick_col(df, ["Time-gas (s)", "Time-gas (s).1"])
c_no = pick_col(df, ["NO concentration (ppm)", "NO concentration (ppm).1"])
c_no2 = pick_col(df, ["NO2 concentration (ppm)", "NO2 concentration (ppm).1"])

# AC series (5k points)
ac = df[[c_t_ac, c_right, c_left]].dropna().copy()
ac = ac.rename(columns={c_t_ac: "t_ac", c_right: "AC_right", c_left: "AC_left"})
ac = (ac.groupby("t_ac", as_index=False)[["AC_right", "AC_left"]]
      .mean()
      .sort_values("t_ac")
      .reset_index(drop=True))

# Gas series (209 points)
gas = df[[c_t_gas, c_no, c_no2]].dropna().copy()
gas = gas.rename(columns={c_t_gas: "t_gas", c_no: "NO", c_no2: "NO2"})
gas = (gas.groupby("t_gas", as_index=False)[["NO", "NO2"]]
       .mean()
       .sort_values("t_gas")
       .reset_index(drop=True))

print(f"[INFO] AC points:  {len(ac)}  time range [{ac.t_ac.min()}, {ac.t_ac.max()}]")
print(f"[INFO] Gas points: {len(gas)} time range [{gas.t_gas.min()}, {gas.t_gas.max()}]")


# -----------------------------
# 2) Build dense time grid on AC clock (per-second)
#    No lag: gas at time t is used for sensor at time t
# -----------------------------
t_min = int(np.floor(ac["t_ac"].min()))
t_max = int(np.ceil(ac["t_ac"].max()))
t_grid = np.arange(t_min, t_max + 1, 1, dtype=float)  # ~ 2..10000

# Interpolate gas onto this grid (no lag)
gmin, gmax = float(gas["t_gas"].min()), float(gas["t_gas"].max())
t_gas_query = np.clip(t_grid, gmin, gmax)

NO_dense = np.interp(t_gas_query, gas["t_gas"].to_numpy(), gas["NO"].to_numpy())
NO2_dense = np.interp(t_gas_query, gas["t_gas"].to_numpy(), gas["NO2"].to_numpy())

# Interpolate AC onto same grid
AC_right_dense = np.interp(t_grid, ac["t_ac"].to_numpy(), ac["AC_right"].to_numpy())

dense = pd.DataFrame({
    "t": t_grid,
    "NO_interp": NO_dense,
    "NO2_interp": NO2_dense,
    "AC_right": AC_right_dense,
})

print(f"[INFO] Dense dataset rows: {len(dense)}")
print("\n[Dense preview]")
print(dense.head(10))


# -----------------------------
# 3) Train/test split (50/50 chronological)
#    X = [NO_interp, NO2_interp], y = AC_right
# -----------------------------
X = dense[["NO_interp", "NO2_interp"]].to_numpy()
y = dense["AC_right"].to_numpy()
t = dense["t"].to_numpy()

n = len(dense)
split = n // 2

X_train, y_train = X[:split], y[:split]
X_test, y_test, t_test = X[split:], y[split:], t[split:]

model = HistGradientBoostingRegressor(
    max_depth=4,
    learning_rate=0.05,
    max_iter=800,
    random_state=RANDOM_SEED
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\n===== RESULTS: Dense Interpolation (NO LAG) =====")
print(f"Train size = {len(y_train)}, Test size = {len(y_test)}")
print(f"MAE  = {mae:.6f}")
print(f"RMSE = {rmse:.6f}")
print(f"R^2  = {r2:.6f}")


y_train_pred = model.predict(X_train)
'''
# -----------------------------
# 5) Plot True vs Pred on TRAIN half
# -----------------------------
plt.figure()
plt.plot(t[:split], y_train, label="True (train)", linewidth=2)
plt.plot(t[:split], y_train_pred, label="Predicted (train)", linewidth=2)
plt.xlabel("Time (s) on AC grid")
plt.ylabel("AC_right")
plt.title("Dense Interpolation (no lag): True vs Predicted (Train Half)")
plt.grid(True)
plt.legend()
plt.show()
'''

# -----------------------------
# 4) Plot True vs Pred on test half
# -----------------------------
plt.figure()
plt.plot(t_test, y_test, label="True (test)", linewidth=2)
plt.plot(t_test, y_pred, label="Predicted (test)", linewidth=2)
plt.xlabel("Time (s) on AC grid")
plt.ylabel("AC_right")
plt.title("Dense Interpolation (no lag): True vs Predicted")
plt.grid(True)
plt.legend()
plt.show()

