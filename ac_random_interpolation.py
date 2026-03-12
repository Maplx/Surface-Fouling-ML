import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

CSV_PATH = "12-02-2025 raw sensor data.csv"
RANDOM_SEED = 0
OUT_DIR = "outputs"


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}")


def eval_model(name: str, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput="raw_values"))
    r2 = r2_score(y_test, y_pred, multioutput="raw_values")

    return {
        "model": name,
        "MAE": float(np.mean(mae)),
        "RMSE": float(np.mean(rmse)),
        "R2": float(np.mean(r2)),
        "MAE_NO": float(mae[0]),
        "MAE_NO2": float(mae[1]),
        "RMSE_NO": float(rmse[0]),
        "RMSE_NO2": float(rmse[1]),
        "R2_NO": float(r2[0]),
        "R2_NO2": float(r2[1]),
    }


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

# AC series
ac = df[[c_t_ac, c_right, c_left]].dropna().copy()
ac = ac.rename(columns={c_t_ac: "t_ac", c_right: "AC_right", c_left: "AC_left"})
ac = (ac.groupby("t_ac", as_index=False)[["AC_right", "AC_left"]]
      .mean()
      .sort_values("t_ac")
      .reset_index(drop=True))

# Gas series
gas = df[[c_t_gas, c_no, c_no2]].dropna().copy()
gas = gas.rename(columns={c_t_gas: "t_gas", c_no: "NO", c_no2: "NO2"})
gas = (gas.groupby("t_gas", as_index=False)[["NO", "NO2"]]
       .mean()
       .sort_values("t_gas")
       .reset_index(drop=True))

print(f"[INFO] AC points:  {len(ac)}  time range [{ac.t_ac.min()}, {ac.t_ac.max()}]")
print(f"[INFO] Gas points: {len(gas)} time range [{gas.t_gas.min()}, {gas.t_gas.max()}]")


# -----------------------------
# 2) Build dense time grid on AC clock (per-second), NO LAG
# -----------------------------

t_min = int(np.floor(ac["t_ac"].min()))
t_max = int(np.ceil(ac["t_ac"].max()))
t_grid = np.arange(t_min, t_max + 1, 1, dtype=float)

# Interpolate gas onto this grid (no lag)
gmin, gmax = float(gas["t_gas"].min()), float(gas["t_gas"].max())
t_gas_query = np.clip(t_grid, gmin, gmax)

NO_dense = np.interp(t_gas_query, gas["t_gas"].to_numpy(), gas["NO"].to_numpy())
NO2_dense = np.interp(t_gas_query, gas["t_gas"].to_numpy(), gas["NO2"].to_numpy())

# Interpolate AC onto same grid
AC_right_dense = np.interp(t_grid, ac["t_ac"].to_numpy(), ac["AC_right"].to_numpy())
AC_left_dense = np.interp(t_grid, ac["t_ac"].to_numpy(), ac["AC_left"].to_numpy())

dense = pd.DataFrame({
    "t": t_grid,
    "NO_interp": NO_dense,
    "NO2_interp": NO2_dense,
    "AC_right": AC_right_dense,
    "AC_left": AC_left_dense,
})

print(f"[INFO] Dense dataset rows: {len(dense)}")
print("\n[Dense preview]")
print(dense.head(8))


# -----------------------------
# 3) Random 50/50 split (shuffle)
# -----------------------------

X = dense[["AC_right", "AC_left"]].to_numpy()
y = dense[["NO_interp", "NO2_interp"]].to_numpy()
t = dense["t"].to_numpy()

X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
    X, y, t, test_size=0.5, shuffle=True, random_state=RANDOM_SEED
)

print(f"\n[INFO] Train size = {len(y_train)}, Test size = {len(y_test)}")


# -----------------------------
# 4) Model zoo + selection
# -----------------------------

models = [
    ("LinearRegression", LinearRegression()),
    ("Ridge(alpha=1.0)", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
    ("Lasso(alpha=1e-3)", Lasso(alpha=1e-3, random_state=RANDOM_SEED, max_iter=10000)),
    ("ElasticNet(alpha=1e-3,l1=0.5)", ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=10000)),

    ("SVR(RBF)", MultiOutputRegressor(Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.001)),
    ]))),
    ("KNN(k=5)", Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=5)),
    ])),

    ("DecisionTree", DecisionTreeRegressor(random_state=RANDOM_SEED, max_depth=5)),
    ("RandomForest", RandomForestRegressor(
        n_estimators=500, random_state=RANDOM_SEED, max_depth=None, min_samples_leaf=2
    )),

    ("GradientBoosting", MultiOutputRegressor(GradientBoostingRegressor(random_state=RANDOM_SEED))),
    ("HistGradientBoosting", MultiOutputRegressor(HistGradientBoostingRegressor(random_state=RANDOM_SEED))),
]

results = []
for name, model in models:
    try:
        results.append(eval_model(name, model, X_train, y_train, X_test, y_test))
    except Exception as exc:
        results.append({"model": name, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan})
        print(f"[WARN] Model failed: {name} -> {type(exc).__name__}: {exc}")

res_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)

os.makedirs(OUT_DIR, exist_ok=True)
res_path = os.path.join(OUT_DIR, "ac_random_interpolation_models.csv")
res_df.to_csv(res_path, index=False)

print("\n===== MODEL COMPARISON: AC Dense Interpolation + RANDOM =====")
print(res_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print(f"[INFO] Saved model comparison to {res_path}")


# -----------------------------
# 5) Tune the best model (limited search)
# -----------------------------

best_name = res_df.iloc[0]["model"]
print(f"\n[INFO] Best model by R2: {best_name}")

best_model = None
for name, mdl in models:
    if name == best_name:
        best_model = mdl
        break

if best_model is None:
    raise RuntimeError(f"Could not find model object for: {best_name}")

best_params = None

if best_name.startswith("KNN"):
    scorer = make_scorer(r2_score, multioutput="uniform_average")
    grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }
    search = GridSearchCV(best_model, grid, scoring=scorer, cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    best_params = search.best_params_
elif best_name == "RandomForest":
    scorer = make_scorer(r2_score, multioutput="uniform_average")
    grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [1, 2, 4],
    }
    search = GridSearchCV(best_model, grid, scoring=scorer, cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    best_params = search.best_params_
else:
    print("[INFO] Skipping tuning for this model type.")

if best_params:
    print(f"[INFO] Best tuned params: {best_params}")


# -----------------------------
# 6) Final fit + metrics + plots
# -----------------------------

best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_test_pred, multioutput="raw_values")
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred, multioutput="raw_values"))
r2 = r2_score(y_test, y_test_pred, multioutput="raw_values")

print("\n===== BEST MODEL RESULTS (AC DENSE + RANDOM) =====")
print(f"MAE  = {float(np.mean(mae)):.6f} (NO={mae[0]:.6f}, NO2={mae[1]:.6f})")
print(f"RMSE = {float(np.mean(rmse)):.6f} (NO={rmse[0]:.6f}, NO2={rmse[1]:.6f})")
print(f"R^2  = {float(np.mean(r2)):.6f} (NO={r2[0]:.6f}, NO2={r2[1]:.6f})")

order_test = np.argsort(t_test)

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(t_test[order_test], y_test[order_test, 0], label="NO True (test)", linewidth=2)
axes[0].plot(t_test[order_test], y_test_pred[order_test, 0], label="NO Predicted (test)", linewidth=2)
axes[0].set_ylabel("NO (ppm)")
axes[0].set_title(f"AC Dense Random: NO ({best_name})")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(t_test[order_test], y_test[order_test, 1], label="NO2 True (test)", linewidth=2)
axes[1].plot(t_test[order_test], y_test_pred[order_test, 1], label="NO2 Predicted (test)", linewidth=2)
axes[1].set_xlabel("Time (s) on AC grid")
axes[1].set_ylabel("NO2 (ppm)")
axes[1].set_title(f"AC Dense Random: NO2 ({best_name})")
axes[1].grid(True)
axes[1].legend()

fig_path = os.path.join(OUT_DIR, "ac_random_interpolation_best.png")
plt.tight_layout()
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"[INFO] Saved plot to {fig_path}")
