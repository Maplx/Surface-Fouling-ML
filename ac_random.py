import os

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor


XLSX_PATH = "12-02-2025 raw sensor data.xlsx"
SHEET_NAME = "Sheet1"
RANDOM_SEED = 0
OUT_DIR = "outputs"


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Pick the first existing column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}")


def align_gas_to_ac(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treat CSV as two sequences:
      AC series: (AC-time, AC-right, AC-left)
      Gas series: (Time-gas, NO, NO2)
    Round gas time to integer seconds, and for each gas sample,
    interpolate AC signals at that rounded second on the AC time axis.

    Returns a gas_df with aligned targets and timestamps.
    """
    # Column names (support .1 variants)
    c_t_ac = pick_col(df, ["AC-time/s", "AC-time/s.1"])
    c_right = pick_col(df, ["AC-NO-600C-Right", "AC-NO-600C-Right.1"])
    c_left = pick_col(df, ["AC-NO-600C-Left", "AC-NO-600C-Left.1"])

    c_t_gas = pick_col(df, ["Time-gas (s)", "Time-gas (s).1"])
    c_no = pick_col(df, ["NO concentration (ppm)", "NO concentration (ppm).1"])
    c_no2 = pick_col(df, ["NO2 concentration (ppm)", "NO2 concentration (ppm).1"])

    # Build AC series
    ac_df = df[[c_t_ac, c_right, c_left]].copy().dropna()
    ac_df = ac_df.rename(columns={c_t_ac: "t_ac", c_right: "y_right", c_left: "y_left"})

    # Build Gas series
    gas_df = df[[c_t_gas, c_no, c_no2]].copy().dropna().reset_index(drop=True)
    gas_df = gas_df.rename(columns={c_t_gas: "t_gas", c_no: "NO", c_no2: "NO2"})

    print(f"[INFO] AC rows:  {len(ac_df)}")
    print(f"[INFO] Gas rows: {len(gas_df)}")

    # Round gas time to integer seconds
    gas_df["t_gas_round"] = gas_df["t_gas"].round().astype(int)

    # Prepare AC data for interpolation: sort and average duplicates by AC-time
    ac_sorted = (ac_df.groupby("t_ac", as_index=False)[["y_right", "y_left"]]
                 .mean()
                 .sort_values("t_ac")
                 .reset_index(drop=True))

    t_ac = ac_sorted["t_ac"].to_numpy()
    yR = ac_sorted["y_right"].to_numpy()
    yL = ac_sorted["y_left"].to_numpy()

    # Query times on AC timeline = rounded gas seconds
    t_query = gas_df["t_gas_round"].to_numpy().astype(float)

    t_min, t_max = float(t_ac.min()), float(t_ac.max())
    out_of_range = (t_query < t_min) | (t_query > t_max)
    num_oor = int(out_of_range.sum())

    print(f"[INFO] AC time range: [{t_min}, {t_max}]")
    print(f"[INFO] Gas rounded seconds range: [{t_query.min()}, {t_query.max()}]")
    print(f"[INFO] Out-of-range queries: {num_oor}")

    if num_oor > 0:
        print("[WARN] Some rounded gas times are outside AC range; clipping to preserve sample size.")
        t_query = np.clip(t_query, t_min, t_max)

    gas_df["t_ac_used"] = t_query
    gas_df["AC_right_aligned"] = np.interp(t_query, t_ac, yR)
    gas_df["AC_left_aligned"] = np.interp(t_query, t_ac, yL)

    print(f"[INFO] Aligned samples: {len(gas_df)} (should equal gas rows)")
    print("\n[Aligned sample preview]")
    print(gas_df[["t_gas", "t_gas_round", "t_ac_used", "NO", "NO2",
                  "AC_right_aligned", "AC_left_aligned"]].head(8))

    return gas_df


def random_split(gas_df: pd.DataFrame, test_size: float = 0.5):
    """
    Random 50/50 split (shuffle).
    X = [AC_right_aligned, AC_left_aligned]
    y = [NO, NO2]
    """
    d = gas_df.copy()
    X = d[["AC_right_aligned", "AC_left_aligned"]].to_numpy()
    y = d[["NO", "NO2"]].to_numpy()
    t = d["t_gas_round"].to_numpy()

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=test_size, shuffle=True, random_state=RANDOM_SEED
    )
    return X_train, y_train, X_test, y_test, t_test


def eval_model(name: str, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
    mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")
    rmse = np.sqrt(mse)
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
# Main
# -----------------------------
df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)
gas_df = align_gas_to_ac(df)

# Inputs = AC (right/left), Outputs = NO/NO2
X_train, y_train, X_test, y_test, t_test = random_split(gas_df, test_size=0.5)
print(f"\n[INFO] Train size = {len(y_train)}, Test size = {len(y_test)}")

# Define model zoo (no feature engineering)
models = [
    ("LinearRegression", LinearRegression()),
    ("Ridge(alpha=1.0)", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
    ("Lasso(alpha=1e-3)", Lasso(alpha=1e-3, random_state=RANDOM_SEED, max_iter=10000)),
    ("ElasticNet(alpha=1e-3,l1=0.5)", ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=10000)),

    # Needs scaling
    ("SVR(RBF)", MultiOutputRegressor(Pipeline([("scaler", StandardScaler()),
                          ("svr", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.001))]))),
    ("KNN(k=5)", Pipeline([("scaler", StandardScaler()),
                           ("knn", KNeighborsRegressor(n_neighbors=5))])),

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
    except Exception as e:
        results.append({"model": name, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan})
        print(f"[WARN] Model failed: {name} -> {type(e).__name__}: {e}")

res_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)

print("\n===== MODEL COMPARISON (NO/NO2 OUTPUTS) - RANDOM 50/50 =====")
print(res_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

os.makedirs(OUT_DIR, exist_ok=True)
res_path = os.path.join(OUT_DIR, "ac_random_models.csv")
res_df.to_csv(res_path, index=False)
print(f"[INFO] Saved model comparison to {res_path}")

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Plot best model on TEST set
# -----------------------------

# 1) Find best model name from results
best_name = res_df.iloc[0]["model"]
best_name_plot = best_name
print(f"\n[INFO] Best model by R2: {best_name}")

# 2) Retrieve the actual model object from your models list
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
    if best_name.startswith("KNN"):
        best_name_plot = (
            f"KNN(k={best_params['knn__n_neighbors']},"
            f"w={best_params['knn__weights']},p={best_params['knn__p']})"
        )

# 3) Fit best model and predict on test
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)

# 4) Sort test points by time for a clean line plot
order = np.argsort(t_test)
t_sorted = t_test[order]
y_sorted = y_test[order]
pred_sorted = y_test_pred[order]

# 5) Print a small preview for sanity check
print("\n[TEST sample preview sorted by time]")
preview = min(12, len(t_sorted))
for i in range(preview):
    print(
        f"t={t_sorted[i]:.0f}  "
        f"NO_true={y_sorted[i, 0]:.6f}  NO_pred={pred_sorted[i, 0]:.6f}  "
        f"NO2_true={y_sorted[i, 1]:.6f}  NO2_pred={pred_sorted[i, 1]:.6f}"
    )

# 6) Plot: True vs Predicted (test) over time for NO and NO2
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(t_sorted, y_sorted[:, 0], label="NO True (test)", linewidth=2)
axes[0].plot(t_sorted, pred_sorted[:, 0], label="NO Predicted (test)", linewidth=2)
axes[0].set_ylabel("NO (ppm)")
axes[0].set_title(f"Best Model Test Fit (NO): {best_name_plot}")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(t_sorted, y_sorted[:, 1], label="NO2 True (test)", linewidth=2)
axes[1].plot(t_sorted, pred_sorted[:, 1], label="NO2 Predicted (test)", linewidth=2)
axes[1].set_xlabel("t_gas_round (s)")
axes[1].set_ylabel("NO2 (ppm)")
axes[1].set_title(f"Best Model Test Fit (NO2): {best_name_plot}")
axes[1].grid(True)
axes[1].legend()

fig_path = os.path.join(OUT_DIR, "ac_random_best.png")
plt.tight_layout()
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"[INFO] Saved plot to {fig_path}")

