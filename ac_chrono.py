import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor


CSV_PATH = "12-02-2025 raw sensor data.csv"
RANDOM_SEED = 0


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


def chronological_split(gas_df: pd.DataFrame, target_col: str):
    """
    50/50 chronological split based on rounded gas time.
    X = [NO, NO2]
    y = aligned target column (Right or Left)
    """
    d = gas_df.sort_values("t_gas_round").reset_index(drop=True)
    X = d[["NO", "NO2"]].to_numpy()
    y = d[target_col].to_numpy()
    t = d["t_gas_round"].to_numpy()

    n = len(d)
    train_ratio = 0.5
    split = int(n * train_ratio)

    return X[:split], y[:split], X[split:], y[split:], t[split:]


def eval_model(name: str, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


# -----------------------------
# Main
# -----------------------------
df = pd.read_csv(CSV_PATH, low_memory=False)
gas_df = align_gas_to_ac(df)

# RIGHT target first (as you asked)
X_train, y_train, X_test, y_test, t_test = chronological_split(gas_df, "AC_right_aligned")
print(f"\n[INFO] Train size = {len(y_train)}, Test size = {len(y_test)}")

# Define model zoo (no feature engineering)
models = [
    ("LinearRegression", LinearRegression()),
    ("Ridge(alpha=1.0)", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
    ("Lasso(alpha=1e-3)", Lasso(alpha=1e-3, random_state=RANDOM_SEED, max_iter=10000)),
    ("ElasticNet(alpha=1e-3,l1=0.5)", ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=10000)),

    # Needs scaling
    ("SVR(RBF)", Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.001))])),
    ("KNN(k=5)", Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=5))])),

    ("DecisionTree", DecisionTreeRegressor(random_state=RANDOM_SEED, max_depth=5)),
    ("RandomForest", RandomForestRegressor(
        n_estimators=500, random_state=RANDOM_SEED, max_depth=None, min_samples_leaf=2
    )),

    ("GradientBoosting", GradientBoostingRegressor(random_state=RANDOM_SEED)),
    ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=RANDOM_SEED)),
]

results = []
for name, model in models:
    try:
        results.append(eval_model(name, model, X_train, y_train, X_test, y_test))
    except Exception as e:
        results.append({"model": name, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan})
        print(f"[WARN] Model failed: {name} -> {type(e).__name__}: {e}")

res_df = pd.DataFrame(results)
res_df = res_df.sort_values("R2", ascending=False).reset_index(drop=True)

print("\n===== MODEL COMPARISON (RIGHT) =====")
print(res_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

