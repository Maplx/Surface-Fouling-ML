"""
DC soot mass prediction with a time-based train/test split.

Same dataset as dc_soot_mass.py (X = [Resistance, Temp-DC], y = Soot left-mg,
412 rows aligned with the 'calculation' sheet), but instead of a random 50/50
shuffle, samples are sorted by Time-DC and the first 70% (early time) go to
train, the last 30% (late time) to test. This stops temporally-adjacent points
from leaking between train and test, which is the failure mode that makes the
random-split R^2 look misleadingly high.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

XLSX_PATH = "12-02-2025 raw sensor data.xlsx"
SHEET_RAW = "Sheet1"
SHEET_CALC = "calculation"
RANDOM_SEED = 0
OUT_DIR = "outputs"
TRAIN_FRAC = 0.7


def eval_model(name: str, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {"model": name, "MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def build_dataset() -> pd.DataFrame:
    raw = pd.read_excel(XLSX_PATH, sheet_name=SHEET_RAW)
    calc = pd.read_excel(XLSX_PATH, sheet_name=SHEET_CALC)

    page7 = (raw[["Time-DC", "Temp-DC", "CO2-DC"]]
             .dropna()
             .reset_index(drop=True))

    print(f"[INFO] page-7 (Time-DC,Temp-DC,CO2-DC) rows: {len(page7)}")
    print(f"[INFO] calculation sheet rows:               {len(calc)}")

    if len(page7) != len(calc):
        raise RuntimeError(
            f"Row count mismatch: page-7 DC block has {len(page7)} rows but "
            f"'calculation' sheet has {len(calc)}. Cannot align row-by-row."
        )

    co2_diff = np.abs(page7["CO2-DC"].to_numpy() - calc["CO2-DC-ppm"].to_numpy())
    print(f"[INFO] CO2-DC vs CO2-DC-ppm max abs diff: {co2_diff.max():.6f} "
          f"(should be ~0 if rows really align)")

    df = pd.concat(
        [page7.reset_index(drop=True), calc.reset_index(drop=True)],
        axis=1,
    )

    rs = (raw[["TIME-sensor", "Resistance"]]
          .dropna()
          .groupby("TIME-sensor", as_index=False)["Resistance"]
          .mean()
          .sort_values("TIME-sensor")
          .reset_index(drop=True))

    t_dc = df["Time-DC"].to_numpy(dtype=float)
    t_rs = rs["TIME-sensor"].to_numpy(dtype=float)
    r = rs["Resistance"].to_numpy(dtype=float)

    t_query = np.clip(t_dc, t_rs.min(), t_rs.max())
    df["Resistance"] = np.interp(t_query, t_rs, r)

    df = df.sort_values("Time-DC").reset_index(drop=True)

    print("\n[Dataset preview]")
    print(df.head(8))

    return df


# -----------------------------
# Main
# -----------------------------

data = build_dataset()

X = data[["Resistance", "Temp-DC"]].to_numpy()
y = data["Soot left-mg"].to_numpy()
t = data["Time-DC"].to_numpy()

n = len(y)
n_train = int(round(n * TRAIN_FRAC))

X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]
t_train, t_test = t[:n_train], t[n_train:]

print(f"\n[INFO] Time-based split ({TRAIN_FRAC:.0%} train / {1-TRAIN_FRAC:.0%} test)")
print(f"[INFO] Train size = {len(y_train)}, Test size = {len(y_test)}")
print(f"[INFO] Train time range: [{t_train.min():.1f}, {t_train.max():.1f}] s")
print(f"[INFO] Test  time range: [{t_test.min():.1f}, {t_test.max():.1f}] s")
print(f"[INFO] Train Temp range: [{X_train[:,1].min():.1f}, {X_train[:,1].max():.1f}] degC")
print(f"[INFO] Test  Temp range: [{X_test[:,1].min():.1f}, {X_test[:,1].max():.1f}] degC")
print(f"[INFO] Train soot-mg range: [{y_train.min():.3f}, {y_train.max():.3f}]")
print(f"[INFO] Test  soot-mg range: [{y_test.min():.3f}, {y_test.max():.3f}]")

models = [
    ("LinearRegression", LinearRegression()),
    ("Ridge(alpha=1.0)", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
    ("Lasso(alpha=1e-3)", Lasso(alpha=1e-3, random_state=RANDOM_SEED, max_iter=10000)),
    ("ElasticNet(alpha=1e-3,l1=0.5)", ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=10000)),

    ("SVR(RBF)", Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.001)),
    ])),
    ("KNN(k=5)", Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=5)),
    ])),

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
    except Exception as exc:
        results.append({"model": name, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan})
        print(f"[WARN] Model failed: {name} -> {type(exc).__name__}: {exc}")

res_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)

os.makedirs(OUT_DIR, exist_ok=True)
res_path = os.path.join(OUT_DIR, "dc_soot_mass_time_split_models.csv")
res_df.to_csv(res_path, index=False)

print("\n===== MODEL COMPARISON: DC Soot Left (mg), 70/30 time split =====")
print(res_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print(f"[INFO] Saved model comparison to {res_path}")

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
best_name_plot = best_name

# Tuning uses TimeSeriesSplit (no shuffle) to stay consistent with the
# time-based train/test split — otherwise random CV would leak future into
# past during hyperparameter search.
tscv = TimeSeriesSplit(n_splits=5)

if best_name.startswith("KNN"):
    scorer = make_scorer(r2_score)
    grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }
    search = GridSearchCV(best_model, grid, scoring=scorer, cv=tscv, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_name_plot = (
        f"KNN(k={best_params['knn__n_neighbors']},"
        f"w={best_params['knn__weights']},p={best_params['knn__p']})"
    )
elif best_name == "RandomForest":
    scorer = make_scorer(r2_score)
    grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [1, 2, 4],
    }
    search = GridSearchCV(best_model, grid, scoring=scorer, cv=tscv, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    best_params = search.best_params_
else:
    print("[INFO] Skipping tuning for this model type.")

if best_params:
    print(f"[INFO] Best tuned params: {best_params}")

best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)

print("\n===== BEST MODEL RESULTS (DC Soot Left mg, 70/30 time split) =====")
print(f"MAE  = {mae:.6f} mg")
print(f"RMSE = {rmse:.6f} mg")
print(f"R^2  = {r2:.6f}")

# Plot: show train (true) and test (true vs predicted) on one time axis so the
# gap between training regime and held-out future is obvious.
y_train_pred = best_model.predict(X_train)

plt.figure(figsize=(9, 5))
plt.plot(t_train, y_train, label="Train True", linewidth=1.5, color="tab:blue")
plt.plot(t_train, y_train_pred, label="Train Pred", linewidth=1.0, color="tab:cyan", linestyle="--")
plt.plot(t_test, y_test, label="Test True", linewidth=2.0, color="tab:red")
plt.plot(t_test, y_test_pred, label="Test Pred", linewidth=2.0, color="tab:orange", linestyle="--")
plt.axvline(t_train.max(), color="gray", linestyle=":", label="train/test boundary")
plt.xlabel("Time-DC (s)")
plt.ylabel("Soot left (mg)")
plt.title(f"DC Soot Mass, 70/30 time split ({best_name_plot})")
plt.grid(True)
plt.legend()

fig_path = os.path.join(OUT_DIR, "dc_soot_mass_time_split_best.png")
plt.tight_layout()
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"[INFO] Saved plot to {fig_path}")
