"""
Predict remaining soot mass (Soot left-mg, from the 'calculation' sheet) from
DC resistance and temperature, using the page-7 DC soot block on Sheet1.

Alignment logic:
  - The 'calculation' sheet has 412 rows of (CO2-DC-ppm, Soot left-%, Soot left-mg).
  - On Sheet1, the page-7 DC block has 412 jointly-non-null rows of
    (Time-DC, Temp-DC, CO2-DC). These rows align row-by-row with 'calculation'.
  - Resistance lives on a separate time axis (TIME-sensor, 1898 rows). We
    interpolate Resistance onto the 412 Time-DC values to get one R per
    (Temp-DC, Soot left-mg) row.
  - Features X = [Resistance, Temp-DC], target y = Soot left-mg. 412 samples.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
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

    # Page-7 DC block on Sheet1: (Time-DC, Temp-DC, CO2-DC) share one axis,
    # 412 jointly-non-null rows. The calculation sheet has 412 rows aligned
    # row-by-row with those, so we take the dropna() subset and concat.
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

    # Sanity-check the CO2 column matches between the two sheets.
    co2_diff = np.abs(page7["CO2-DC"].to_numpy() - calc["CO2-DC-ppm"].to_numpy())
    print(f"[INFO] CO2-DC vs CO2-DC-ppm max abs diff: {co2_diff.max():.6f} "
          f"(should be ~0 if rows really align)")

    df = pd.concat(
        [page7.reset_index(drop=True), calc.reset_index(drop=True)],
        axis=1,
    )

    # Resistance lives on TIME-sensor. Interpolate it onto Time-DC.
    rs = (raw[["TIME-sensor", "Resistance"]]
          .dropna()
          .groupby("TIME-sensor", as_index=False)["Resistance"]
          .mean()
          .sort_values("TIME-sensor")
          .reset_index(drop=True))

    t_dc = df["Time-DC"].to_numpy(dtype=float)
    t_rs = rs["TIME-sensor"].to_numpy(dtype=float)
    r = rs["Resistance"].to_numpy(dtype=float)

    # Clip queries to the resistance time range, then linear-interp.
    t_query = np.clip(t_dc, t_rs.min(), t_rs.max())
    df["Resistance"] = np.interp(t_query, t_rs, r)

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

X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
    X, y, t, test_size=0.5, shuffle=True, random_state=RANDOM_SEED
)

print(f"\n[INFO] Train size = {len(y_train)}, Test size = {len(y_test)}")

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
res_path = os.path.join(OUT_DIR, "dc_soot_mass_models.csv")
res_df.to_csv(res_path, index=False)

print("\n===== MODEL COMPARISON: DC Soot Left (mg) =====")
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

if best_name.startswith("KNN"):
    scorer = make_scorer(r2_score)
    grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }
    search = GridSearchCV(best_model, grid, scoring=scorer, cv=3, n_jobs=-1)
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
    search = GridSearchCV(best_model, grid, scoring=scorer, cv=3, n_jobs=-1)
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

print("\n===== BEST MODEL RESULTS (DC Soot Left mg) =====")
print(f"MAE  = {mae:.6f} mg")
print(f"RMSE = {rmse:.6f} mg")
print(f"R^2  = {r2:.6f}")

order = np.argsort(t_test)

plt.figure()
plt.plot(t_test[order], y_test[order], label="Soot left True", linewidth=2)
plt.plot(t_test[order], y_test_pred[order], label="Soot left Pred", linewidth=2)
plt.xlabel("Time-DC (s)")
plt.ylabel("Soot left (mg)")
plt.title(f"DC Soot Mass: True vs Predicted ({best_name_plot})")
plt.grid(True)
plt.legend()

fig_path = os.path.join(OUT_DIR, "dc_soot_mass_best.png")
plt.tight_layout()
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"[INFO] Saved plot to {fig_path}")
