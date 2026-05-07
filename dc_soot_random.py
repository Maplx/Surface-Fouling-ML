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

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "model": name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def build_aligned_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    c_t_res = pick_col(df, ["Time-sensor", "TIME-sensor", "Time-sensor (s)", "Time-sensor.1"])
    c_res = pick_col(df, ["Resistance", "Resistance.1"])

    c_t_temp = pick_col(df, ["Time-DC", "Time-DC.1"])
    c_temp = pick_col(df, ["Temp-DC", "Temp-DC.1", "Temp"])

    c_t_co2 = pick_col(df, ["Time-CO2", "Time-CO2.1"])
    c_co2 = pick_col(df, ["CO2-DC", "CO2-DC.1"])

    res_df = df[[c_t_res, c_res]].dropna().copy()
    res_df = res_df.rename(columns={c_t_res: "t_res", c_res: "resistance"})
    res_df["t_res"] = pd.to_numeric(res_df["t_res"], errors="coerce")
    res_df["resistance"] = pd.to_numeric(res_df["resistance"], errors="coerce")
    res_df = res_df.dropna()
    res_df = (res_df.groupby("t_res", as_index=False)[["resistance"]]
              .mean()
              .sort_values("t_res")
              .reset_index(drop=True))

    temp_df = df[[c_t_temp, c_temp]].dropna().copy()
    temp_df = temp_df.rename(columns={c_t_temp: "t_temp", c_temp: "temp"})
    temp_df["t_temp"] = pd.to_numeric(temp_df["t_temp"], errors="coerce")
    temp_df["temp"] = pd.to_numeric(temp_df["temp"], errors="coerce")
    temp_df = temp_df.dropna()
    temp_df = (temp_df.groupby("t_temp", as_index=False)[["temp"]]
               .mean()
               .sort_values("t_temp")
               .reset_index(drop=True))

    co2_df = df[[c_t_co2, c_co2]].dropna().copy()
    co2_df = co2_df.rename(columns={c_t_co2: "t_co2", c_co2: "co2"})
    co2_df["t_co2"] = pd.to_numeric(co2_df["t_co2"], errors="coerce")
    co2_df["co2"] = pd.to_numeric(co2_df["co2"], errors="coerce")
    co2_df = co2_df.dropna()
    co2_df = (co2_df.groupby("t_co2", as_index=False)[["co2"]]
              .mean()
              .sort_values("t_co2")
              .reset_index(drop=True))

    t_res = res_df["t_res"].to_numpy()
    res = res_df["resistance"].to_numpy()

    t_temp = temp_df["t_temp"].to_numpy()
    temp = temp_df["temp"].to_numpy()

    t_co2 = co2_df["t_co2"].to_numpy()
    co2 = co2_df["co2"].to_numpy()

    # Oxidation rate = d(CO2)/d(time)
    ox_rate = np.gradient(co2, t_co2)

    temp_aligned = np.interp(t_res, t_temp, temp)
    ox_aligned = np.interp(t_res, t_co2, ox_rate)

    aligned = pd.DataFrame({
        "t": t_res,
        "resistance": res,
        "temp": temp_aligned,
        "ox_rate": ox_aligned,
    })

    print(f"[INFO] Aligned rows: {len(aligned)}")
    print("\n[Aligned preview]")
    print(aligned.head(8))

    return aligned


# -----------------------------
# Main
# -----------------------------

df = pd.read_csv(CSV_PATH, low_memory=False)

data = build_aligned_dataframe(df)

X = data[["resistance", "temp"]].to_numpy()
y = data["ox_rate"].to_numpy()
t = data["t"].to_numpy()

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
res_path = os.path.join(OUT_DIR, "dc_soot_ox_random_models.csv")
res_df.to_csv(res_path, index=False)

print("\n===== MODEL COMPARISON: DC Ox Rate (Aligned) =====")
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

print("\n===== BEST MODEL RESULTS (DC Ox Rate Aligned) =====")
print(f"MAE  = {mae:.6f}")
print(f"RMSE = {rmse:.6f}")
print(f"R^2  = {r2:.6f}")

order = np.argsort(t_test)

plt.figure()
plt.plot(t_test[order], y_test[order], label="Ox Rate True", linewidth=2)
plt.plot(t_test[order], y_test_pred[order], label="Ox Rate Pred", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Ox Rate")
plt.title(f"Aligned DC: Ox Rate ({best_name_plot})")
plt.grid(True)
plt.legend()

fig_path = os.path.join(OUT_DIR, "dc_soot_ox_random_best.png")
plt.tight_layout()
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"[INFO] Saved plot to {fig_path}")
