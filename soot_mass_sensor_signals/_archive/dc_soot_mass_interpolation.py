"""
Predict remaining soot mass (Soot left-mg) from DC Resistance ONLY, using
dense per-second interpolation to expand the training set.

Why interpolation:
  - The native aligned dataset has only 412 (Resistance, Soot left-mg) pairs
    on the Time-DC axis (see dc_soot_mass.py / dc_soot_mass_report.py).
  - Mirroring dc_soot_random_interpolation.py / dc_random_interpolation.py,
    we build a per-second time grid and linearly interpolate both signals
    onto it. This gives ~10x more training rows for SVR/KNN-style models
    that benefit from denser sample support.
  - Caveat (same as the existing interpolation scripts): a random split on
    the dense grid lets adjacent seconds leak between train and test, which
    inflates R^2. We additionally run a time-based 70/30 split as an honest
    out-of-distribution check.

Pipeline:
  1. Build the 412-row aligned (Time-DC, Resistance, Soot left-mg) dataset
     exactly as dc_soot_mass.py does.
  2. Drop Time-DC > 17500s (the no-signal tail, same cutoff as
     dc_soot_mass_report.py).
  3. Build a per-second time grid over the kept Time-DC range.
  4. Interpolate Resistance from raw (TIME-sensor, Resistance) onto the grid.
  5. Interpolate Soot left-mg from the 412-point (Time-DC, Soot) onto the grid.
  6. X = [Resistance] only. y = Soot left-mg.
  7. Run the 10-model zoo under random 50/50 split AND 70/30 time split.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
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
TIME_CUTOFF = 17500.0
TRAIN_FRAC_TIME = 0.7


def eval_model(name: str, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {"model": name, "MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def build_dense_dataset() -> pd.DataFrame:
    raw = pd.read_excel(XLSX_PATH, sheet_name=SHEET_RAW)
    calc = pd.read_excel(XLSX_PATH, sheet_name=SHEET_CALC)

    # 412-row row-by-row alignment between page-7 DC block and 'calculation'.
    page7 = (raw[["Time-DC", "Temp-DC", "CO2-DC"]]
             .dropna()
             .reset_index(drop=True))
    if len(page7) != len(calc):
        raise RuntimeError(
            f"Row count mismatch: page-7 DC block has {len(page7)} rows but "
            f"'calculation' sheet has {len(calc)}."
        )

    aligned = pd.concat([page7.reset_index(drop=True),
                         calc.reset_index(drop=True)], axis=1)
    aligned = aligned.sort_values("Time-DC").reset_index(drop=True)
    print(f"[INFO] native aligned rows: {len(aligned)}")
    print(f"[INFO] Time-DC range:       [{aligned['Time-DC'].min():.1f}, "
          f"{aligned['Time-DC'].max():.1f}] s")

    # Cutoff the no-signal tail, consistent with dc_soot_mass_report.py.
    aligned = aligned[aligned["Time-DC"] <= TIME_CUTOFF].reset_index(drop=True)
    print(f"[INFO] after Time-DC <= {TIME_CUTOFF:g}s cutoff: {len(aligned)} rows")

    # Raw resistance source: (TIME-sensor, Resistance) on its own time axis.
    rs = (raw[["TIME-sensor", "Resistance"]]
          .dropna()
          .groupby("TIME-sensor", as_index=False)["Resistance"]
          .mean()
          .sort_values("TIME-sensor")
          .reset_index(drop=True))
    print(f"[INFO] raw resistance points: {len(rs)}  "
          f"time range [{rs['TIME-sensor'].min():.1f}, "
          f"{rs['TIME-sensor'].max():.1f}] s")

    # Per-second grid bounded by the intersection of:
    #   - kept Time-DC range (soot mass support)
    #   - raw resistance time range (avoid extrapolating resistance)
    t_lo = max(float(aligned["Time-DC"].min()), float(rs["TIME-sensor"].min()))
    t_hi = min(float(aligned["Time-DC"].max()), float(rs["TIME-sensor"].max()))
    t_grid = np.arange(int(np.ceil(t_lo)), int(np.floor(t_hi)) + 1, 1, dtype=float)
    print(f"[INFO] dense per-second grid: t in [{t_grid.min():.1f}, "
          f"{t_grid.max():.1f}] s, {len(t_grid)} rows")

    # Linear interpolation of both signals onto the shared grid.
    resistance_dense = np.interp(
        t_grid,
        rs["TIME-sensor"].to_numpy(dtype=float),
        rs["Resistance"].to_numpy(dtype=float),
    )
    soot_dense = np.interp(
        t_grid,
        aligned["Time-DC"].to_numpy(dtype=float),
        aligned["Soot left-mg"].to_numpy(dtype=float),
    )

    dense = pd.DataFrame({
        "t": t_grid,
        "Resistance": resistance_dense,
        "Soot left-mg": soot_dense,
    })

    print("\n[Dense preview]")
    print(dense.head(8))
    print(f"[INFO] dense rows: {len(dense)} (vs native {len(aligned)})")

    return dense


def model_zoo():
    return [
        ("LinearRegression", LinearRegression()),
        ("Ridge(alpha=1.0)", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        ("Lasso(alpha=1e-3)", Lasso(alpha=1e-3, random_state=RANDOM_SEED, max_iter=10000)),
        ("ElasticNet(alpha=1e-3,l1=0.5)", ElasticNet(
            alpha=1e-3, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=10000
        )),
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
            n_estimators=500, random_state=RANDOM_SEED,
            max_depth=None, min_samples_leaf=2,
        )),
        ("GradientBoosting", GradientBoostingRegressor(random_state=RANDOM_SEED)),
        ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=RANDOM_SEED)),
    ]


def tune_best(best_name, best_model, X_train, y_train, cv):
    """Light tuning for KNN and RandomForest; mirrors dc_soot_mass.py."""
    if best_name.startswith("KNN"):
        scorer = make_scorer(r2_score)
        grid = {
            "knn__n_neighbors": [3, 5, 7, 9, 11],
            "knn__weights": ["uniform", "distance"],
            "knn__p": [1, 2],
        }
        search = GridSearchCV(best_model, grid, scoring=scorer, cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        params = search.best_params_
        plot_name = (
            f"KNN(k={params['knn__n_neighbors']},"
            f"w={params['knn__weights']},p={params['knn__p']})"
        )
        return search.best_estimator_, params, plot_name
    if best_name == "RandomForest":
        scorer = make_scorer(r2_score)
        grid = {
            "n_estimators": [200, 500],
            "max_depth": [None, 8, 16],
            "min_samples_leaf": [1, 2, 4],
        }
        search = GridSearchCV(best_model, grid, scoring=scorer, cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_, best_name
    return best_model, None, best_name


def run_split(tag: str, dense: pd.DataFrame, split_kind: str):
    """split_kind in {'random', 'time'}."""
    X = dense[["Resistance"]].to_numpy()
    y = dense["Soot left-mg"].to_numpy()
    t = dense["t"].to_numpy()

    if split_kind == "random":
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            X, y, t, test_size=0.5, shuffle=True, random_state=RANDOM_SEED,
        )
        cv = 3
        split_label = "random 50/50"
    elif split_kind == "time":
        n_train = int(round(len(y) * TRAIN_FRAC_TIME))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        t_train, t_test = t[:n_train], t[n_train:]
        cv = TimeSeriesSplit(n_splits=5)
        split_label = f"time {int(TRAIN_FRAC_TIME*100)}/{100-int(TRAIN_FRAC_TIME*100)}"
    else:
        raise ValueError(split_kind)

    print(f"\n[INFO] === {tag} ({split_label}) ===")
    print(f"[INFO] Train size = {len(y_train)}, Test size = {len(y_test)}")
    if split_kind == "time":
        print(f"[INFO] Train time range: [{t_train.min():.1f}, {t_train.max():.1f}] s")
        print(f"[INFO] Test  time range: [{t_test.min():.1f}, {t_test.max():.1f}] s")

    results = []
    for name, model in model_zoo():
        try:
            results.append(eval_model(name, model, X_train, y_train, X_test, y_test))
        except Exception as exc:
            results.append({"model": name, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan})
            print(f"[WARN] Model failed: {name} -> {type(exc).__name__}: {exc}")

    res_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    res_path = os.path.join(OUT_DIR, f"dc_soot_mass_interpolation_{tag}_models.csv")
    res_df.to_csv(res_path, index=False)

    print(f"\n===== MODEL COMPARISON: Resistance-only, dense interp, {split_label} =====")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"[INFO] Saved model comparison to {res_path}")

    # Tune the best one and re-evaluate.
    best_name = res_df.iloc[0]["model"]
    print(f"\n[INFO] Best model by R2 ({tag}): {best_name}")

    base_model = next(m for n, m in model_zoo() if n == best_name)
    tuned_model, best_params, plot_name = tune_best(best_name, base_model, X_train, y_train, cv)
    if best_params:
        print(f"[INFO] Best tuned params: {best_params}")
    else:
        print("[INFO] Skipping tuning for this model type.")

    tuned_model.fit(X_train, y_train)
    y_test_pred = tuned_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)

    print(f"\n===== BEST MODEL ({tag}, {split_label}) =====")
    print(f"MAE  = {mae:.6f} mg")
    print(f"RMSE = {rmse:.6f} mg")
    print(f"R^2  = {r2:.6f}")

    # Plot
    fig_path = os.path.join(OUT_DIR, f"dc_soot_mass_interpolation_{tag}_best.png")
    plt.figure(figsize=(9, 5))
    if split_kind == "random":
        order = np.argsort(t_test)
        plt.plot(t_test[order], y_test[order], label="Soot left True", linewidth=2)
        plt.plot(t_test[order], y_test_pred[order], label="Soot left Pred",
                 linewidth=1.6, linestyle="--")
    else:
        y_train_pred = tuned_model.predict(X_train)
        plt.plot(t_train, y_train, label="Train True", linewidth=1.5, color="tab:blue")
        plt.plot(t_train, y_train_pred, label="Train Pred", linewidth=1.0,
                 color="tab:cyan", linestyle="--")
        plt.plot(t_test, y_test, label="Test True", linewidth=2.0, color="tab:red")
        plt.plot(t_test, y_test_pred, label="Test Pred", linewidth=2.0,
                 color="tab:orange", linestyle="--")
        plt.axvline(t_train.max(), color="gray", linestyle=":", label="train/test boundary")

    plt.xlabel("Time (s)")
    plt.ylabel("Soot left (mg)")
    plt.title(f"Dense DC Soot Mass (Resistance only, {split_label}): {plot_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot to {fig_path}")


# -----------------------------
# Main
# -----------------------------

dense = build_dense_dataset()

run_split("random", dense, "random")
run_split("time", dense, "time")

print("\n[INFO] Done.")
