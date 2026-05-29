"""
One-shot reorganization + regeneration for soot_mass_resistance_only/.

What it does:
1. Wipes the old data/ and figures/ subfolders.
2. Builds a clean folder layout (datasets/, main_results/, extras/).
3. Rebuilds the 379-row native dataset (with Temp-DC) and 17425-row dense one.
4. Runs the 6-cell matrix: features in {R, T, R+T} x split in {random, time}.
5. Saves per-cell model leaderboards (10 models), per-cell best predictions,
   and per-cell best-model true-vs-predicted plots.
6. Saves a combined 2x3 panel figure summarizing all 6 cells.
7. Moves the side experiments (dense R-only, R->T) into extras/.

Run from repo root:  python soot_mass_resistance_only/regenerate_all.py
"""

import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

XLSX_PATH = "12-02-2025 raw sensor data.xlsx"
SHEET_RAW = "Sheet1"
SHEET_CALC = "calculation"
SEED = 0
TIME_CUTOFF = 17500.0
TRAIN_FRAC_TIME = 0.7
ROOT = "soot_mass_resistance_only"


def model_zoo():
    return [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge(alpha=1.0, random_state=SEED)),
        ("Lasso", Lasso(alpha=1e-3, random_state=SEED, max_iter=10000)),
        ("ElasticNet", ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=SEED, max_iter=10000)),
        ("SVR(RBF)", Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.001)),
        ])),
        ("KNN(k=5)", Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=5)),
        ])),
        ("DecisionTree", DecisionTreeRegressor(random_state=SEED, max_depth=5)),
        ("RandomForest", RandomForestRegressor(
            n_estimators=500, random_state=SEED, max_depth=None, min_samples_leaf=2
        )),
        ("GradientBoosting", GradientBoostingRegressor(random_state=SEED)),
        ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=SEED)),
    ]


def build_native_379():
    raw = pd.read_excel(XLSX_PATH, sheet_name=SHEET_RAW)
    calc = pd.read_excel(XLSX_PATH, sheet_name=SHEET_CALC)

    page7 = raw[["Time-DC", "Temp-DC", "CO2-DC"]].dropna().reset_index(drop=True)
    assert len(page7) == len(calc), f"row mismatch: {len(page7)} vs {len(calc)}"

    df = pd.concat([page7, calc.reset_index(drop=True)], axis=1)
    df = df.sort_values("Time-DC").reset_index(drop=True)
    df = df[df["Time-DC"] <= TIME_CUTOFF].reset_index(drop=True)

    rs = (raw[["TIME-sensor", "Resistance"]]
          .dropna()
          .groupby("TIME-sensor", as_index=False)["Resistance"]
          .mean()
          .sort_values("TIME-sensor")
          .reset_index(drop=True))

    df["Resistance"] = np.interp(
        df["Time-DC"].to_numpy(),
        rs["TIME-sensor"].to_numpy(),
        rs["Resistance"].to_numpy(),
    )
    return df[["Time-DC", "Temp-DC", "Resistance", "Soot left-mg"]].copy()


def build_dense_17425(native_df):
    raw = pd.read_excel(XLSX_PATH, sheet_name=SHEET_RAW)
    rs = (raw[["TIME-sensor", "Resistance"]]
          .dropna()
          .groupby("TIME-sensor", as_index=False)["Resistance"]
          .mean()
          .sort_values("TIME-sensor")
          .reset_index(drop=True))

    t_lo = max(float(native_df["Time-DC"].min()), float(rs["TIME-sensor"].min()))
    t_hi = min(float(native_df["Time-DC"].max()), float(rs["TIME-sensor"].max()))
    t_grid = np.arange(int(np.ceil(t_lo)), int(np.floor(t_hi)) + 1, 1, dtype=float)

    R_dense = np.interp(t_grid,
                        rs["TIME-sensor"].to_numpy(),
                        rs["Resistance"].to_numpy())
    S_dense = np.interp(t_grid,
                        native_df["Time-DC"].to_numpy(),
                        native_df["Soot left-mg"].to_numpy())

    return pd.DataFrame({"t": t_grid, "Resistance": R_dense, "Soot left-mg": S_dense})


def run_cell(X, y, t, split_kind):
    """Fit zoo, return leaderboard + best-model handle + train/test arrays."""
    if split_kind == "random":
        Xtr, Xte, ytr, yte, ttr, tte = train_test_split(
            X, y, t, test_size=0.5, shuffle=True, random_state=SEED
        )
    else:
        o = np.argsort(t)
        Xs, ys, ts = X[o], y[o], t[o]
        n_tr = int(round(len(ys) * TRAIN_FRAC_TIME))
        Xtr, Xte = Xs[:n_tr], Xs[n_tr:]
        ytr, yte = ys[:n_tr], ys[n_tr:]
        ttr, tte = ts[:n_tr], ts[n_tr:]

    rows = []
    best_r2 = -1e18
    best_name = best_pred = best_model = None
    for name, model in model_zoo():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        r2 = r2_score(yte, pred)
        rows.append({
            "model": name,
            "MAE": float(mean_absolute_error(yte, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
            "R2": float(r2),
        })
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pred = pred
            best_model = model

    lb = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return {
        "lb": lb,
        "best_name": best_name,
        "best_pred": best_pred,
        "best_model": best_model,
        "Xtr": Xtr, "ytr": ytr, "ttr": ttr,
        "Xte": Xte, "yte": yte, "tte": tte,
    }


def plot_one_cell(ax, t_test, y_true, y_pred, title, feat_cols):
    order = np.argsort(t_test)
    ax.plot(t_test[order], y_true[order], label="Measured", linewidth=2.0, color="tab:blue")
    ax.plot(t_test[order], y_pred[order], label="Predicted", linewidth=1.6,
            color="tab:orange", linestyle="--")
    ax.set_xlabel("Time-DC (s)")
    ax.set_ylabel("Soot left (mg)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def plot_time_split_full(ax, ttr, ytr, ytr_pred, tte, yte, yte_pred, title):
    """Full-view time-split plot: train true/pred + test true/pred + boundary line.
    Matches the style of outputs/dc_soot_mass_time_split_best.png."""
    ax.plot(ttr, ytr, label="Train True", linewidth=1.5, color="tab:blue")
    ax.plot(ttr, ytr_pred, label="Train Pred", linewidth=1.0,
            color="tab:cyan", linestyle="--")
    ax.plot(tte, yte, label="Test True", linewidth=2.0, color="tab:red")
    ax.plot(tte, yte_pred, label="Test Pred", linewidth=2.0,
            color="tab:orange", linestyle="--")
    ax.axvline(ttr.max(), color="gray", linestyle=":", label="train/test boundary")
    ax.set_xlabel("Time-DC (s)")
    ax.set_ylabel("Soot left (mg)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


# -----------------------------
# 0. Clean and create folder structure
# -----------------------------

old_data = os.path.join(ROOT, "data")
old_fig = os.path.join(ROOT, "figures")
if os.path.isdir(old_data):
    shutil.rmtree(old_data)
if os.path.isdir(old_fig):
    shutil.rmtree(old_fig)

new_dirs = [
    f"{ROOT}/datasets",
    f"{ROOT}/main_results/model_leaderboards",
    f"{ROOT}/main_results/best_predictions",
    f"{ROOT}/main_results/figures",
    f"{ROOT}/extras",
    f"{ROOT}/extras/dense_interpolation",
    f"{ROOT}/extras/predict_T_from_R",
]
for d in new_dirs:
    os.makedirs(d, exist_ok=True)


# -----------------------------
# 1. Datasets
# -----------------------------

native = build_native_379()
native.to_csv(f"{ROOT}/datasets/native_379.csv", index=False)
print(f"[INFO] Wrote datasets/native_379.csv ({native.shape})")

dense = build_dense_17425(native)
dense.to_csv(f"{ROOT}/datasets/dense_17425.csv", index=False)
print(f"[INFO] Wrote datasets/dense_17425.csv ({dense.shape})")


# -----------------------------
# 2. Main 6-cell matrix (R/T/R+T x random/time, native 379)
# -----------------------------

FEATURE_SETS = {
    "R":  ["Resistance"],
    "T":  ["Temp-DC"],
    "RT": ["Resistance", "Temp-DC"],
}
SPLITS = ["random", "time"]

PRETTY_FEATURES = {"R": "Resistance only", "T": "Temperature only", "RT": "Resistance + Temperature"}
PRETTY_SPLIT = {"random": "random 50/50", "time": "time 70/30"}

y_all = native["Soot left-mg"].to_numpy()
t_all = native["Time-DC"].to_numpy()

summary_rows = []
panel_data = {}  # (feat, split) -> (t_test, y_true, y_pred, best_name, r2, mae)
time_split_full = {}  # feat_key -> dict with train/test arrays for full-view plot

for feat_key, cols in FEATURE_SETS.items():
    X = native[cols].to_numpy()
    for sk in SPLITS:
        out = run_cell(X, y_all, t_all, sk)
        lb = out["lb"]; best_name = out["best_name"]; best_pred = out["best_pred"]
        y_test = out["yte"]; t_test = out["tte"]; X_test = out["Xte"]
        best_row = lb[lb["model"] == best_name].iloc[0]
        r2 = best_row["R2"]; mae = best_row["MAE"]; rmse = best_row["RMSE"]

        # leaderboard CSV
        lb.to_csv(f"{ROOT}/main_results/model_leaderboards/{feat_key}_{sk}.csv", index=False)

        # predictions CSV (true + pred at each test point)
        pred_df = pd.DataFrame({"Time-DC": t_test})
        for i, c in enumerate(cols):
            pred_df[c] = X_test[:, i]
        pred_df["Soot left-mg"] = y_test
        pred_df["Soot left-mg (pred)"] = best_pred
        pred_df.sort_values("Time-DC").to_csv(
            f"{ROOT}/main_results/best_predictions/{feat_key}_{sk}.csv", index=False
        )

        # per-cell figure (test-only zoomed view)
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_one_cell(ax, t_test, y_test, best_pred,
                      f"{PRETTY_FEATURES[feat_key]}, {PRETTY_SPLIT[sk]}: {best_name}  "
                      f"(R²={r2:+.4f}, MAE={mae:.4f} mg)",
                      cols)
        plt.tight_layout()
        plt.savefig(f"{ROOT}/main_results/figures/{feat_key}_{sk}.png", dpi=150)
        plt.close()

        # Stash time-split training data so we can do the full train+test view.
        if sk == "time":
            ytr_pred = out["best_model"].predict(out["Xtr"])
            time_split_full[feat_key] = dict(
                ttr=out["ttr"], ytr=out["ytr"], ytr_pred=ytr_pred,
                tte=t_test, yte=y_test, yte_pred=best_pred,
                best_name=best_name, r2=r2, mae=mae,
            )

        panel_data[(feat_key, sk)] = (t_test, y_test, best_pred, best_name, r2, mae)
        summary_rows.append({
            "features": PRETTY_FEATURES[feat_key],
            "feat_key": feat_key,
            "split": PRETTY_SPLIT[sk],
            "n_test": len(y_test),
            "best_model": best_name,
            "R2": r2,
            "MAE_mg": mae,
            "RMSE_mg": rmse,
            "test_y_std_mg": float(np.std(y_test)),
        })
        print(f"  {feat_key:<3} {sk:<6}  best={best_name:<22}  R^2={r2:+.4f}  MAE={mae:.4f}")

# Three full-view time-split plots (train + test, like dc_soot_mass_time_split_best.png)
for feat_key in FEATURE_SETS:
    d = time_split_full[feat_key]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_time_split_full(
        ax, d["ttr"], d["ytr"], d["ytr_pred"], d["tte"], d["yte"], d["yte_pred"],
        f"{PRETTY_FEATURES[feat_key]} — time 70/30, full train+test view\n"
        f"{d['best_name']}: test R²={d['r2']:+.4f}, MAE={d['mae']:.4f} mg",
    )
    plt.tight_layout()
    plt.savefig(f"{ROOT}/main_results/figures/{feat_key}_time_full.png", dpi=150)
    plt.close()
    print(f"  [full-view time] {feat_key}: wrote {feat_key}_time_full.png")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{ROOT}/main_results/summary_matrix.csv", index=False)

# 2x3 combined panel: rows = splits, cols = feature sets
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for ri, sk in enumerate(SPLITS):
    for ci, feat_key in enumerate(FEATURE_SETS):
        t_test, y_test, best_pred, bn, r2, mae = panel_data[(feat_key, sk)]
        plot_one_cell(axes[ri, ci], t_test, y_test, best_pred,
                      f"{PRETTY_FEATURES[feat_key]}\n{PRETTY_SPLIT[sk]}: {bn}\n"
                      f"R²={r2:+.4f},  MAE={mae:.4f} mg",
                      FEATURE_SETS[feat_key])
fig.suptitle("Soot mass prediction — all 6 feature/split combinations (native 379 rows, t ≤ 17500 s)",
             fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(f"{ROOT}/main_results/figures/_ALL_2x3_panel.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[INFO] Wrote main_results/figures/_ALL_2x3_panel.png")

print("\n===== SUMMARY MATRIX =====")
print(summary_df[["feat_key", "split", "best_model", "R2", "MAE_mg", "test_y_std_mg"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# -----------------------------
# 3. Extras: dense interpolation R-only (random + time) and R->T side experiment
# -----------------------------

# Dense R-only
y_d = dense["Soot left-mg"].to_numpy()
t_d = dense["t"].to_numpy()
X_d = dense[["Resistance"]].to_numpy()

for sk in SPLITS:
    out = run_cell(X_d, y_d, t_d, sk)
    lb, best_name, best_pred, yte, tte = out["lb"], out["best_name"], out["best_pred"], out["yte"], out["tte"]
    lb.to_csv(f"{ROOT}/extras/dense_interpolation/R_{sk}_models.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    r2 = r2_score(yte, best_pred); mae = mean_absolute_error(yte, best_pred)
    plot_one_cell(ax, tte, yte, best_pred,
                  f"Dense interp ({len(dense)} rows) - R only, {PRETTY_SPLIT[sk]}\n"
                  f"{best_name}: R²={r2:+.4f}, MAE={mae:.4f} mg",
                  ["Resistance"])
    plt.tight_layout()
    plt.savefig(f"{ROOT}/extras/dense_interpolation/R_{sk}_best.png", dpi=150)
    plt.close()
    print(f"  [extras/dense] R {sk}: {best_name}, R^2={r2:+.4f}")

# Predict T from R (side experiment)
X_rt = native[["Resistance"]].to_numpy()
y_t = native["Temp-DC"].to_numpy()
for sk in SPLITS:
    out = run_cell(X_rt, y_t, t_all, sk)
    lb, best_name, best_pred, yte, tte = out["lb"], out["best_name"], out["best_pred"], out["yte"], out["tte"]
    lb.to_csv(f"{ROOT}/extras/predict_T_from_R/{sk}_models.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    r2 = r2_score(yte, best_pred); mae = mean_absolute_error(yte, best_pred)
    o = np.argsort(tte)
    ax.plot(tte[o], yte[o], label="Measured Temp", linewidth=2.0, color="tab:red")
    ax.plot(tte[o], best_pred[o], label="Predicted Temp", linewidth=1.6,
            color="tab:orange", linestyle="--")
    ax.set_xlabel("Time-DC (s)"); ax.set_ylabel("Temp-DC (°C)")
    ax.set_title(f"Predict Temperature from Resistance, {PRETTY_SPLIT[sk]}\n"
                 f"{best_name}: R²={r2:+.4f}, MAE={mae:.2f} °C")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{ROOT}/extras/predict_T_from_R/{sk}_best.png", dpi=150)
    plt.close()
    print(f"  [extras/predict_T_from_R] {sk}: {best_name}, R^2={r2:+.4f}")

# Diagnostic R vs Soot scatter + time-series (rebuild from native_379)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sc = axes[0].scatter(native["Resistance"], native["Soot left-mg"],
                     c=native["Time-DC"], cmap="viridis", s=18)
axes[0].set_xlabel("Resistance (Ω)")
axes[0].set_ylabel("Soot left (mg)")
axes[0].set_title("Resistance vs Soot left  (color = time)")
axes[0].grid(True, alpha=0.3)
plt.colorbar(sc, ax=axes[0], label="Time-DC (s)")
ax2 = axes[1]
ax2.plot(native["Time-DC"], native["Resistance"], color="tab:blue", label="Resistance")
ax2.set_xlabel("Time-DC (s)"); ax2.set_ylabel("Resistance (Ω)", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")
ax2b = ax2.twinx()
ax2b.plot(native["Time-DC"], native["Soot left-mg"], color="tab:red", label="Soot left-mg")
ax2b.set_ylabel("Soot left (mg)", color="tab:red")
ax2b.tick_params(axis="y", labelcolor="tab:red")
ax2.set_title("Time series: Resistance vs Soot")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{ROOT}/extras/diagnostic_resistance_vs_soot.png", dpi=150)
plt.close()
print("[INFO] Wrote extras/diagnostic_resistance_vs_soot.png")

print("\n[DONE]")
