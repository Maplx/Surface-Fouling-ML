"""
Generate the data + figures backing the soot-mass report.

Pipeline:
  1. Build the 412-row aligned dataset (Time-DC, Temp-DC, Resistance, Soot left-mg).
  2. Drop rows with Time-DC > 17500s (post-experiment tail with no useful signal).
  3. Random 50/50 train/test split (seed=0) — same split applied to both feature sets.
  4. Two feature sets:
        (A) X = [Resistance]
        (B) X = [Resistance, Temp-DC]
     Target in both: Soot left-mg.
  5. 10-model zoo, identical to dc_soot_mass.py.
  6. Correlation analysis (Pearson + Spearman) between Resistance, Temp-DC, Soot left-mg.
  7. Interpretability: RandomForest feature importance + partial dependence,
     Ridge coefficients on z-scored features.
  8. All outputs land under soot_mass_report/{data,figures}/.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.inspection import PartialDependenceDisplay
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
RANDOM_SEED = 0
TIME_CUTOFF = 17500.0

REPORT_DIR = "soot_mass_report"
DATA_DIR = os.path.join(REPORT_DIR, "data")
FIG_DIR = os.path.join(REPORT_DIR, "figures")


def build_dataset() -> pd.DataFrame:
    raw = pd.read_excel(XLSX_PATH, sheet_name=SHEET_RAW)
    calc = pd.read_excel(XLSX_PATH, sheet_name=SHEET_CALC)

    page7 = raw[["Time-DC", "Temp-DC", "CO2-DC"]].dropna().reset_index(drop=True)
    if len(page7) != len(calc):
        raise RuntimeError(f"row count mismatch: page7={len(page7)} calc={len(calc)}")

    df = pd.concat([page7, calc.reset_index(drop=True)], axis=1)

    rs = (raw[["TIME-sensor", "Resistance"]]
          .dropna()
          .groupby("TIME-sensor", as_index=False)["Resistance"]
          .mean()
          .sort_values("TIME-sensor")
          .reset_index(drop=True))

    t_dc = df["Time-DC"].to_numpy(dtype=float)
    t_rs = rs["TIME-sensor"].to_numpy(dtype=float)
    r = rs["Resistance"].to_numpy(dtype=float)
    df["Resistance"] = np.interp(np.clip(t_dc, t_rs.min(), t_rs.max()), t_rs, r)

    return df.sort_values("Time-DC").reset_index(drop=True)


def model_zoo():
    return [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        ("Lasso", Lasso(alpha=1e-3, random_state=RANDOM_SEED, max_iter=10000)),
        ("ElasticNet", ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=10000)),
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


def eval_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "model": name,
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
        "_y_pred": y_pred,
        "_model": model,
    }


# -----------------------------
# Main
# -----------------------------

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

data_full = build_dataset()
print(f"[INFO] raw aligned rows: {len(data_full)}  "
      f"time range [{data_full['Time-DC'].min():.1f}, {data_full['Time-DC'].max():.1f}] s")

data = data_full[data_full["Time-DC"] <= TIME_CUTOFF].reset_index(drop=True)
print(f"[INFO] after dropping t > {TIME_CUTOFF:g} s: {len(data)} rows")

data_view = data[["Time-DC", "Temp-DC", "Resistance", "Soot left-mg"]].copy()
data_view.to_csv(os.path.join(DATA_DIR, "full_dataset.csv"), index=False)

# -----------------------------
# 1) Correlation analysis
# -----------------------------

print("\n===== PAIRWISE CORRELATIONS =====")
corr_pairs = [
    ("Resistance", "Temp-DC"),
    ("Resistance", "Soot left-mg"),
    ("Temp-DC",    "Soot left-mg"),
]
corr_rows = []
for a, b in corr_pairs:
    pr, pp = pearsonr(data[a], data[b])
    sr, sp = spearmanr(data[a], data[b])
    corr_rows.append({"pair": f"{a} vs {b}",
                      "Pearson_r": pr, "Pearson_p": pp,
                      "Spearman_rho": sr, "Spearman_p": sp})
    print(f"  {a:<13} vs {b:<13}  Pearson={pr:+.4f} (p={pp:.2e})  Spearman={sr:+.4f}")

pd.DataFrame(corr_rows).to_csv(os.path.join(DATA_DIR, "correlations.csv"), index=False)

# Scatter plots colored by time
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (a, b) in zip(axes, corr_pairs):
    sc = ax.scatter(data[a], data[b], c=data["Time-DC"], cmap="viridis", s=14, alpha=0.85)
    pr, _ = pearsonr(data[a], data[b])
    sr, _ = spearmanr(data[a], data[b])
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title(f"{a} vs {b}\nPearson={pr:+.3f},  Spearman={sr:+.3f}")
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Time-DC (s)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "correlations.png"), dpi=150)
plt.close()

# Correlation heatmap (just the 3x3)
sub = data[["Resistance", "Temp-DC", "Soot left-mg"]]
corr_mat = sub.corr(method="pearson")
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels(corr_mat.columns, rotation=30, ha="right")
ax.set_yticks(range(3)); ax.set_yticklabels(corr_mat.columns)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{corr_mat.values[i, j]:+.2f}", ha="center", va="center",
                color="white" if abs(corr_mat.values[i, j]) > 0.5 else "black")
ax.set_title("Pearson correlation matrix")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()
corr_mat.to_csv(os.path.join(DATA_DIR, "correlation_matrix.csv"))

# Raw time-series plot of the three signals
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 7))
axes[0].plot(data["Time-DC"], data["Resistance"], color="tab:blue")
axes[0].set_ylabel("Resistance (Ω)")
axes[0].grid(True, alpha=0.3)
axes[1].plot(data["Time-DC"], data["Temp-DC"], color="tab:red")
axes[1].set_ylabel("Temperature (°C)")
axes[1].grid(True, alpha=0.3)
axes[2].plot(data["Time-DC"], data["Soot left-mg"], color="tab:green")
axes[2].set_xlabel("Time-DC (s)")
axes[2].set_ylabel("Soot left (mg)")
axes[2].grid(True, alpha=0.3)
fig.suptitle(f"Aligned time series after dropping t > {TIME_CUTOFF:g}s (n={len(data)})")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "timeseries.png"), dpi=150)
plt.close()

# -----------------------------
# 2) Ablation: random 50/50 split, two feature sets
# -----------------------------

y = data["Soot left-mg"].to_numpy()
t = data["Time-DC"].to_numpy()

feature_sets = {
    "resistance_only": ["Resistance"],
    "resistance_temp": ["Resistance", "Temp-DC"],
}

all_results = []
fits = {}   # (feature_set_name, model_name) -> (model, y_pred, test_pieces)

for fname, cols in feature_sets.items():
    X = data[cols].to_numpy()
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=0.5, shuffle=True, random_state=RANDOM_SEED
    )

    # Save train/test splits to CSV (for the report's reproducibility section)
    train_df = pd.DataFrame(X_train, columns=cols)
    train_df["Soot left-mg"] = y_train
    train_df["Time-DC"] = t_train
    train_df.to_csv(os.path.join(DATA_DIR, f"train_{fname}.csv"), index=False)

    test_df = pd.DataFrame(X_test, columns=cols)
    test_df["Soot left-mg"] = y_test
    test_df["Time-DC"] = t_test
    test_df.to_csv(os.path.join(DATA_DIR, f"test_{fname}.csv"), index=False)

    print(f"\n[INFO] feature set '{fname}': train={len(y_train)}, test={len(y_test)}")
    for name, model in model_zoo():
        try:
            r = eval_model(name, model, X_train, y_train, X_test, y_test)
        except Exception as exc:
            print(f"[WARN] {fname}/{name} failed: {type(exc).__name__}: {exc}")
            continue
        fits[(fname, name)] = (r.pop("_model"), r.pop("_y_pred"),
                               {"X_train": X_train, "y_train": y_train,
                                "X_test": X_test, "y_test": y_test,
                                "t_train": t_train, "t_test": t_test,
                                "cols": cols})
        r["features"] = fname
        all_results.append(r)

res = pd.DataFrame(all_results)
res.to_csv(os.path.join(DATA_DIR, "ablation_metrics_long.csv"), index=False)

wide_r2 = res.pivot(index="model", columns="features", values="R2")
wide_r2["delta_R2"] = wide_r2["resistance_temp"] - wide_r2["resistance_only"]
wide_r2 = wide_r2[["resistance_only", "resistance_temp", "delta_R2"]].sort_values("resistance_temp", ascending=False)
wide_r2.to_csv(os.path.join(DATA_DIR, "r2_wide.csv"))

wide_mae = res.pivot(index="model", columns="features", values="MAE")
wide_mae = wide_mae[["resistance_only", "resistance_temp"]].loc[wide_r2.index]
wide_mae.to_csv(os.path.join(DATA_DIR, "mae_wide.csv"))

wide_rmse = res.pivot(index="model", columns="features", values="RMSE")
wide_rmse = wide_rmse[["resistance_only", "resistance_temp"]].loc[wide_r2.index]
wide_rmse.to_csv(os.path.join(DATA_DIR, "rmse_wide.csv"))

print("\n===== TEST R^2 =====")
print(wide_r2.to_string(float_format=lambda x: f"{x:+.4f}"))
print("\n===== TEST MAE (mg) =====")
print(wide_mae.to_string(float_format=lambda x: f"{x:.4f}"))

# Ablation bar plot
fig, ax = plt.subplots(figsize=(10, 5.5))
models_sorted = wide_r2.index.tolist()
x = np.arange(len(models_sorted))
w = 0.38
ax.bar(x - w/2, wide_r2["resistance_only"].clip(lower=-0.1), w,
       label="Resistance only", color="tab:gray")
ax.bar(x + w/2, wide_r2["resistance_temp"].clip(lower=-0.1), w,
       label="Resistance + Temperature", color="tab:blue")
ax.set_xticks(x); ax.set_xticklabels(models_sorted, rotation=30, ha="right")
ax.set_ylabel(r"Test $R^2$ (clipped at $-0.1$)")
ax.set_title("Ablation: predicting remaining soot mass on the random 50/50 test set")
ax.axhline(0, color="black", linewidth=0.6)
ax.grid(True, alpha=0.3, axis="y")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "ablation_r2.png"), dpi=150)
plt.close()

# Best-model true vs predicted plot for each feature set
best_models = {}
for fname in feature_sets:
    set_results = res[res["features"] == fname].sort_values("R2", ascending=False)
    best_name = set_results.iloc[0]["model"]
    best_r2 = set_results.iloc[0]["R2"]
    best_models[fname] = best_name

    model, y_pred_test, pieces = fits[(fname, best_name)]
    order = np.argsort(pieces["t_test"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pieces["t_test"][order], pieces["y_test"][order],
            label="Measured (test)", linewidth=2.0, color="tab:blue")
    ax.plot(pieces["t_test"][order], y_pred_test[order],
            label=f"Predicted ({best_name})", linewidth=1.8,
            color="tab:orange", linestyle="--")
    ax.set_xlabel("Time-DC (s)")
    ax.set_ylabel("Remaining Soot Mass (mg)")
    title_features = "Resistance only" if fname == "resistance_only" else "Resistance + Temperature"
    ax.set_title(f"Best model under {title_features}: {best_name}  (test $R^2$ = {best_r2:+.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"best_pred_{fname}.png"), dpi=150)
    plt.close()

# -----------------------------
# 3) Interpretability (resistance + temp model only)
# -----------------------------

cols = feature_sets["resistance_temp"]
pretty = {"Resistance": "Resistance", "Temp-DC": "Temperature"}
pretty_cols = [pretty[c] for c in cols]

# RandomForest feature importance
rf_model = fits[("resistance_temp", "RandomForest")][0]
rf_importances = rf_model.feature_importances_

# GradientBoosting feature importance for cross-check
gb_model = fits[("resistance_temp", "GradientBoosting")][0]
gb_importances = gb_model.feature_importances_

fig, ax = plt.subplots(figsize=(7, 4.5))
xpos = np.arange(len(cols))
w = 0.38
ax.bar(xpos - w/2, rf_importances, w, label="RandomForest", color="tab:blue")
ax.bar(xpos + w/2, gb_importances, w, label="GradientBoosting", color="tab:orange")
ax.set_xticks(xpos); ax.set_xticklabels(pretty_cols)
ax.set_ylabel("Feature importance (sum to 1)")
ax.set_ylim(0, 1.05)
ax.set_title("Feature importance under the Resistance + Temperature model")
for i, (a, b) in enumerate(zip(rf_importances, gb_importances)):
    ax.text(i - w/2, a + 0.02, f"{a:.2f}", ha="center")
    ax.text(i + w/2, b + 0.02, f"{b:.2f}", ha="center")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=150)
plt.close()

# Ridge coefficients on z-scored features (for fair comparison of magnitudes)
pieces_rt = fits[("resistance_temp", "Ridge")][2]
scaler = StandardScaler().fit(pieces_rt["X_train"])
X_train_s = scaler.transform(pieces_rt["X_train"])
ridge_scaled = Ridge(alpha=1.0, random_state=RANDOM_SEED).fit(X_train_s, pieces_rt["y_train"])
ridge_coefs_scaled = ridge_scaled.coef_

# Partial dependence plot (best tree model on R+T)
best_rt = best_models["resistance_temp"]
pd_model = fits[("resistance_temp", best_rt)][0]
pd_pieces = fits[("resistance_temp", best_rt)][2]

try:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    PartialDependenceDisplay.from_estimator(
        pd_model, pd_pieces["X_train"], features=[0, 1],
        feature_names=pretty_cols, ax=ax, grid_resolution=60,
    )
    fig.suptitle(f"Partial dependence of soot mass on each feature ({best_rt})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "partial_dependence.png"), dpi=150)
    plt.close()
except Exception as exc:
    print(f"[WARN] partial dependence failed: {type(exc).__name__}: {exc}")

# Save interpretability summary
with open(os.path.join(DATA_DIR, "interpretability.txt"), "w") as f:
    f.write("RandomForest feature importance (resistance + temp, sums to 1):\n")
    for feat, imp in zip(pretty_cols, rf_importances):
        f.write(f"  {feat:<14}: {imp:.4f}\n")
    f.write("\nGradientBoosting feature importance:\n")
    for feat, imp in zip(pretty_cols, gb_importances):
        f.write(f"  {feat:<14}: {imp:.4f}\n")
    f.write("\nRidge coefficients on z-scored features (magnitude comparable):\n")
    for feat, c in zip(pretty_cols, ridge_coefs_scaled):
        f.write(f"  {feat:<14}: {c:+.4e}\n")
    f.write(f"  intercept    : {ridge_scaled.intercept_:+.4e}\n")
    f.write(f"\nBest model under resistance only:    {best_models['resistance_only']}\n")
    f.write(f"Best model under resistance + temp:  {best_models['resistance_temp']}\n")

print(f"\n===== INTERPRETABILITY =====")
print(f"  RF importance:   Resistance={rf_importances[0]:.3f}, Temperature={rf_importances[1]:.3f}")
print(f"  GB importance:   Resistance={gb_importances[0]:.3f}, Temperature={gb_importances[1]:.3f}")
print(f"  Ridge (z-scaled): Resistance={ridge_coefs_scaled[0]:+.4f}, Temperature={ridge_coefs_scaled[1]:+.4f}")
print(f"  Best model (R only): {best_models['resistance_only']}")
print(f"  Best model (R+T):    {best_models['resistance_temp']}")
print(f"\n[INFO] All artifacts saved under {REPORT_DIR}/")
