"""
ACT 4 — R, T, cumR head-to-head: which single signal best predicts soot mass?

Three single-signal candidates compared side by side. Each one is the sole
input feature for the same 10-model zoo on the same random 50/50 split
(seed=0), so the bars are directly comparable.

    [R]      Resistance alone -- the raw sensor reading.
    [T]      Temperature alone -- the controlled co-variate; requires an
             independent thermocouple.
    [cumR]   Cumulative resistance = trapezoidal integral of R over time
             (units: ohm*s). Physically a cumulative-oxidation-exposure
             quantity. No thermocouple required, derived from the sensor's
             own trace alone.

This isolates the question: among signals available from a single device,
which one is most predictive?

Outputs (next to this script):
  figures/figure_R_T_cumR.png    one bar plot, 10 models x 3 single signals
  data/leaderboard_R.csv         10-model leaderboard for [R]
  data/leaderboard_T.csv         10-model leaderboard for [T]
  data/leaderboard_cumR.csv      10-model leaderboard for [cumR]
  data/predictions_R.csv         true vs predicted at each test point, [R]
  data/predictions_T.csv         true vs predicted at each test point, [T]
  data/predictions_cumR.csv      true vs predicted at each test point, [cumR]
  data/summary.csv               best model + R^2 + MAE per feature set
  data/r2_wide.csv               10 models x 3 feature sets, R^2 side-by-side

Run:  python soot_mass_sensor_signals/04_R_T_cumR_comparison/run.py
"""

import os
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


SEED = 0
ZONE = (6000.0, 12000.0)             # active oxidation window (the hard region)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_IN = os.path.join(HERE, "..", "datasets", "native_379.csv")
FIG = os.path.join(HERE, "figures")
DAT = os.path.join(HERE, "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DAT, exist_ok=True)


def model_zoo():
    return [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge(alpha=1.0, random_state=SEED)),
        ("Lasso", Lasso(alpha=1e-3, random_state=SEED, max_iter=10000)),
        ("ElasticNet", ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=SEED, max_iter=10000)),
        ("SVR(RBF)", Pipeline([("s", StandardScaler()),
                                ("m", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.001))])),
        ("KNN(k=5)", Pipeline([("s", StandardScaler()),
                                ("m", KNeighborsRegressor(n_neighbors=5))])),
        ("DecisionTree", DecisionTreeRegressor(random_state=SEED, max_depth=5)),
        ("RandomForest", RandomForestRegressor(n_estimators=500, random_state=SEED,
                                                max_depth=None, min_samples_leaf=2)),
        ("GradientBoosting", GradientBoostingRegressor(random_state=SEED)),
        ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=SEED)),
    ]


def zone_mae(t, y_true, y_pred):
    m = (t >= ZONE[0]) & (t <= ZONE[1])
    return (mean_absolute_error(y_true[m], y_pred[m]),
            mean_absolute_error(y_true[~m], y_pred[~m]))


def run_cell(X, y, t):
    """Random 50/50 split, fit zoo, return leaderboard + per-model R^2 + best."""
    Xtr, Xte, ytr, yte, ttr, tte = train_test_split(
        X, y, t, test_size=0.5, shuffle=True, random_state=SEED)
    rows, best = [], None
    all_r2 = {}
    for name, model in model_zoo():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        r2 = r2_score(yte, pred)
        all_r2[name] = r2
        rows.append({
            "model": name,
            "MAE": float(mean_absolute_error(yte, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
            "R2": float(r2),
        })
        if best is None or r2 > best["r2"]:
            best = {"name": name, "r2": r2, "pred": pred}
    lb = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return lb, best["name"], best["pred"], yte, tte, all_r2


# -----------------------------
# 1) Load + derive cumR
# -----------------------------

d = pd.read_csv(DATA_IN).sort_values("Time-DC").reset_index(drop=True)
t = d["Time-DC"].to_numpy()
R = d["Resistance"].to_numpy()
y = d["Soot left-mg"].to_numpy()

# Trapezoidal cumulative integral of R over time, in ohm*s.
d["cumR"] = np.concatenate(([0.0],
                            np.cumsum(0.5 * (R[1:] + R[:-1]) * np.diff(t))))

print(f"[INFO] {len(d)} rows, time [{t.min():.0f}, {t.max():.0f}] s; "
      f"cumR range [{d['cumR'].min():.0f}, {d['cumR'].max():.0f}] ohm*s")

# -----------------------------
# 2) Run the zoo on each single-feature set
# -----------------------------

FEATURE_SETS = [
    ("R",    "Resistance",                  "Resistance"),
    ("T",    "Temperature",                 "Temp-DC"),
    ("cumR", r"cumR ($=\int R\,dt$)",       "cumR"),
]

r2_per_set = {}
summary_rows = []
print(f"\n{'feature':<8} {'best model':<22} {'best R2':>9} "
      f"{'MAE_in':>9} {'MAE_out':>9}")
print("-" * 64)

for key, pretty, col in FEATURE_SETS:
    X = d[[col]].to_numpy()
    lb, best_name, best_pred, yte, tte, all_r2 = run_cell(X, y, t)
    r2_per_set[key] = all_r2

    mae_in, mae_out = zone_mae(tte, yte, best_pred)
    summary_rows.append({
        "feature_set": key,
        "feature_column": col,
        "best_model": best_name,
        "best_R2": lb.iloc[0]["R2"],
        "best_MAE": lb.iloc[0]["MAE"],
        "MAE_in_window_mg": mae_in,
        "MAE_out_window_mg": mae_out,
    })

    lb.to_csv(os.path.join(DAT, f"leaderboard_{key}.csv"), index=False)
    pd.DataFrame({
        "Time-DC": tte,
        "Soot left-mg": yte,
        "Soot left-mg (pred)": best_pred,
    }).sort_values("Time-DC").to_csv(
        os.path.join(DAT, f"predictions_{key}.csv"), index=False)

    print(f"{key:<8} {best_name:<22} {lb.iloc[0]['R2']:>+9.4f} "
          f"{mae_in:>9.4f} {mae_out:>9.4f}")

pd.DataFrame(summary_rows).to_csv(os.path.join(DAT, "summary.csv"), index=False)

# Wide R^2 table for at-a-glance comparison
all_models = [name for name, _ in model_zoo()]
wide = pd.DataFrame({
    "model": all_models,
    "R":    [r2_per_set["R"][m]    for m in all_models],
    "T":    [r2_per_set["T"][m]    for m in all_models],
    "cumR": [r2_per_set["cumR"][m] for m in all_models],
}).set_index("model")
wide = wide.sort_values("cumR", ascending=False)
wide.to_csv(os.path.join(DAT, "r2_wide.csv"))

print("\n===== R^2 BY SINGLE-SIGNAL FEATURE =====")
print(wide.to_string(float_format=lambda x: f"{x:+.4f}"))

# -----------------------------
# 3) Bar plot
# -----------------------------

models_sorted = wide.index.tolist()
xpos = np.arange(len(models_sorted))
w = 0.27
clip = -0.1

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(xpos - w, [max(r2_per_set["R"][m],    clip) for m in models_sorted], w,
       label="Resistance (R)",                       color="0.55")
ax.bar(xpos,     [max(r2_per_set["T"][m],    clip) for m in models_sorted], w,
       label="Temperature (T)",                      color="tab:blue")
ax.bar(xpos + w, [max(r2_per_set["cumR"][m], clip) for m in models_sorted], w,
       label=r"cumR $=\int R\,dt$",                  color="tab:orange")

ax.set_xticks(xpos)
ax.set_xticklabels(models_sorted, rotation=30, ha="right")
ax.set_ylabel(r"Test $R^{2}$ (values $< -0.1$ clipped)")
ax.set_title("Which single signal best predicts remaining soot mass?\n"
             r"Test $R^{2}$ for R, T, and cumR each used alone "
             r"(random 50/50 split, $n_{\mathrm{test}}=190$)")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylim(-0.12, 1.08)
ax.grid(True, alpha=0.3, axis="y")
ax.legend(loc="lower left", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "figure_R_T_cumR.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\n[DONE] artifacts in {HERE}")
