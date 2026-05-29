"""
ACT 1 — The Temperature reveal.

Predict remaining soot mass from sensor signals under a RANDOM 50/50 split
(seed=0), comparing three feature sets:

    [R]      Resistance only
    [T]      Temperature only
    [R, T]   Resistance + Temperature

Story this figure tells:
  * R alone fails badly in the 6000-12000 s window -- the most important
    region, where soot oxidation is most intense.
  * R + T predicts almost perfectly.
  * T alone ALSO predicts almost perfectly -> the predictive power of R+T
    comes essentially entirely from Temperature, not Resistance.

Outputs (next to this script):
  figures/figure_R_T_RT.png   one figure, three panels (R / T / R+T)
  data/leaderboard_R.csv      full 10-model leaderboard, R only   (best = top row)
  data/leaderboard_T.csv      full 10-model leaderboard, T only
  data/leaderboard_RT.csv     full 10-model leaderboard, R + T
  data/predictions_R.csv      true vs predicted soot mass at each test point (R)
  data/predictions_T.csv      true vs predicted soot mass at each test point (T)
  data/predictions_RT.csv     true vs predicted soot mass at each test point (R+T)

Run:  python soot_mass_sensor_signals/01_temperature_reveal/run.py
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
ZONE = (6000.0, 12000.0)            # active-oxidation window (the hard region)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "datasets", "native_379.csv")
FIG = os.path.join(HERE, "figures")   # all .png go here
DAT = os.path.join(HERE, "data")      # all .csv go here
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


def run_cell(X, y, t):
    """Random 50/50 split, fit the whole zoo, return leaderboard + best test arrays."""
    Xtr, Xte, ytr, yte, ttr, tte = train_test_split(
        X, y, t, test_size=0.5, shuffle=True, random_state=SEED)
    rows, best = [], None
    for name, model in model_zoo():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        r2 = r2_score(yte, pred)
        rows.append({"model": name,
                     "MAE": float(mean_absolute_error(yte, pred)),
                     "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
                     "R2": float(r2)})
        if best is None or r2 > best["r2"]:
            best = {"name": name, "r2": r2, "pred": pred}
    lb = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return lb, best["name"], best["pred"], yte, tte


def zone_mae(t, y_true, y_pred):
    m = (t >= ZONE[0]) & (t <= ZONE[1])
    return (mean_absolute_error(y_true[m], y_pred[m]),
            mean_absolute_error(y_true[~m], y_pred[~m]))


d = pd.read_csv(DATA).sort_values("Time-DC").reset_index(drop=True)
y = d["Soot left-mg"].to_numpy()
t = d["Time-DC"].to_numpy()

FEATURE_SETS = [
    ("R",  "Resistance only",          ["Resistance"]),
    ("T",  "Temperature only",         ["Temp-DC"]),
    ("RT", "Resistance + Temperature", ["Resistance", "Temp-DC"]),
]

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), sharey=True)
print(f"{'features':<26} {'best model':<22} {'R2':>9} {'MAE_in':>9} {'MAE_out':>9}")
print("-" * 80)

for ax, (key, pretty, cols) in zip(axes, FEATURE_SETS):
    X = d[cols].to_numpy()
    lb, best_name, best_pred, yte, tte = run_cell(X, y, t)
    r2 = lb.iloc[0]["R2"]
    mae_in, mae_out = zone_mae(tte, yte, best_pred)

    # leaderboard CSV (best model = top row)
    lb.to_csv(os.path.join(DAT, f"leaderboard_{key}.csv"), index=False)

    # predictions CSV (the data behind the plot: true vs predicted)
    pred_df = pd.DataFrame({"Time-DC": tte,
                            "Soot left-mg": yte,
                            "Soot left-mg (pred)": best_pred})
    pred_df.sort_values("Time-DC").to_csv(
        os.path.join(DAT, f"predictions_{key}.csv"), index=False)

    # plot
    o = np.argsort(tte)
    ax.plot(tte[o], yte[o], color="tab:blue", lw=2.0, label="Measured")
    ax.plot(tte[o], best_pred[o], color="tab:orange", lw=1.6, ls="--", label="Predicted")
    ax.axvspan(*ZONE, color="red", alpha=0.08, label="oxidation window")
    ax.set_title(f"{pretty}\n{best_name}: R$^2$={r2:+.4f}\n"
                 f"MAE in-window={mae_in:.4f}, out={mae_out:.4f} mg", fontsize=10)
    ax.set_xlabel("Time-DC (s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

axes[0].set_ylabel("Soot left (mg)")
fig.suptitle("Act 1 — Resistance alone fails in the oxidation window; "
             "the perfect fit comes from Temperature (random 50/50, native 379)",
             fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "figure_R_T_RT.png"), dpi=150, bbox_inches="tight")
plt.close()

# console summary (re-read top rows for a tidy table)
for key, pretty, _ in FEATURE_SETS:
    lb = pd.read_csv(os.path.join(DAT, f"leaderboard_{key}.csv"))
    pr = pd.read_csv(os.path.join(DAT, f"predictions_{key}.csv"))
    mae_in, mae_out = zone_mae(pr["Time-DC"].to_numpy(),
                               pr["Soot left-mg"].to_numpy(),
                               pr["Soot left-mg (pred)"].to_numpy())
    top = lb.iloc[0]
    print(f"{pretty:<26} {top['model']:<22} {top['R2']:>+9.4f} {mae_in:>9.4f} {mae_out:>9.4f}")

print(f"\n[DONE] artifacts in {HERE}")
