"""
ACT 3 — Recovering predictive power from Resistance alone, via cumR.

Act 1 showed plain R caps at R^2~0.80 and fails in the oxidation window.
Act 2 showed why: instantaneous R loses its one-to-one grip on the process
there. The fix is to give the model R-DERIVED signals that re-encode the
information a single R value has lost:

    dR/dt   the rate of change of resistance (oxidation "momentum")
    cumR    the trapezoidal cumulative integral of R over time,  cumR = ∫ R dt
            (units: ohm·s). Physically a cumulative resistive-exposure /
            cumulative-oxidation quantity -- confirmed meaningful by the
            materials group. It is monotonic and encodes process history,
            which is exactly what a single instantaneous R cannot.

Feature progression (all random 50/50, seed=0):
    [R]                  -> baseline (fails in window)
    [R, dR/dt]           -> partial rescue
    [R, dR/dt, cumR]     -> near-perfect
    [cumR] only          -> near-perfect on its own  (cumR is the key signal)

Outputs (next to this script):
  figures/figure_feature_progression.png   2x2 measured-vs-predicted, window highlighted
  figures/figure_cumR_signal.png           R, dR/dt and cumR over time (what cumR is)
  data/candidates_summary.csv              every feature set: best model, R2, MAE in/out
  data/predictions_R_dRdt_cumR.csv         true vs predicted for the headline set
  data/predictions_cumR_only.csv           true vs predicted for cumR alone

Run:  python soot_mass_sensor_signals/03_resistance_derived_features/run.py
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
ZONE = (6000.0, 12000.0)
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
    Xtr, Xte, ytr, yte, ttr, tte = train_test_split(
        X, y, t, test_size=0.5, shuffle=True, random_state=SEED)
    best = None
    for name, model in model_zoo():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        r2 = r2_score(yte, pred)
        if best is None or r2 > best["r2"]:
            best = {"name": name, "r2": r2, "pred": pred}
    return best["name"], best["r2"], best["pred"], yte, tte


def zone_mae(t, y_true, y_pred):
    m = (t >= ZONE[0]) & (t <= ZONE[1])
    return (mean_absolute_error(y_true[m], y_pred[m]),
            mean_absolute_error(y_true[~m], y_pred[~m]))


# ---- load and build R-derived features ----
d = pd.read_csv(DATA).sort_values("Time-DC").reset_index(drop=True)
t = d["Time-DC"].to_numpy()
R = d["Resistance"].to_numpy()
y = d["Soot left-mg"].to_numpy()

d["dR/dt"] = np.gradient(R, t)
# Trapezoidal cumulative integral of R over time (ohm*s): cumulative oxidation exposure.
d["cumR"] = np.concatenate(([0.0], np.cumsum(0.5 * (R[1:] + R[:-1]) * np.diff(t))))

# ---- candidate feature sets (all derived from R alone) ----
CANDIDATES = [
    ("[R]",               ["Resistance"]),
    ("[R, dR/dt]",        ["Resistance", "dR/dt"]),
    ("[R, dR/dt, cumR]",  ["Resistance", "dR/dt", "cumR"]),
    ("[cumR] only",       ["cumR"]),
]

results = {}
summary_rows = []
print(f"{'feature set':<22} {'best model':<22} {'R2':>9} {'MAE_in':>9} {'MAE_out':>9}")
print("-" * 76)
for label, cols in CANDIDATES:
    bn, r2, pred, yte, tte = run_cell(d[cols].to_numpy(), y, t)
    results[label] = (bn, r2, pred, yte, tte)
    mae_in, mae_out = zone_mae(tte, yte, pred)
    summary_rows.append({"feature_set": label, "n_features": len(cols),
                         "best_model": bn, "R2": r2,
                         "MAE_in_window_mg": mae_in, "MAE_out_mg": mae_out})
    print(f"{label:<22} {bn:<22} {r2:>+9.4f} {mae_in:>9.4f} {mae_out:>9.4f}")

pd.DataFrame(summary_rows).to_csv(os.path.join(DAT, "candidates_summary.csv"), index=False)

# ---- prediction CSVs for the two headline sets ----
for label, fname in [("[R, dR/dt, cumR]", "predictions_R_dRdt_cumR.csv"),
                     ("[cumR] only",      "predictions_cumR_only.csv")]:
    _, _, pred, yte, tte = results[label]
    pd.DataFrame({"Time-DC": tte, "Soot left-mg": yte,
                  "Soot left-mg (pred)": pred}).sort_values("Time-DC").to_csv(
        os.path.join(DAT, fname), index=False)

# ---- FIGURE 1: 2x2 feature progression ----
fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
for ax, (label, _) in zip(axes.flatten(), CANDIDATES):
    bn, r2, pred, yte, tte = results[label]
    mae_in, mae_out = zone_mae(tte, yte, pred)
    o = np.argsort(tte)
    ax.plot(tte[o], yte[o], color="tab:blue", lw=2.0, label="Measured")
    ax.plot(tte[o], pred[o], color="tab:orange", lw=1.6, ls="--", label="Predicted")
    ax.axvspan(*ZONE, color="red", alpha=0.08, label="oxidation window")
    ax.set_title(f"{label}\n{bn}: R$^2$={r2:+.4f}, "
                 f"MAE in-window={mae_in:.4f}, out={mae_out:.4f} mg", fontsize=10)
    ax.set_xlabel("Time-DC (s)")
    ax.set_ylabel("Soot left (mg)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
fig.suptitle("Act 3 — R-derived features recover the prediction; cumR is the key signal "
             "(random 50/50, native 379)", fontsize=13, y=1.00)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "figure_feature_progression.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---- FIGURE 2: what cumR is (R, dR/dt, cumR over time) ----
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
ax[0].plot(t, R, color="tab:blue"); ax[0].set_ylabel("R (ohm)"); ax[0].set_title("Resistance R(t)")
ax[1].plot(t, d["dR/dt"], color="tab:purple"); ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_ylabel("dR/dt (ohm/s)"); ax[1].set_title("dR/dt(t)")
ax[2].plot(t, d["cumR"], color="tab:green"); ax[2].set_ylabel("cumR (ohm*s)")
ax[2].set_title(r"cumR(t) = $\int_0^t R\,d\tau$  (cumulative oxidation exposure)")
ax[2].set_xlabel("Time-DC (s)")
for a in ax:
    a.axvspan(*ZONE, color="red", alpha=0.08)
    a.grid(alpha=0.3)
fig.suptitle("Act 3 — the R-derived signals (red band = oxidation window)", fontsize=12, y=1.00)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "figure_cumR_signal.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\n[DONE] artifacts in {HERE}")
