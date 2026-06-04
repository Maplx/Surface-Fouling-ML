"""
ACT 2 — Why does Resistance fail in the oxidation window?

If Temperature carries all the predictive power (Act 1), the natural question
is: how well does Resistance track Temperature? We turn R into the feature and
Temperature into the target, random 50/50 split (seed=0).

Story this figure tells:
  * Globally R predicts T fairly well (R and T are strongly anti-correlated).
  * BUT inside the 6000-12000 s oxidation window the prediction visibly breaks
    down: R has gone nearly flat while T keeps climbing, so a given R maps to
    many different T values. R loses its grip on T exactly where it loses its
    grip on soot mass -- the two failures are the same failure.

Outputs (next to this script):
  figures/figure_T_from_R.png        measured vs predicted Temperature over time
  figures/diagnostic_R_flatlines.png R, T and Soot vs time -- R flatlines in the
                                     window while T and Soot keep evolving
  figures/diagnostic_R_vs_soot.png   R-vs-Soot (L-shape / hysteresis: one R maps
                                     to many Soot values) + twin-axis time series
  data/leaderboard.csv               full 10-model leaderboard (best = top row)
  data/predictions.csv               true vs predicted Temperature at each test point

The two diagnostic figures are plotted directly from datasets/native_379.csv
(raw signals, no model) -- that CSV is their data.

Run:  python soot_mass_sensor_signals/02_predict_T_from_R/run.py
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


d = pd.read_csv(DATA).sort_values("Time-DC").reset_index(drop=True)
t = d["Time-DC"].to_numpy()
X = d[["Resistance"]].to_numpy()     # feature: Resistance
y = d["Temp-DC"].to_numpy()          # target : Temperature

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
lb.to_csv(os.path.join(DAT, "leaderboard.csv"), index=False)

pred_df = pd.DataFrame({"Time-DC": tte, "Temp-DC": yte, "Temp-DC (pred)": best["pred"]})
pred_df.sort_values("Time-DC").to_csv(os.path.join(DAT, "predictions.csv"), index=False)

m = (tte >= ZONE[0]) & (tte <= ZONE[1])
mae_in = mean_absolute_error(yte[m], best["pred"][m])
mae_out = mean_absolute_error(yte[~m], best["pred"][~m])

o = np.argsort(tte)
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(tte[o], yte[o], color="tab:red", lw=2.0, label="Measured Temp")
ax.plot(tte[o], best["pred"][o], color="tab:orange", lw=1.6, ls="--", label="Predicted Temp (from R)")
ax.axvspan(*ZONE, color="red", alpha=0.08, label="oxidation window")
ax.set_xlabel("Time-DC (s)")
ax.set_ylabel("Temp-DC (deg C)")
ax.set_title(f"Act 2 — Predict Temperature from Resistance (random 50/50, native 379)\n"
             f"{best['name']}: R$^2$={best['r2']:+.4f}, "
             f"MAE in-window={mae_in:.2f}, out={mae_out:.2f} deg C")
ax.grid(alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
fig.savefig(os.path.join(FIG, "figure_T_from_R.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"best={best['name']}  R2={best['r2']:+.4f}  "
      f"MAE_in={mae_in:.2f}  MAE_out={mae_out:.2f} deg C  ratio={mae_in/mae_out:.1f}x")


# ---------------------------------------------------------------------------
# Diagnostics straight from the raw signals (no model) — the "why" behind Act 2.
# ---------------------------------------------------------------------------
R = d["Resistance"].to_numpy()
T = d["Temp-DC"].to_numpy()
S = d["Soot left-mg"].to_numpy()
inside = (t >= ZONE[0]) & (t <= ZONE[1])


def band_stats(v):
    vi, vo = v[inside], v[~inside]
    return (f"in-window: std={vi.std():.1f}, range={vi.max()-vi.min():.1f}\n"
            f"elsewhere: std={vo.std():.1f}, range={vo.max()-vo.min():.1f}")


# Diagnostic 1: R, T, Soot vs time — R flatlines while T and Soot keep moving.
fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for a, v, lab, col, unit in [
        (ax[0], R, "Resistance", "tab:blue", "ohm"),
        (ax[1], T, "Temperature", "tab:red", "deg C"),
        (ax[2], S, "Soot left", "tab:green", "mg")]:
    a.plot(t, v, color=col)
    a.axvspan(*ZONE, color="red", alpha=0.10)
    a.set_ylabel(f"{lab} ({unit})")
    a.set_title(f"{lab} vs time", fontsize=10)
    a.grid(alpha=0.3)
    a.text(0.985, 0.05, band_stats(v), transform=a.transAxes, ha="right", va="bottom",
           fontsize=8, family="monospace",
           bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
ax[2].set_xlabel("Time-DC (s)")
fig.suptitle("Inside the oxidation window R has flatlined, but Temperature and Soot "
             "are still evolving\n-> a single R value has no information left to "
             "tell those points apart", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "diagnostic_R_flatlines.png"), dpi=150, bbox_inches="tight")
plt.close()

# Diagnostic 2: R-vs-Soot (colored by time) + twin-axis time series.
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sc = axes[0].scatter(R, S, c=t, cmap="viridis", s=16)
axes[0].set_xlabel("Resistance (ohm)")
axes[0].set_ylabel("Soot left (mg)")
axes[0].set_title("Resistance vs Soot (color = time)\nL-shape: one low R maps to many Soot values")
axes[0].grid(alpha=0.3)
plt.colorbar(sc, ax=axes[0], label="Time-DC (s)")
a = axes[1]
a.plot(t, R, color="tab:blue", label="Resistance")
a.set_xlabel("Time-DC (s)")
a.set_ylabel("Resistance (ohm)", color="tab:blue")
a.tick_params(axis="y", labelcolor="tab:blue")
a.axvspan(*ZONE, color="red", alpha=0.08)
ab = a.twinx()
ab.plot(t, S, color="tab:red", label="Soot left")
ab.set_ylabel("Soot left (mg)", color="tab:red")
ab.tick_params(axis="y", labelcolor="tab:red")
a.set_title("Time series: Resistance vs Soot")
a.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "diagnostic_R_vs_soot.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"[DONE] artifacts in {HERE}")
