"""
Predict coating thickness from AC impedance Z  (17500 Hz, 400 C).

Sheet "17500Hz_400C_AC vs Time" of `three different tests results.xlsx`.

The experiment steps the cumulative thickness up in ~6.7 um increments
(0 -> 136 um, 21 levels). Z vs time shows one plateau per thickness: Z settles
to a low, conductive value while that layer is measured, then jumps to a ~95 kOhm
reference state between measurements. The settled low Z is what tracks thickness
(more material deposited -> more conductive -> lower Z); the ~95 kOhm high state
is constant and carries no thickness info.

Pipeline:
  1. Segment the low plateaus (Z < 40 kOhm, >= 20 samples each).
  2. Reduce each plateau to one representative Z = median of its 10 lowest samples
     (the settled conductive value; robust to noise).
  3. If one extra plateau is found (a short noise burst), drop the shortest, then
     pair the 21 plateaus with the 21 thickness levels in time order.
  4. Calibration (physics): the film adds conductance in proportion to thickness
     over a baseline, 1/Z = k*thickness + G0, i.e. thickness = a/Z + b. An
     empirical log fit is kept only for comparison.
  5. ML: a 10-model zoo predicts thickness from the raw representative Z, scored by
     leave-one-out CV (n=21 is small, so LOOCV rather than a single split).

Outputs (next to this script):
  data/calibration_points.csv   21 rows: thickness, repZ, plateau start/end, n
  data/ml_leaderboard.csv        LOOCV R2 / MAE per model
  figures/Z_vs_time_plateaus.png Z(t) with each kept plateau shaded + thickness
  figures/conductance_linear.png 1/Z vs thickness (the physics: linear) + R2
  figures/calibration.png        thickness vs Z, physics fit a/Z+b (log fit for comparison)
  figures/fit_pred_vs_measured.png  measured vs LOOCV-predicted thickness over time (best model)

Run:  python Surface-Fouling-ML/thickness_from_Z/run.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

SEED = 0
XL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                  "three different tests results.xlsx")
SHEET = "17500Hz_400C_AC vs Time"
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
DAT = os.path.join(HERE, "data")
for p in (FIG, DAT):
    os.makedirs(p, exist_ok=True)


def model_zoo():
    return [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge(alpha=1.0, random_state=SEED)),
        ("Lasso", Lasso(alpha=1e-3, random_state=SEED, max_iter=10000)),
        ("ElasticNet", ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=SEED, max_iter=10000)),
        ("SVR(RBF)", Pipeline([("s", StandardScaler()),
                                ("m", SVR(kernel="rbf", C=1000.0, gamma="scale", epsilon=0.01))])),
        ("KNN(k=3)", Pipeline([("s", StandardScaler()),
                                ("m", KNeighborsRegressor(n_neighbors=3))])),
        ("DecisionTree", DecisionTreeRegressor(random_state=SEED, max_depth=5)),
        ("RandomForest", RandomForestRegressor(n_estimators=500, random_state=SEED,
                                                max_depth=None, min_samples_leaf=1)),
        ("GradientBoosting", GradientBoostingRegressor(random_state=SEED)),
        ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=SEED)),
    ]


# -----------------------------------------------------------------------------
# 1) Load Z(t) and the thickness levels
# -----------------------------------------------------------------------------
d = pd.read_excel(XL, sheet_name=SHEET, header=None, skiprows=2)[[0, 3]]
d.columns = ["t", "Z"]
d = d.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

th = pd.to_numeric(pd.read_excel(XL, sheet_name=SHEET, header=None)[8], errors="coerce").dropna()
th = th[th >= 0].to_numpy()

# -----------------------------------------------------------------------------
# 2) Segment low plateaus, one representative settled-Z each
# -----------------------------------------------------------------------------
hi = (d.Z > 40000).to_numpy().astype(int)
runs, i, n = [], 0, len(hi)
while i < n:
    if hi[i] == 0:
        j = i
        while j < n and hi[j] == 0:
            j += 1
        runs.append((i, j))
        i = j
    else:
        i += 1
runs = [(a, b) for a, b in runs if b - a >= 20]           # substantial plateaus only

rows = []
for a, b in runs:
    z = np.sort(d.Z.iloc[a:b].to_numpy())
    rows.append({"t0": d.t.iloc[a], "t1": d.t.iloc[b - 1], "n": b - a,
                 "repZ": float(np.median(z[:10]))})        # settled = median of 10 lowest
rep = pd.DataFrame(rows)

# 3) reconcile count with thickness levels, pair in time order
if len(rep) == len(th) + 1:
    drop = rep["n"].idxmin()
    print(f"[INFO] {len(rep)} plateaus vs {len(th)} levels -> dropping shortest "
          f"(n={rep.loc[drop,'n']}, t={rep.loc[drop,'t0']:.0f}s, noise burst)")
    rep = rep.drop(drop).reset_index(drop=True)
if len(rep) != len(th):
    raise SystemExit(f"[ERR] {len(rep)} plateaus != {len(th)} thickness levels; check segmentation")
rep["thickness"] = th
rep.to_csv(os.path.join(DAT, "calibration_points.csv"), index=False)

Z = rep["repZ"].to_numpy()
y = rep["thickness"].to_numpy()
print(f"[INFO] {len(rep)} calibration points; repZ {Z.min():.0f}->{Z.max():.0f} ohm, "
      f"thickness {y.min():.0f}->{y.max():.0f} um")

# -----------------------------------------------------------------------------
# 4) Calibration
#    Physical model (parallel conductance): the film adds conductance in
#    proportion to its thickness, on top of a baseline substrate conductance:
#        1/Z = k * thickness + G0
#    Solving for thickness gives the calibration:  thickness = a/Z + b
#    (a = 1/k, b = -G0/k).  The empirical log fit is kept only for comparison.
# -----------------------------------------------------------------------------
span = y.max() - y.min()
G = 1.0 / Z                                     # conductance (S)
k, G0 = np.polyfit(y, G, 1)                      # 1/Z = k*thickness + G0
cond_r2 = r2_score(G, k * y + G0)
a_inv, b_inv = 1.0 / k, -G0 / k                  # thickness = a/Z + b
phys_pred = a_inv / Z + b_inv
phys_r2 = r2_score(y, phys_pred)
phys_mae = mean_absolute_error(y, phys_pred)
print(f"[CAL-phys] 1/Z = {k:.3e}*t + {G0:.3e}  (conductance R2={cond_r2:.4f})")
print(f"           -> thickness = {a_inv:.0f}/Z + {b_inv:.1f}   "
      f"R2={phys_r2:.4f}  MAE={phys_mae:.2f} um ({phys_mae/span*100:.1f}%)")

a_ln, b_ln = np.polyfit(np.log(Z), y, 1)         # empirical, comparison only
ln_pred = a_ln * np.log(Z) + b_ln
ln_r2 = r2_score(y, ln_pred)
print(f"[CAL-log ] thickness = {a_ln:.1f}*ln(Z) + {b_ln:.1f}   "
      f"R2={ln_r2:.4f}  (empirical, for comparison)")

# -----------------------------------------------------------------------------
# 5) ML zoo, leave-one-out CV, predict thickness from raw Z
# -----------------------------------------------------------------------------
X = Z.reshape(-1, 1)
loo = LeaveOneOut()
lb, best = [], None
for name, model in model_zoo():
    pred = np.zeros(len(y))
    for tr, te in loo.split(X):
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    lb.append({"model": name, "LOOCV_R2": r2, "MAE_um": mae, "MAE_pct": mae / span * 100})
    if best is None or r2 > best["r2"]:
        best = {"name": name, "r2": r2, "mae": mae, "pred": pred.copy()}
lb = pd.DataFrame(lb).sort_values("LOOCV_R2", ascending=False).reset_index(drop=True)
lb.to_csv(os.path.join(DAT, "ml_leaderboard.csv"), index=False)
print("\n===== ML leaderboard (LOOCV) =====")
print(lb.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# also a random 50/50 split (same protocol as the earlier analyses)
t0 = rep["t0"].to_numpy()
Xtr, Xte, ytr, yte, ttr, tte = train_test_split(
    X, y, t0, test_size=0.5, shuffle=True, random_state=SEED)
lb5, best5 = [], None
for name, model in model_zoo():
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    r2 = r2_score(yte, pred)
    mae = mean_absolute_error(yte, pred)
    lb5.append({"model": name, "R2": r2, "MAE_um": mae, "MAE_pct": mae / span * 100})
    if best5 is None or r2 > best5["r2"]:
        best5 = {"name": name, "r2": r2, "mae": mae, "pred": pred}
lb5 = pd.DataFrame(lb5).sort_values("R2", ascending=False).reset_index(drop=True)
lb5.to_csv(os.path.join(DAT, "ml_leaderboard_5050.csv"), index=False)
print(f"\n===== ML leaderboard (random 50/50, n_test={len(yte)}) =====")
print(lb5.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
# (a) Z vs time with plateaus + thickness
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(d.t, d.Z, lw=0.6, color="0.4")
ax.set_ylim(0, np.nanpercentile(d.Z, 98))
for _, r in rep.iterrows():
    ax.axvspan(r.t0, r.t1, color="tab:orange", alpha=0.18)
    ax.text((r.t0 + r.t1) / 2, ax.get_ylim()[1] * 0.93, f"{r.thickness:.0f}",
            ha="center", va="top", fontsize=7, rotation=90, color="tab:red")
ax.set_xlabel("time / s")
ax.set_ylabel("Z / ohm")
ax.set_title("AC impedance Z vs time - one low plateau per thickness "
             "(orange = kept plateau, red = thickness in um)")
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "Z_vs_time_plateaus.png"), dpi=150, bbox_inches="tight")
plt.close()

# (b1) the physics: conductance 1/Z is linear in thickness
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.scatter(y, G * 1e3, s=45, color="tab:blue", zorder=3, label="plateaus (measured)")
tt = np.linspace(y.min(), y.max(), 200)
ax.plot(tt, (k * tt + G0) * 1e3, color="tab:orange", lw=2,
        label=f"fit: 1/Z = {k:.2e}·t + {G0:.2e}\n$R^2$={cond_r2:.4f}")
ax.set_xlabel("cumulative thickness / um")
ax.set_ylabel("conductance  1/Z  / mS")
ax.set_title("Physics: conductance is linear in thickness (17500 Hz, 400 C)\n"
             "film adds conductance in proportion to thickness, over a baseline G0")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG, "conductance_linear.png"), dpi=150, bbox_inches="tight")
plt.close()

# (b2) calibration curve: physics thickness = a/Z + b (log fit shown for comparison)
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.scatter(Z, y, s=45, color="tab:blue", zorder=3, label="plateaus (measured)")
zz = np.linspace(Z.min(), Z.max(), 300)
ax.plot(zz, a_inv / zz + b_inv, color="tab:orange", lw=2,
        label=f"physics: thickness = {a_inv:.0f}/Z + {b_inv:.1f}\n"
              f"$R^2$={phys_r2:.4f}, MAE={phys_mae:.1f} um")
ax.plot(zz, a_ln * np.log(zz) + b_ln, "--", color="0.55", lw=1.6,
        label=f"log fit (comparison), $R^2$={ln_r2:.4f}")
ax.set_xlabel("representative (settled) Z / ohm")
ax.set_ylabel("cumulative thickness / um")
ax.set_title("Z -> thickness calibration (17500 Hz, 400 C)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG, "calibration.png"), dpi=150, bbox_inches="tight")
plt.close()

# (c) best ML: measured vs predicted over plateau time (same style as earlier plots)
order = np.argsort(rep["t0"].to_numpy())
tt_time = rep["t0"].to_numpy()[order]
fig, ax = plt.subplots(figsize=(11, 5))
ax.scatter(tt_time, y[order], s=45, color="0.35", label="Measured", zorder=3)
ax.plot(tt_time, best["pred"][order], color="tab:green", lw=2, marker="o", ms=4,
        label="Predicted (LOOCV)", zorder=2)
ax.set_xlabel("plateau time / s")
ax.set_ylabel("thickness / um")
ax.set_title(f"Thickness from Z - {best['name']} (LOOCV)   "
             f"$R^2$={best['r2']:.4f}   MAE={best['mae']:.1f} um ({best['mae']/span*100:.1f}%)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG, "fit_pred_vs_measured.png"), dpi=150, bbox_inches="tight")
plt.close()

# (d) same thing for the random 50/50 split -> test set only
o5 = np.argsort(tte)
fig, ax = plt.subplots(figsize=(11, 5))
ax.scatter(tte[o5], yte[o5], s=55, color="0.35", label="Measured (test)", zorder=3)
ax.plot(tte[o5], best5["pred"][o5], color="tab:green", lw=2, marker="o", ms=5,
        label="Predicted (test)", zorder=2)
ax.set_xlabel("plateau time / s")
ax.set_ylabel("thickness / um")
ax.set_title(f"Thickness from Z - {best5['name']} (random 50/50, n_test={len(yte)})   "
             f"$R^2$={best5['r2']:.4f}   MAE={best5['mae']:.1f} um ({best5['mae']/span*100:.1f}%)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG, "fit_pred_5050.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\n[DONE] artifacts under {HERE}")
