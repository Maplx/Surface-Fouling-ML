"""
AC dynamic soot oxidation -- Z, T, cumZ head-to-head for predicting soot.

Supporting-information analogue of the DC R / T / cumR study
(soot_mass_sensor_signals/04_R_T_cumR_comparison). Same recipe, AC data:

    Z       AC impedance (ohm)         -- the raw AC sensor reading  (DC analogue: R)
    T       Temperature (degC)         -- controlled co-variate, needs a thermocouple
    cumZ    = integral of Z over time  -- cumulative impedance exposure (ohm*s),
                                          derived from the sensor's own trace alone
                                          (DC analogue: cumR)

Target: remaining soot, the "Soot conversion-ac" trace (100 % -> 0 % as it burns off).

Each of Z, T, cumZ is the SOLE input feature for the same 10-model zoo on the
same random 50/50 split (seed=0), so results are directly comparable -- exactly
the protocol used for DC.

Pipeline:
  1. The raw sheet stores Temp, soot and the Z sensor on three different time
     axes. We align Temp and Z onto the soot-conversion timestamps by linear
     interpolation (the sensor covers the whole oxidation), giving one tidy
     aligned table -- the AC analogue of native_379.csv.
  2. cumZ = trapezoidal cumulative integral of Z over time.
  3. Run the zoo on [Z], [T], [cumZ]; report R^2 leaderboards + fitting plots.

Red band in the plots = 6000-12000 s active oxidation window, the region that
matters most -- same window as the DC study.

Outputs (next to this script):
  datasets/ac_aligned.csv           aligned Time, Temp, Z, cumZ, Soot-conversion
  data/leaderboard_{Z,T,cumZ}.csv   10-model leaderboard per single signal
  data/predictions_{Z,T,cumZ}.csv   true vs predicted at each test point
  data/summary.csv                  best model + R^2 + MAE (in/out window) per signal
  data/r2_wide.csv                  10 models x 3 signals, R^2 side by side
  figures/figure_Z_T_cumZ_R2.png    bar plot: R^2, 10 models x 3 signals
  figures/figure_fit_Z_T_cumZ.png   fitting plots: measured vs predicted over time
  figures/figure_signals.png        overview of Temp, Z, cumZ, soot vs time

Run:  python Surface-Fouling-ML/ac_soot_oxidation_si/run.py
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
ZONE = (6000.0, 12000.0)              # active oxidation window (DC analogue range)
HERE = os.path.dirname(os.path.abspath(__file__))
# xlsx lives two levels up: surface fouling/Copy of AC soot oxidation ramping ML data.xlsx
XLSX = os.path.join(HERE, "..", "..", "Copy of AC soot oxidation ramping ML data.xlsx")
SHEET = "AC mode dynamic soot oxidation"
DSET = os.path.join(HERE, "datasets")
FIG = os.path.join(HERE, "figures")
DAT = os.path.join(HERE, "data")
for p in (DSET, FIG, DAT):
    os.makedirs(p, exist_ok=True)


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
    """Random 50/50 split, fit the zoo, return leaderboard + per-model R^2 + best."""
    Xtr, Xte, ytr, yte, ttr, tte = train_test_split(
        X, y, t, test_size=0.5, shuffle=True, random_state=SEED)
    rows, best, all_r2 = [], None, {}
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


# -----------------------------------------------------------------------------
# 1) Load the three series and align Temp + Z onto the soot timestamps
# -----------------------------------------------------------------------------
raw = pd.read_excel(XLSX, sheet_name=SHEET)

temp = (raw[["Time-temp", "Temp"]].dropna()
        .rename(columns={"Time-temp": "t", "Temp": "Temp"})
        .sort_values("t").reset_index(drop=True))
soot = (raw[["TIME-soot conversion-ac", "Soot conversion-ac"]].dropna()
        .rename(columns={"TIME-soot conversion-ac": "t", "Soot conversion-ac": "Soot"})
        .sort_values("t").reset_index(drop=True))
sens = (raw[["time-sensor", "Z/ohm-AC"]].dropna()
        .rename(columns={"time-sensor": "t", "Z/ohm-AC": "Z"})
        .sort_values("t").reset_index(drop=True))
# average any duplicate sensor timestamps before interpolation
sens = sens.groupby("t", as_index=False)["Z"].mean()

# common range where all three series exist (sensor is the limiting one)
t_lo = max(temp["t"].min(), soot["t"].min(), sens["t"].min())
t_hi = min(temp["t"].max(), soot["t"].max(), sens["t"].max())

d = soot[(soot["t"] >= t_lo) & (soot["t"] <= t_hi)].copy().reset_index(drop=True)
d["Temp"] = np.interp(d["t"], temp["t"], temp["Temp"])
d["Z"] = np.interp(d["t"], sens["t"], sens["Z"])
d = d[["t", "Temp", "Z", "Soot"]].rename(columns={"t": "Time"})

t = d["Time"].to_numpy()
Z = d["Z"].to_numpy()
y = d["Soot"].to_numpy()

# cumZ = trapezoidal cumulative integral of Z over time, in ohm*s
d["cumZ"] = np.concatenate(([0.0], np.cumsum(0.5 * (Z[1:] + Z[:-1]) * np.diff(t))))
d = d[["Time", "Temp", "Z", "cumZ", "Soot"]]
d.to_csv(os.path.join(DSET, "ac_aligned.csv"), index=False)

print(f"[INFO] aligned {len(d)} rows, time [{t.min():.0f}, {t.max():.0f}] s")
print(f"[INFO] Temp [{d.Temp.min():.0f}, {d.Temp.max():.0f}] C | "
      f"Z [{d.Z.min():.0f}, {d.Z.max():.0f}] ohm | "
      f"cumZ [{d.cumZ.min():.0f}, {d.cumZ.max():.0f}] ohm*s | "
      f"Soot [{d.Soot.min():.1f}, {d.Soot.max():.1f}] %")

# -----------------------------------------------------------------------------
# 2) Run the zoo on each single-signal feature set
# -----------------------------------------------------------------------------
FEATURE_SETS = [
    ("Z",    "Impedance (Z)",            "Z"),
    ("T",    "Temperature (T)",          "Temp"),
    ("cumZ", r"cumZ ($=\int Z\,dt$)",    "cumZ"),
]

r2_per_set, summary_rows, fits = {}, [], {}
print(f"\n{'signal':<8}{'best model':<22}{'best R2':>9}{'MAE_in':>10}{'MAE_out':>10}")
print("-" * 59)

for key, pretty, col in FEATURE_SETS:
    X = d[[col]].to_numpy()
    lb, best_name, best_pred, yte, tte, all_r2 = run_cell(X, y, t)
    r2_per_set[key] = all_r2
    fits[key] = (best_name, tte, yte, best_pred, pretty)

    mae_in, mae_out = zone_mae(tte, yte, best_pred)
    summary_rows.append({
        "signal": key,
        "feature_column": col,
        "best_model": best_name,
        "best_R2": lb.iloc[0]["R2"],
        "best_MAE": lb.iloc[0]["MAE"],
        "MAE_in_window": mae_in,
        "MAE_out_window": mae_out,
    })

    lb.to_csv(os.path.join(DAT, f"leaderboard_{key}.csv"), index=False)
    pd.DataFrame({"Time": tte, "Soot": yte, "Soot_pred": best_pred}) \
        .sort_values("Time").to_csv(os.path.join(DAT, f"predictions_{key}.csv"), index=False)

    print(f"{key:<8}{best_name:<22}{lb.iloc[0]['R2']:>+9.4f}{mae_in:>10.4f}{mae_out:>10.4f}")

pd.DataFrame(summary_rows).to_csv(os.path.join(DAT, "summary.csv"), index=False)

all_models = [name for name, _ in model_zoo()]
wide = pd.DataFrame({
    "model": all_models,
    "Z":    [r2_per_set["Z"][m]    for m in all_models],
    "T":    [r2_per_set["T"][m]    for m in all_models],
    "cumZ": [r2_per_set["cumZ"][m] for m in all_models],
}).set_index("model").sort_values("cumZ", ascending=False)
wide.to_csv(os.path.join(DAT, "r2_wide.csv"))

print("\n===== R^2 BY SINGLE-SIGNAL FEATURE (AC) =====")
print(wide.to_string(float_format=lambda x: f"{x:+.4f}"))

# -----------------------------------------------------------------------------
# 3a) Bar plot: R^2 across the 10 models for Z, T, cumZ
# -----------------------------------------------------------------------------
models_sorted = wide.index.tolist()
xpos = np.arange(len(models_sorted))
w, clip = 0.27, -0.1

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(xpos - w, [max(r2_per_set["Z"][m],    clip) for m in models_sorted], w,
       label="Impedance (Z)", color="0.55")
ax.bar(xpos,     [max(r2_per_set["T"][m],    clip) for m in models_sorted], w,
       label="Temperature (T)", color="tab:blue")
ax.bar(xpos + w, [max(r2_per_set["cumZ"][m], clip) for m in models_sorted], w,
       label=r"cumZ $=\int Z\,dt$", color="tab:orange")
ax.set_xticks(xpos)
ax.set_xticklabels(models_sorted, rotation=30, ha="right")
ax.set_ylabel(r"Test $R^{2}$ (values $< -0.1$ clipped)")
ax.set_title("AC dynamic soot oxidation: which single signal best predicts remaining soot?\n"
             r"Test $R^{2}$ for Z, T, and cumZ each used alone "
             rf"(random 50/50 split, $n_{{\mathrm{{test}}}}={len(y)//2}$)")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylim(-0.12, 1.08)
ax.grid(True, alpha=0.3, axis="y")
ax.legend(loc="lower left", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "figure_Z_T_cumZ_R2.png"), dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# 3b) Fitting plots: measured vs predicted over time, best model per signal
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, key in zip(axes, ["Z", "T", "cumZ"]):
    best_name, tte, yte, pred, pretty = fits[key]
    order = np.argsort(tte)
    ts, yt, yp = tte[order], yte[order], pred[order]
    r2 = r2_per_set[key][best_name]
    ax.axvspan(ZONE[0], ZONE[1], color="tab:red", alpha=0.10)
    ax.scatter(ts, yt, s=14, color="0.35", label="Measured", zorder=2)
    ax.plot(ts, yp, color="tab:orange", linewidth=2, label="Predicted", zorder=3)
    ax.set_title(f"{pretty}\nbest: {best_name}   $R^2$={r2:.4f}")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
axes[0].set_ylabel("Remaining soot, conversion (%)")
fig.suptitle("AC soot oxidation -- best-model test fit per single signal "
             "(random 50/50 split, seed=0; red band = 6000-12000 s oxidation window)",
             y=1.02, fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "figure_fit_Z_T_cumZ.png"), dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# 3c) Signal overview: Temp, Z, cumZ, soot vs time
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
panels = [("Temp", "Temperature (degC)", "tab:blue"),
          ("Z", "Impedance Z (ohm)", "0.35"),
          ("cumZ", r"cumZ $=\int Z\,dt$ (ohm$\cdot$s)", "tab:orange"),
          ("Soot", "Remaining soot, conversion (%)", "tab:green")]
for ax, (col, ylab, c) in zip(axes.ravel(), panels):
    ax.axvspan(ZONE[0], ZONE[1], color="tab:red", alpha=0.10)
    ax.plot(d["Time"], d[col], color=c, linewidth=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
fig.suptitle("AC dynamic soot oxidation -- aligned signals (red band = 6000-12000 s window)",
             fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "figure_signals.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\n[DONE] artifacts written under {HERE}")
