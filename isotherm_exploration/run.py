"""
Isotherm case -- predict soot oxidation from the DC sensor current.

The team wants, as a first step, to predict oxidation from the sensor CURRENT
alone. "Oxidation" has two readings, so we do both, same 10-model zoo:

  TARGET A  extent  = cumulative soot oxidized (%)   [ = normalized  integral CO2 ]
                      "how much soot has burned so far"  -- the monitoring quantity
  TARGET B  rate    = instantaneous CO2 (ppm)        [ = oxidation rate right now ]
                      "how fast it is burning"        -- what the two-mechanism
                                                        deconvolution needs

Current-derived features (the only inputs -- no temperature):

  I       raw current (A)
  dI/dt   rate of change of current (soot burns -> conductive bridge disappears
          -> current drops, so dI/dt carries the instantaneous rate)
  cumI    = integral I dt  (A*s), cumulative current -- the current analogue of
          cumR/cumZ; monotonic, so it tracks how much soot has gone
  all     I + dI/dt + cumI together

Data: sheet "Isotherm condition". Current is on its own fast time axis; Temp and
CO2 on another. We compute dI/dt and cumI on the native current grid, then
interpolate everything onto the CO2 timestamps (the target's grid) over the
heating program (<= 10900 s, before cooldown). Random 50/50 split, seed=0 --
same protocol as the ramp / DC studies, for comparability.

Outputs (next to this script):
  datasets/isotherm_aligned.csv        aligned Time, Temp, I, dIdt, cumI, CO2, extent
  data/leaderboard_{target}_{feat}.csv 10-model leaderboard per (target, feature)
  data/predictions_{target}_{feat}.csv true vs predicted at each test point
  data/summary.csv                     best model + R^2 per (target, feature)
  data/r2_wide_{target}.csv            10 models x feature sets, R^2 side by side
  figures/figure_fit_{target}.png      best-model fit per feature set
  figures/figure_R2_{target}.png       bar plot: R^2, 10 models x feature sets

Run:  python Surface-Fouling-ML/isotherm_exploration/run.py
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
END = 10900.0                          # end of heating program (before cooldown)
HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "..", "..", "Copy of AC soot oxidation ramping ML data.xlsx")
SHEET = "Isotherm condition"
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
        rows.append({"model": name,
                     "MAE": float(mean_absolute_error(yte, pred)),
                     "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
                     "R2": float(r2)})
        if best is None or r2 > best["r2"]:
            best = {"name": name, "r2": r2, "pred": pred}
    lb = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return lb, best["name"], best["pred"], yte, tte, all_r2


def smooth(x, w=5):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


# -----------------------------------------------------------------------------
# 1) Load + build features on the native current grid, then align to CO2 grid
# -----------------------------------------------------------------------------
raw = pd.read_excel(XLSX, sheet_name=SHEET)

cur = (raw[["TIME-C-0.9", "Current-ISOTHERM"]].dropna()
       .rename(columns={"TIME-C-0.9": "t", "Current-ISOTHERM": "I"})
       .sort_values("t").reset_index(drop=True))
tc = (raw[["TIME-T-0.7", "Temperature-ISOTHERM", "CO2-ISOTHERM"]].dropna()
      .rename(columns={"TIME-T-0.7": "t", "Temperature-ISOTHERM": "Temp",
                       "CO2-ISOTHERM": "CO2"})
      .sort_values("t").reset_index(drop=True))

# derivative + cumulative integral of current on its own (fine, uniform) grid
ct, cI = cur["t"].to_numpy(), cur["I"].to_numpy()
cIs = smooth(cI, 5)
cdI = np.gradient(cIs, ct)
ccum = np.concatenate(([0.0], np.cumsum(0.5 * (cI[1:] + cI[:-1]) * np.diff(ct))))

# align onto the CO2 timestamps over the heating program
d = tc[(tc["t"] >= max(cur["t"].min(), tc["t"].min())) & (tc["t"] <= END)].copy().reset_index(drop=True)
tt = d["t"].to_numpy()
d["I"] = np.interp(tt, ct, cI)
d["dIdt"] = np.interp(tt, ct, cdI)
d["cumI"] = np.interp(tt, ct, ccum)
d["CO2"] = np.clip(d["CO2"].to_numpy(), 0, None)

# TARGET A: cumulative soot oxidized (%), normalized integral of CO2 (the rate)
cumCO2 = np.concatenate(([0.0], np.cumsum(0.5 * (d["CO2"].values[1:] + d["CO2"].values[:-1]) * np.diff(tt))))
d["extent"] = cumCO2 / cumCO2[-1] * 100.0
# TARGET B: oxidation rate = instantaneous CO2
d["rate"] = d["CO2"]

d = d[["t", "Temp", "I", "dIdt", "cumI", "CO2", "rate", "extent"]].rename(columns={"t": "Time"})
d.to_csv(os.path.join(DSET, "isotherm_aligned.csv"), index=False)
print(f"[INFO] aligned {len(d)} rows, time [{tt.min():.0f}, {tt.max():.0f}] s")

t = d["Time"].to_numpy()

FEATURE_SETS = [
    ("I",    "Current (I)",            ["I"]),
    ("dIdt", "dI/dt",                  ["dIdt"]),
    ("cumI", r"cumI ($=\int I\,dt$)",  ["cumI"]),
    ("all",  "I + dI/dt + cumI",       ["I", "dIdt", "cumI"]),
]
TARGETS = [
    # "extent" = normalized cumulative CO2 = soot conversion (% of soot burned so far)
    ("extent", "Soot conversion (%)",            "extent"),
    ("rate",   "Oxidation rate -- CO2 (ppm)",    "rate"),
]

colors = {"I": "0.55", "dIdt": "tab:green", "cumI": "tab:orange", "all": "tab:purple"}
summary_rows = []

for tkey, tlabel, tcol in TARGETS:
    y = d[tcol].to_numpy()
    yspan = float(y.max() - y.min())          # full scale, for normalized MAE
    r2_per_set, mae_pct_per_set, fits = {}, {}, {}
    print(f"\n===== TARGET: {tkey} ({tlabel}) =====")
    print(f"{'feature':<8}{'best model':<22}{'best R2':>9}{'MAE%':>8}")
    print("-" * 47)
    for fkey, fpretty, cols in FEATURE_SETS:
        X = d[cols].to_numpy()
        lb, best_name, best_pred, yte, tte, all_r2 = run_cell(X, y, t)
        r2_per_set[fkey] = all_r2
        mae_pct = lb.iloc[0]["MAE"] / yspan * 100.0   # MAE as % of full scale
        mae_pct_per_set[fkey] = mae_pct
        fits[fkey] = (best_name, tte, yte, best_pred, fpretty)
        summary_rows.append({"target": tkey, "feature_set": fkey,
                             "best_model": best_name, "best_R2": lb.iloc[0]["R2"],
                             "best_MAE": lb.iloc[0]["MAE"], "best_MAE_pct": mae_pct})
        lb.to_csv(os.path.join(DAT, f"leaderboard_{tkey}_{fkey}.csv"), index=False)
        pd.DataFrame({"Time": tte, "true": yte, "pred": best_pred}) \
            .sort_values("Time").to_csv(os.path.join(DAT, f"predictions_{tkey}_{fkey}.csv"), index=False)
        print(f"{fkey:<8}{best_name:<22}{lb.iloc[0]['R2']:>+9.4f}{mae_pct:>7.1f}%")

    # wide R^2 table
    all_models = [n for n, _ in model_zoo()]
    wide = pd.DataFrame({"model": all_models,
                         **{fk: [r2_per_set[fk][m] for m in all_models]
                            for fk, _, _ in FEATURE_SETS}}).set_index("model")
    wide.to_csv(os.path.join(DAT, f"r2_wide_{tkey}.csv"))

    # ---- fitting figure: best model per feature set ----
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), sharey=True)
    for ax, (fkey, fpretty, _) in zip(axes, FEATURE_SETS):
        best_name, tte, yte, pred, pretty = fits[fkey]
        order = np.argsort(tte)
        ax.scatter(tte[order], yte[order], s=14, color="0.35", label="Measured", zorder=2)
        ax.plot(tte[order], pred[order], color=colors[fkey], linewidth=2, label="Predicted", zorder=3)
        ax.set_title(f"{pretty}\nbest: {best_name}   $R^2$={r2_per_set[fkey][best_name]:.3f}   "
                     f"MAE={mae_pct_per_set[fkey]:.1f}%")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel(tlabel)
    fig.suptitle(f"Isotherm - predicting {tlabel} from the DC sensor current "
                 f"(random 50/50 split, seed 0)", y=1.03, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, f"figure_fit_{tkey}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- bar plot: R^2 across models x feature sets ----
    ms = wide.mean(axis=1).sort_values(ascending=False).index.tolist()
    xp = np.arange(len(ms))
    nfs = len(FEATURE_SETS)
    w = 0.8 / nfs
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (fkey, fpretty, _) in enumerate(FEATURE_SETS):
        ax.bar(xp + (i - (nfs - 1) / 2) * w,
               [max(r2_per_set[fkey][m], -0.1) for m in ms], w,
               label=fpretty, color=colors[fkey])
    ax.set_xticks(xp)
    ax.set_xticklabels(ms, rotation=30, ha="right")
    ax.set_ylabel(r"Test $R^2$ (clipped at $-0.1$)")
    ax.set_title(f"Isotherm - predicting {tlabel} from current-derived signals\n"
                 f"random 50/50 split, $n_{{\\mathrm{{test}}}}={len(y)//2}$")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylim(-0.12, 1.08)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, f"figure_R2_{tkey}.png"), dpi=150, bbox_inches="tight")
    plt.close()

pd.DataFrame(summary_rows).to_csv(os.path.join(DAT, "summary.csv"), index=False)
print(f"\n[DONE] artifacts under {HERE}")
