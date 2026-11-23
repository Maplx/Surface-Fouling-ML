"""
Soot-loading series -- predict soot conversion from the DC sensor current.

Same pipeline as isotherm_exploration / isotherm_exploration_2, applied to the
four extra experiments in `Data summary for different soot loading in one
excel.xlsx` (sheet Ark1):

  50-1, 20-1, 10-1   catalyst-to-soot loading ratios
  wall flow          wall-flow structure

Each run is a temperature ramp (~325 -> 755 C at ~2 C/min, 230 min).

Two targets, both from the CO2 signal (ground truth):
  extent  = soot conversion (%)  = normalized cumulative CO2 ("how much burned")
  rate    = instantaneous CO2 (ppm)                          ("how fast now")

Current-derived features (the only inputs): I, dI/dt, cumI = integral I dt, all.
Random 50/50 split, seed=0, same 10-model zoo. MAE reported as % of full scale.
Each case is fitted and tested on its own.

Differences vs the isotherm runs (handled here):
  - Time is in minutes in the sheet; converted to seconds.
  - From ~167 min on, the current timestamps are written on a 10 s grid, so each
    stamp appears twice (the sampling itself stays 5 s). The first sample of
    each duplicated pair is moved back by one 5 s step.
  - No cooldown in the record, so the whole run is used.

Outputs (next to this script):
  figures/overview.png                         Temp / CO2 / conversion / current, all cases
  figures/figure_summary_conversion.png        conversion fit from I and from cumI, all cases
  figures/figure_fit_extent_{case}.png         conversion: best fit per feature set
  datasets/aligned_{case}.csv                  aligned Time, Temp, I, dIdt, cumI, CO2, rate, extent
  data/{case}/leaderboard_*.csv, predictions_*.csv, r2_wide_*.csv
  data/summary.csv                             best model per case x target x feature set

Run:  python soot_loading_exploration/run.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "..", "Data summary for different soot loading in one excel.xlsx")
SHEET = "Ark1"
DSET = os.path.join(HERE, "datasets")
FIG = os.path.join(HERE, "figures")
DAT = os.path.join(HERE, "data")
for p in (DSET, FIG, DAT):
    os.makedirs(p, exist_ok=True)

# key, label, (gas time, temp, CO2, current time, current) column names
CASES = [
    ("50-1", "Catalyst:soot 50:1",
     ("time-min-50-1", "T-C-50-1", "CO2-ppm-50-1", "I-Time-min-50-1", " Current/A-50-1")),
    ("20-1", "Catalyst:soot 20:1",
     ("time-min-20-1", "T-C-20-1", "CO2-ppm-20-1", "I-Time-min-20-1", " Current/A-20-1")),
    ("10-1", "Catalyst:soot 10:1",
     ("time-min-10-1", "T-C-10-1", "CO2-ppm-10-1", "I-Time-min-10-1", " Current/A-10-1")),
    ("wall_flow", "Wall flow",
     ("time-min-wall flow", "T-C-wall flow", "CO2-wall flow", "Time-min-wall flow",
      " Current/A-wall flow")),
]

FEATURE_SETS = [
    ("I",    "Current (I)",            ["I"]),
    ("dIdt", "dI/dt",                  ["dIdt"]),
    ("cumI", r"cumI ($=\int I\,dt$)",  ["cumI"]),
    ("all",  "I + dI/dt + cumI",       ["I", "dIdt", "cumI"]),
]
TARGETS = [
    # "extent" = normalized cumulative CO2 = soot conversion (% burned so far)
    ("extent", "Soot conversion (%)",            "extent"),
    ("rate",   "Oxidation rate -- CO2 (ppm)",    "rate"),
]
COLORS = {"I": "0.55", "dIdt": "tab:green", "cumI": "tab:orange", "all": "tab:purple"}


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


def fix_duplicate_stamps(t):
    """Timestamps written on a coarser grid appear twice; move the first of each
    pair back by one native sampling step so the axis is strictly increasing."""
    t = t.copy()
    dt = np.diff(t)
    step = np.median(dt[dt > 0])
    dup = np.where(dt <= 0)[0]
    t[dup] -= step
    assert (np.diff(t) > 0).all(), "current time axis still not increasing"
    return t, len(dup)


def load_case(raw, cols):
    """Features on the native current grid, aligned to the CO2 timestamps."""
    tg, T, C, ti, I = cols
    gas = raw[[tg, T, C]].dropna().astype(float).sort_values(tg).to_numpy()
    cur = raw[[ti, I]].dropna().astype(float).to_numpy()

    ct, ndup = fix_duplicate_stamps(cur[:, 0] * 60.0)          # min -> s
    cI = cur[:, 1]
    cdI = np.gradient(smooth(cI, 5), ct)
    ccum = np.concatenate(([0.0], np.cumsum(0.5 * (cI[1:] + cI[:-1]) * np.diff(ct))))

    gt = gas[:, 0] * 60.0
    keep = (gt >= ct.min()) & (gt <= ct.max())
    tt = gt[keep]
    d = pd.DataFrame({"Time": tt, "Temp": gas[keep, 1]})
    d["I"] = np.interp(tt, ct, cI)
    d["dIdt"] = np.interp(tt, ct, cdI)
    d["cumI"] = np.interp(tt, ct, ccum)
    d["CO2"] = np.clip(gas[keep, 2], 0, None)
    co2 = d["CO2"].to_numpy()
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (co2[1:] + co2[:-1]) * np.diff(tt))))
    d["rate"] = d["CO2"]                          # oxidation rate (ppm)
    d["extent"] = cum / cum[-1] * 100.0           # soot conversion (%)
    return d, (ct, cI), ndup


raw = pd.read_excel(XLSX, sheet_name=SHEET)
summary_rows = []
data = {}          # case -> aligned frame
fits_all = {}      # (case, target, feature) -> (best_name, tte, yte, pred, r2, mae_pct)

for ckey, clabel, cols in CASES:
    d, cur, ndup = load_case(raw, cols)
    data[ckey] = (d, cur)
    d.to_csv(os.path.join(DSET, f"aligned_{ckey}.csv"), index=False)
    cdir = os.path.join(DAT, ckey)
    os.makedirs(cdir, exist_ok=True)
    t = d["Time"].to_numpy()
    print(f"\n######## {clabel}: {len(d)} rows, time [{t.min():.0f}, {t.max():.0f}] s, "
          f"{ndup} duplicated current stamps fixed")

    for tkey, tlabel, tcol in TARGETS:
        y = d[tcol].to_numpy()
        yspan = float(y.max() - y.min())
        r2_per_set, mae_pct_per_set, fits = {}, {}, {}
        print(f"===== TARGET: {tkey} ({tlabel}) =====")
        print(f"{'feature':<8}{'best model':<22}{'best R2':>9}{'MAE%':>8}")
        print("-" * 47)
        for fkey, fpretty, fcols in FEATURE_SETS:
            X = d[fcols].to_numpy()
            lb, best_name, best_pred, yte, tte, all_r2 = run_cell(X, y, t)
            r2_per_set[fkey] = all_r2
            mae_pct = lb.iloc[0]["MAE"] / yspan * 100.0
            mae_pct_per_set[fkey] = mae_pct
            fits[fkey] = (best_name, tte, yte, best_pred, fpretty)
            fits_all[(ckey, tkey, fkey)] = (best_name, tte, yte, best_pred,
                                            lb.iloc[0]["R2"], mae_pct)
            summary_rows.append({"case": ckey, "target": tkey, "feature_set": fkey,
                                 "best_model": best_name, "best_R2": lb.iloc[0]["R2"],
                                 "best_MAE": lb.iloc[0]["MAE"], "best_MAE_pct": mae_pct})
            lb.to_csv(os.path.join(cdir, f"leaderboard_{tkey}_{fkey}.csv"), index=False)
            pd.DataFrame({"Time": tte, "true": yte, "pred": best_pred}) \
                .sort_values("Time").to_csv(os.path.join(cdir, f"predictions_{tkey}_{fkey}.csv"),
                                            index=False)
            print(f"{fkey:<8}{best_name:<22}{lb.iloc[0]['R2']:>+9.4f}{mae_pct:>7.1f}%")

        all_models = [n for n, _ in model_zoo()]
        wide = pd.DataFrame({"model": all_models,
                             **{fk: [r2_per_set[fk][m] for m in all_models]
                                for fk, _, _ in FEATURE_SETS}}).set_index("model")
        wide.to_csv(os.path.join(cdir, f"r2_wide_{tkey}.csv"))

        # ---- fitting figure (soot conversion only) ----
        if tkey != "extent":
            continue
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), sharey=True)
        for ax_, (fkey, fpretty, _) in zip(axes, FEATURE_SETS):
            best_name, tte, yte, pred, pretty = fits[fkey]
            order = np.argsort(tte)
            ax_.scatter(tte[order] / 60, yte[order], s=14, color="0.35", label="Measured", zorder=2)
            ax_.plot(tte[order] / 60, pred[order], color=COLORS[fkey], linewidth=2,
                     label="Predicted", zorder=3)
            ax_.set_title(f"{pretty}\nbest: {best_name}   $R^2$={r2_per_set[fkey][best_name]:.4f}   "
                          f"MAE={mae_pct_per_set[fkey]:.1f}%")
            ax_.set_xlabel("Time (min)")
            ax_.grid(True, alpha=0.3)
            ax_.legend(loc="best", fontsize=8)
        axes[0].set_ylabel(tlabel)
        fig.suptitle(f"{clabel} - predicting {tlabel} from the DC sensor current "
                     f"(random 50/50 split, seed 0)", y=1.03, fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(FIG, f"figure_fit_{tkey}_{ckey}.png"), dpi=150, bbox_inches="tight")
        plt.close()

pd.DataFrame(summary_rows).to_csv(os.path.join(DAT, "summary.csv"), index=False)

# -----------------------------------------------------------------------------
# Overview: temperature, CO2, conversion, current for all cases
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(4, len(CASES), figsize=(5 * len(CASES), 11), sharex=True)
for j, (ckey, clabel, _) in enumerate(CASES):
    d, (ct, cI) = data[ckey]
    tm = d["Time"] / 60
    ax[0, j].plot(tm, d["Temp"], color="tab:blue")
    ax[1, j].plot(tm, d["CO2"], color="tab:green")
    ax[2, j].plot(tm, d["extent"], color="tab:orange"); ax[2, j].set_ylim(-5, 105)
    ax[3, j].plot(ct / 60, cI * 1e3, color="0.3")
    ax[0, j].set_title(clabel)
    ax[3, j].set_xlabel("Time (min)")
    for x in ax[:, j]:
        x.grid(True, alpha=0.3)
for i, lab in enumerate(["Temperature (C)", "CO2 (ppm)\n= oxidation rate",
                         "cumulative CO2\n= soot conversion (%)", "Current (mA)\n= DC sensor"]):
    ax[i, 0].set_ylabel(lab)
fig.suptitle("Soot-loading series: temperature, CO2 (rate), cumulative CO2 (soot conversion), "
             "DC current", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "overview.png"), dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# Summary: soot conversion from raw current vs cumI, all cases
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(2, len(CASES), figsize=(5 * len(CASES), 8), sharex=True, sharey=True)
for j, (ckey, clabel, _) in enumerate(CASES):
    for i, (fkey, fname) in enumerate([("I", "Current (I)"), ("cumI", r"cumI ($=\int I\,dt$)")]):
        best_name, tte, yte, pred, r2, mae = fits_all[(ckey, "extent", fkey)]
        order = np.argsort(tte)
        ax[i, j].scatter(tte[order] / 60, yte[order], s=14, color="0.35", label="Measured", zorder=2)
        ax[i, j].plot(tte[order] / 60, pred[order], color=COLORS[fkey], linewidth=2,
                      label="Predicted", zorder=3)
        ax[i, j].set_title(f"{clabel} - {fname}\n{best_name}   $R^2$={r2:.4f}   MAE={mae:.1f}%",
                           fontsize=10)
        ax[i, j].grid(True, alpha=0.3)
        ax[i, j].legend(loc="upper left", fontsize=8)
    ax[1, j].set_xlabel("Time (min)")
for i in range(2):
    ax[i, 0].set_ylabel("Soot conversion (%)")
fig.suptitle("Predicting soot conversion from the DC sensor current "
             "(held-out half of a random 50/50 split, seed 0)", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "figure_summary_conversion.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\n[DONE] artifacts under {HERE}")
