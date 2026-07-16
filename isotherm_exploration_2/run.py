"""
Isotherm run #2 -- predict soot oxidation from the DC sensor current.

Same pipeline as isotherm_exploration (run #1), applied to the new dataset
`New isothermal soot oxidation date.xlsx` (sheet Ark1, columns *-ISOTHERM-2).

Two targets, both from the CO2 signal (ground truth):
  extent  = soot conversion (%)  = normalized cumulative CO2 ("how much burned")
  rate    = instantaneous CO2 (ppm)                          ("how fast now")

Current-derived features (the only inputs): I, dI/dt, cumI = integral I dt, all.
Random 50/50 split, seed=0, same 10-model zoo. MAE reported as % of full scale.

Differences vs run #1 (handled automatically):
  - CO2 and Temp are on separate time axes here (TIME-T-0.5 vs TIME-T-0.7).
  - The main burn happens in the 670 C step (run #1 burned out at 570 C).
  - Heating program ends ~11340 s; END is set to 11400 s (before cooldown).

Outputs (next to this script):
  figures/isotherm_overview.png        Temp / CO2 / conversion / current, 4 panels
  datasets/isotherm_aligned.csv        aligned Time, Temp, I, dIdt, cumI, CO2, rate, extent
  data/leaderboard_{target}_{feat}.csv, predictions_*.csv, summary.csv, r2_wide_*.csv
  figures/figure_fit_{extent,rate}.png, figure_R2_{extent,rate}.png

Run:  python Surface-Fouling-ML/isotherm_exploration_2/run.py
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
END = 11400.0                          # end of heating program (before cooldown)
HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "..", "..", "New isothermal soot oxidation date.xlsx")
SHEET = "Ark1"
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
# 1) Load; features on the native current grid; align to the CO2 timestamps
# -----------------------------------------------------------------------------
raw = pd.read_excel(XLSX, sheet_name=SHEET)

cur = (raw[["TIME-C-0.9", "C-ISOTHERM-2"]].dropna()
       .rename(columns={"TIME-C-0.9": "t", "C-ISOTHERM-2": "I"})
       .sort_values("t").reset_index(drop=True))
co2 = (raw[["TIME-T-0.5", "CO2-ISOTHERM-2"]].dropna()
       .rename(columns={"TIME-T-0.5": "t", "CO2-ISOTHERM-2": "CO2"})
       .sort_values("t").reset_index(drop=True))
tem = (raw[["TIME-T-0.7", "T-ISOTHERM-2"]].dropna()
       .rename(columns={"TIME-T-0.7": "t", "T-ISOTHERM-2": "Temp"})
       .sort_values("t").reset_index(drop=True))

ct, cI = cur["t"].to_numpy(), cur["I"].to_numpy()
cIs = smooth(cI, 5)
cdI = np.gradient(cIs, ct)
ccum = np.concatenate(([0.0], np.cumsum(0.5 * (cI[1:] + cI[:-1]) * np.diff(ct))))

t_lo = max(cur["t"].min(), co2["t"].min(), tem["t"].min())
d = co2[(co2["t"] >= t_lo) & (co2["t"] <= END)].copy().reset_index(drop=True)
tt = d["t"].to_numpy()
d["Temp"] = np.interp(tt, tem["t"], tem["Temp"])
d["I"] = np.interp(tt, ct, cI)
d["dIdt"] = np.interp(tt, ct, cdI)
d["cumI"] = np.interp(tt, ct, ccum)
d["CO2"] = np.clip(d["CO2"].to_numpy(), 0, None)

cumCO2 = np.concatenate(([0.0], np.cumsum(0.5 * (d["CO2"].values[1:] + d["CO2"].values[:-1]) * np.diff(tt))))
d["extent"] = cumCO2 / cumCO2[-1] * 100.0     # soot conversion (%)
d["rate"] = d["CO2"]                          # oxidation rate (ppm)

d = d[["t", "Temp", "I", "dIdt", "cumI", "CO2", "rate", "extent"]].rename(columns={"t": "Time"})
d.to_csv(os.path.join(DSET, "isotherm_aligned.csv"), index=False)
print(f"[INFO] aligned {len(d)} rows, time [{tt.min():.0f}, {tt.max():.0f}] s")

t = d["Time"].to_numpy()

# -----------------------------------------------------------------------------
# 2) Overview figure: stepped temperature, CO2, conversion, current
#    per-step share of total CO2 annotated on the temperature panel
# -----------------------------------------------------------------------------
T_arr = d["Temp"].to_numpy()
bounds = [t.min()]
for lvl in (420.0, 520.0, 620.0):                       # ramp midpoints between holds
    idx = np.where((T_arr[:-1] < lvl) & (T_arr[1:] >= lvl))[0]
    if len(idx):
        bounds.append(t[idx[0]])
bounds.append(t.max())
seg_names = ["~370C", "~470C", "~570C", "~670C"][:len(bounds) - 1]

tot = np.trapz(d["CO2"], t)
segs = []
for name, a, b in zip(seg_names, bounds[:-1], bounds[1:]):
    m = (t >= a) & (t <= b)
    share = np.trapz(d["CO2"][m], t[m]) / tot * 100
    segs.append((name, a, b, share))
    print(f"[STEP] {name}: t={a:.0f}-{b:.0f}s  soot burned = {share:.1f}%")

fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
for name, a, b, share in segs:
    for x in ax:
        x.axvspan(a, b, color="tab:red", alpha=0.06)
    ax[0].text((a + b) / 2, 700, f"{name}\n{share:.1f}%", ha="center", fontsize=9)
ax[0].plot(t, d["Temp"], color="tab:blue"); ax[0].set_ylabel("Temperature (C)")
ax[1].plot(t, d["CO2"], color="tab:green"); ax[1].set_ylabel("CO2 (ppm)\n= oxidation rate")
ax[2].plot(t, d["extent"], color="tab:orange"); ax[2].set_ylabel("cumulative CO2\n= soot conversion (%)")
ax[2].set_ylim(-5, 105)
ax[3].plot(cur["t"], cur["I"], color="0.3"); ax[3].set_ylabel("Current (A)\n= DC sensor")
ax[3].set_xlabel("Time (s)")
for x in ax:
    x.grid(True, alpha=0.3); x.set_xlim(0, END + 400)
fig.suptitle("Isotherm run #2: temperature, CO2 (rate), cumulative CO2 (soot conversion), DC current\n"
             "% = share of total CO2 (soot burned) in each isotherm step", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "isotherm_overview.png"), dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# 3) Model zoo on each target x feature set
# -----------------------------------------------------------------------------
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

colors = {"I": "0.55", "dIdt": "tab:green", "cumI": "tab:orange", "all": "tab:purple"}
summary_rows = []

for tkey, tlabel, tcol in TARGETS:
    y = d[tcol].to_numpy()
    yspan = float(y.max() - y.min())
    r2_per_set, mae_pct_per_set, fits = {}, {}, {}
    print(f"\n===== TARGET: {tkey} ({tlabel}) =====")
    print(f"{'feature':<8}{'best model':<22}{'best R2':>9}{'MAE%':>8}")
    print("-" * 47)
    for fkey, fpretty, cols in FEATURE_SETS:
        X = d[cols].to_numpy()
        lb, best_name, best_pred, yte, tte, all_r2 = run_cell(X, y, t)
        r2_per_set[fkey] = all_r2
        mae_pct = lb.iloc[0]["MAE"] / yspan * 100.0
        mae_pct_per_set[fkey] = mae_pct
        fits[fkey] = (best_name, tte, yte, best_pred, fpretty)
        summary_rows.append({"target": tkey, "feature_set": fkey,
                             "best_model": best_name, "best_R2": lb.iloc[0]["R2"],
                             "best_MAE": lb.iloc[0]["MAE"], "best_MAE_pct": mae_pct})
        lb.to_csv(os.path.join(DAT, f"leaderboard_{tkey}_{fkey}.csv"), index=False)
        pd.DataFrame({"Time": tte, "true": yte, "pred": best_pred}) \
            .sort_values("Time").to_csv(os.path.join(DAT, f"predictions_{tkey}_{fkey}.csv"), index=False)
        print(f"{fkey:<8}{best_name:<22}{lb.iloc[0]['R2']:>+9.4f}{mae_pct:>7.1f}%")

    all_models = [n for n, _ in model_zoo()]
    wide = pd.DataFrame({"model": all_models,
                         **{fk: [r2_per_set[fk][m] for m in all_models]
                            for fk, _, _ in FEATURE_SETS}}).set_index("model")
    wide.to_csv(os.path.join(DAT, f"r2_wide_{tkey}.csv"))

    # ---- fitting figure ----
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), sharey=True)
    for ax_, (fkey, fpretty, _) in zip(axes, FEATURE_SETS):
        best_name, tte, yte, pred, pretty = fits[fkey]
        order = np.argsort(tte)
        ax_.scatter(tte[order], yte[order], s=14, color="0.35", label="Measured", zorder=2)
        ax_.plot(tte[order], pred[order], color=colors[fkey], linewidth=2, label="Predicted", zorder=3)
        ax_.set_title(f"{pretty}\nbest: {best_name}   $R^2$={r2_per_set[fkey][best_name]:.3f}   "
                      f"MAE={mae_pct_per_set[fkey]:.1f}%")
        ax_.set_xlabel("Time (s)")
        ax_.grid(True, alpha=0.3)
        ax_.legend(loc="best", fontsize=8)
    axes[0].set_ylabel(tlabel)
    fig.suptitle(f"Isotherm run #2 - predicting {tlabel} from the DC sensor current "
                 f"(random 50/50 split, seed 0)", y=1.03, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, f"figure_fit_{tkey}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- R^2 bar plot ----
    ms = wide.mean(axis=1).sort_values(ascending=False).index.tolist()
    xp = np.arange(len(ms))
    nfs = len(FEATURE_SETS)
    w = 0.8 / nfs
    fig, ax_ = plt.subplots(figsize=(11, 5.5))
    for i, (fkey, fpretty, _) in enumerate(FEATURE_SETS):
        ax_.bar(xp + (i - (nfs - 1) / 2) * w,
                [max(r2_per_set[fkey][m], -0.1) for m in ms], w,
                label=fpretty, color=colors[fkey])
    ax_.set_xticks(xp)
    ax_.set_xticklabels(ms, rotation=30, ha="right")
    ax_.set_ylabel(r"Test $R^2$ (clipped at $-0.1$)")
    ax_.set_title(f"Isotherm run #2 - predicting {tlabel} from current-derived signals\n"
                  f"random 50/50 split, $n_{{\\mathrm{{test}}}}={len(y)//2}$")
    ax_.axhline(0, color="black", linewidth=0.6)
    ax_.set_ylim(-0.12, 1.08)
    ax_.grid(True, alpha=0.3, axis="y")
    ax_.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, f"figure_R2_{tkey}.png"), dpi=150, bbox_inches="tight")
    plt.close()

pd.DataFrame(summary_rows).to_csv(os.path.join(DAT, "summary.csv"), index=False)
print(f"\n[DONE] artifacts under {HERE}")
