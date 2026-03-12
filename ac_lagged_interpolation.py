import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


CSV_PATH = "12-02-2025 raw sensor data.csv"
RANDOM_SEED = 0


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}")


def align_gas_to_ac(df: pd.DataFrame) -> pd.DataFrame:
    c_t_ac = pick_col(df, ["AC-time/s", "AC-time/s.1"])
    c_right = pick_col(df, ["AC-NO-600C-Right", "AC-NO-600C-Right.1"])
    c_left = pick_col(df, ["AC-NO-600C-Left", "AC-NO-600C-Left.1"])

    c_t_gas = pick_col(df, ["Time-gas (s)", "Time-gas (s).1"])
    c_no = pick_col(df, ["NO concentration (ppm)", "NO concentration (ppm).1"])
    c_no2 = pick_col(df, ["NO2 concentration (ppm)", "NO2 concentration (ppm).1"])

    ac = df[[c_t_ac, c_right, c_left]].dropna().copy()
    ac = ac.rename(columns={c_t_ac: "t_ac", c_right: "AC_right", c_left: "AC_left"})
    ac = (ac.groupby("t_ac", as_index=False)[["AC_right", "AC_left"]]
          .mean()
          .sort_values("t_ac")
          .reset_index(drop=True))

    gas = df[[c_t_gas, c_no, c_no2]].dropna().copy()
    gas = gas.rename(columns={c_t_gas: "t_gas", c_no: "NO", c_no2: "NO2"})
    gas = gas.reset_index(drop=True)

    print(f"[INFO] AC points:  {len(ac)}  time range [{ac.t_ac.min()}, {ac.t_ac.max()}]")
    print(f"[INFO] Gas points: {len(gas)} time range [{gas.t_gas.min()}, {gas.t_gas.max()}]")

    gas["t_gas_round"] = gas["t_gas"].round().astype(int)

    t_ac = ac["t_ac"].to_numpy()
    y_right = ac["AC_right"].to_numpy()
    y_left = ac["AC_left"].to_numpy()

    t_query = gas["t_gas_round"].to_numpy().astype(float)
    t_min, t_max = float(t_ac.min()), float(t_ac.max())
    out_of_range = (t_query < t_min) | (t_query > t_max)
    num_oor = int(out_of_range.sum())

    if num_oor > 0:
        print("[WARN] Some rounded gas times are outside AC range; clipping.")
        t_query = np.clip(t_query, t_min, t_max)

    gas["t_ac_used"] = t_query
    gas["AC_right_aligned"] = np.interp(t_query, t_ac, y_right)
    gas["AC_left_aligned"] = np.interp(t_query, t_ac, y_left)

    print(f"[INFO] Aligned samples: {len(gas)}")
    print("\n[Aligned preview]")
    print(gas[["t_gas", "t_gas_round", "t_ac_used", "NO", "NO2",
               "AC_right_aligned", "AC_left_aligned"]].head(8))

    return gas


def add_lag_features(gas_df: pd.DataFrame, lags: list[int]) -> tuple[pd.DataFrame, list[str]]:
    gas_df = gas_df.copy()
    feature_cols = []
    for lag in lags:
        for col in ("AC_right_aligned", "AC_left_aligned"):
            out_col = f"{col}_lag_{lag}"
            gas_df[out_col] = gas_df[col].shift(lag)
            feature_cols.append(out_col)

    gas_df = gas_df.dropna().reset_index(drop=True)
    return gas_df, feature_cols


def parse_lags(lags_csv: str) -> list[int]:
    lags = [int(x.strip()) for x in lags_csv.split(",") if x.strip()]
    lags = sorted(set(lags))
    if not lags:
        raise ValueError("At least one lag is required")
    return lags


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict NO/NO2 from lagged AC signals.")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to the CSV file.")
    parser.add_argument("--lags", default="0,5,10,30,60", help="Comma-separated lags in seconds.")
    parser.add_argument("--split", choices=["chrono", "random"], default="chrono")
    parser.add_argument("--test-size", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    gas_df = align_gas_to_ac(df)

    lags = parse_lags(args.lags)
    gas_df = gas_df.sort_values("t_gas_round").reset_index(drop=True)
    gas_df, feature_cols = add_lag_features(gas_df, lags)
    print(f"[INFO] Using lags (s): {lags}")
    print(f"[INFO] Feature columns: {len(feature_cols)}")

    X = gas_df[feature_cols].to_numpy()
    y = gas_df[["NO", "NO2"]].to_numpy()
    t = gas_df["t_gas_round"].to_numpy()

    if args.split == "random":
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            X, y, t, test_size=args.test_size, shuffle=True, random_state=RANDOM_SEED
        )
    else:
        n = len(gas_df)
        split = int(n * (1 - args.test_size))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test, t_test = X[split:], y[split:], t[split:]

    print(f"[INFO] Train size = {len(y_train)}, Test size = {len(y_test)}")

    def eval_model(name: str, model):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
        rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput="raw_values"))
        r2 = r2_score(y_test, y_pred, multioutput="raw_values")

        return {
            "model": name,
            "MAE": float(np.mean(mae)),
            "RMSE": float(np.mean(rmse)),
            "R2": float(np.mean(r2)),
            "MAE_NO": float(mae[0]),
            "MAE_NO2": float(mae[1]),
            "RMSE_NO": float(rmse[0]),
            "RMSE_NO2": float(rmse[1]),
            "R2_NO": float(r2[0]),
            "R2_NO2": float(r2[1]),
        }

    models = [
        ("LinearRegression", LinearRegression()),
        ("Ridge(alpha=1.0)", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        ("Lasso(alpha=1e-3)", Lasso(alpha=1e-3, random_state=RANDOM_SEED, max_iter=10000)),
        ("ElasticNet(alpha=1e-3,l1=0.5)", ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=10000)),
        ("SVR(RBF)", MultiOutputRegressor(Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.001)),
        ]))),
        ("KNN(k=5)", Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=5)),
        ])),
        ("DecisionTree", DecisionTreeRegressor(random_state=RANDOM_SEED, max_depth=5)),
        ("RandomForest", RandomForestRegressor(
            n_estimators=500, random_state=RANDOM_SEED, max_depth=None, min_samples_leaf=2
        )),
        ("GradientBoosting", MultiOutputRegressor(GradientBoostingRegressor(random_state=RANDOM_SEED))),
        ("HistGradientBoosting", MultiOutputRegressor(HistGradientBoostingRegressor(random_state=RANDOM_SEED))),
    ]

    results = []
    for name, model in models:
        try:
            results.append(eval_model(name, model))
        except Exception as exc:
            print(f"[WARN] Model failed: {name} -> {type(exc).__name__}: {exc}")
            results.append({
                "model": name,
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
                "MAE_NO": np.nan,
                "MAE_NO2": np.nan,
                "RMSE_NO": np.nan,
                "RMSE_NO2": np.nan,
                "R2_NO": np.nan,
                "R2_NO2": np.nan,
            })

    res_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    print("\n===== MODEL COMPARISON: Lagged AC -> NO/NO2 =====")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    best_name = res_df.iloc[0]["model"]
    best_model = None
    for name, mdl in models:
        if name == best_name:
            best_model = mdl
            break

    if best_model is None:
        raise RuntimeError(f"Could not find model object for: {best_name}")

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    order = np.argsort(t_test)
    t_sorted = t_test[order]
    y_sorted = y_test[order]
    pred_sorted = y_pred[order]

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(t_sorted, y_sorted[:, 0], label="NO True (test)", linewidth=2)
    axes[0].plot(t_sorted, pred_sorted[:, 0], label="NO Predicted (test)", linewidth=2)
    axes[0].set_ylabel("NO (ppm)")
    axes[0].set_title(f"Lagged AC -> NO ({best_name})")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t_sorted, y_sorted[:, 1], label="NO2 True (test)", linewidth=2)
    axes[1].plot(t_sorted, pred_sorted[:, 1], label="NO2 Predicted (test)", linewidth=2)
    axes[1].set_xlabel("Time (s) on AC grid")
    axes[1].set_ylabel("NO2 (ppm)")
    axes[1].set_title(f"Lagged AC -> NO2 ({best_name})")
    axes[1].grid(True)
    axes[1].legend()

    plt.show()


if __name__ == "__main__":
    main()
