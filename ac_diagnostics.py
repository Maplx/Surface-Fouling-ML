import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "12-02-2025 raw sensor data.csv"


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

    gas = df[[c_t_gas, c_no, c_no2]].dropna().copy().reset_index(drop=True)
    gas = gas.rename(columns={c_t_gas: "t_gas", c_no: "NO", c_no2: "NO2"})

    gas["t_gas_round"] = gas["t_gas"].round().astype(int)

    t_ac = ac["t_ac"].to_numpy()
    y_right = ac["AC_right"].to_numpy()
    y_left = ac["AC_left"].to_numpy()

    t_query = gas["t_gas_round"].to_numpy().astype(float)
    t_min, t_max = float(t_ac.min()), float(t_ac.max())
    t_query = np.clip(t_query, t_min, t_max)

    gas["AC_right_aligned"] = np.interp(t_query, t_ac, y_right)
    gas["AC_left_aligned"] = np.interp(t_query, t_ac, y_left)

    return gas


def report_basic_stats(df: pd.DataFrame) -> None:
    print("\n===== BASIC STATS =====")
    cols = ["AC_right_aligned", "AC_left_aligned", "NO", "NO2"]
    stats = df[cols].describe().T[["mean", "std", "min", "max"]]
    print(stats.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n===== NA / INVALID CHECK =====")
    print(df[cols].isna().sum())


def report_correlations(df: pd.DataFrame) -> None:
    print("\n===== PEARSON CORRELATION =====")
    cols = ["AC_right_aligned", "AC_left_aligned", "NO", "NO2"]
    corr = df[cols].corr(method="pearson")
    print(corr.to_string(float_format=lambda x: f"{x:.4f}"))


def plot_scatter(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2)

    axes[0, 0].scatter(df["AC_right_aligned"], df["NO"], s=12, alpha=0.6)
    axes[0, 0].set_xlabel("AC_right_aligned")
    axes[0, 0].set_ylabel("NO")

    axes[0, 1].scatter(df["AC_left_aligned"], df["NO"], s=12, alpha=0.6)
    axes[0, 1].set_xlabel("AC_left_aligned")
    axes[0, 1].set_ylabel("NO")

    axes[1, 0].scatter(df["AC_right_aligned"], df["NO2"], s=12, alpha=0.6)
    axes[1, 0].set_xlabel("AC_right_aligned")
    axes[1, 0].set_ylabel("NO2")

    axes[1, 1].scatter(df["AC_left_aligned"], df["NO2"], s=12, alpha=0.6)
    axes[1, 1].set_xlabel("AC_left_aligned")
    axes[1, 1].set_ylabel("NO2")

    plt.tight_layout()
    plt.show()


def plot_time_series(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(df["t_gas_round"], df["AC_right_aligned"], label="AC_right", linewidth=2)
    axes[0].plot(df["t_gas_round"], df["AC_left_aligned"], label="AC_left", linewidth=2)
    axes[0].set_ylabel("AC")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(df["t_gas_round"], df["NO"], label="NO", linewidth=2)
    axes[1].plot(df["t_gas_round"], df["NO2"], label="NO2", linewidth=2)
    axes[1].set_xlabel("t_gas_round (s)")
    axes[1].set_ylabel("Gas (ppm)")
    axes[1].grid(True)
    axes[1].legend()

    plt.show()


def main() -> None:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    gas_df = align_gas_to_ac(df)

    print(f"[INFO] Samples: {len(gas_df)}")
    report_basic_stats(gas_df)
    report_correlations(gas_df)
    plot_scatter(gas_df)
    plot_time_series(gas_df)


if __name__ == "__main__":
    main()
