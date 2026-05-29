# Predicting soot mass from sensor signals

The question behind this folder is simple to state: can we tell how much soot is
left on the sensor from the signals we measure, and which signal is actually
doing the work? The answer turned out to be more interesting than we expected, so
it's worth walking through how we got there.

Everything here uses a single dataset, [`datasets/native_379.csv`](datasets/native_379.csv)
(379 rows: time, temperature, resistance, and the soot mass we're trying to
predict), and a single evaluation scheme — a random 50/50 train/test split with
`seed=0`. We dropped the time-based split entirely; it was distorting the picture
more than it helped.

The story runs in three acts, one folder each. Every figure has the script that
made it sitting right next to it, along with the actual numbers behind the
plot — both the true-vs-predicted values we drew, and the full leaderboard of all
ten models we tried, so you can see why a given "best model" won.

```
soot_mass_resistance_only/
├── datasets/native_379.csv
├── 01_temperature_reveal/
├── 02_predict_T_from_R/
├── 03_resistance_derived_features/
└── _archive/          ← older time-split / dense-interpolation work, kept for reference
```

One thing to keep an eye on throughout: the **6000–12000 s window**, shaded red in
every figure. That's where soot oxidation is most intense, which makes it both the
most important part of the experiment and, as we'll see, the part where the
obvious approach falls apart.

## Act 1 — resistance can't do it alone, but temperature can

[`01_temperature_reveal/`](01_temperature_reveal/) — figure
[`figure_R_T_RT.png`](01_temperature_reveal/figure_R_T_RT.png), script
[`run.py`](01_temperature_reveal/run.py)

We started by predicting soot mass three ways — from resistance alone, from
temperature alone, and from both together — letting a zoo of ten models pick a
winner each time.

| Features                    | Best model   | R²      | MAE in the window | MAE elsewhere |
|-----------------------------|--------------|---------|-------------------|---------------|
| Resistance only             | DecisionTree | 0.80    | 0.114 mg          | 0.007 mg      |
| Temperature only            | SVR (RBF)    | 0.9997  | 0.005 mg          | 0.002 mg      |
| Resistance + Temperature    | SVR (RBF)    | 0.9997  | 0.003 mg          | 0.002 mg      |

Resistance on its own tops out around R²=0.80, and the error tells you exactly
where it's struggling: inside the oxidation window it's about sixteen times worse
than everywhere else. Add temperature and the fit becomes nearly perfect — but
here's the catch. Temperature *by itself* is already nearly perfect, and putting
resistance on top of it changes essentially nothing. So whatever makes `R+T` look
so good, it isn't the resistance. The temperature is carrying the whole thing.

(The plotted values are in `predictions_{R,T,RT}.csv`; the full model rankings,
best first, in `leaderboard_{R,T,RT}.csv`.)

## Act 2 — so what do resistance and temperature have to do with each other?

[`02_predict_T_from_R/`](02_predict_T_from_R/) — figure
[`figure_T_from_R.png`](02_predict_T_from_R/figure_T_from_R.png), script
[`run.py`](02_predict_T_from_R/run.py)

If temperature is what matters, the natural next question is how faithfully
resistance follows it. So we flipped the problem around and predicted temperature
*from* resistance.

| Best model           | R² overall | MAE in the window | MAE elsewhere |
|----------------------|------------|-------------------|---------------|
| HistGradientBoosting | 0.96       | 43 °C             | 8 °C          |

Across the whole run, resistance tracks temperature pretty well (R²≈0.96). But
zoom into the window and it falls apart — the error is five times larger, and the
figure shows why at a glance: the predicted temperature flat-lines around 500 °C
while the real temperature keeps climbing, from roughly 430 to 620 °C. In that
stretch the resistance has essentially stopped moving, so one resistance reading
no longer pins down one temperature.

That's the whole problem in a single picture. Resistance loses its grip on
temperature in exactly the window where it loses its grip on soot mass — it isn't
two failures, it's the same one. An instantaneous resistance reading just doesn't
carry enough information here.

## Act 3 — giving resistance its memory back

[`03_resistance_derived_features/`](03_resistance_derived_features/) — figures
[`figure_feature_progression.png`](03_resistance_derived_features/figure_feature_progression.png)
and [`figure_cumR_signal.png`](03_resistance_derived_features/figure_cumR_signal.png),
script [`run.py`](03_resistance_derived_features/run.py)

If the trouble is that one resistance value can't see the history of the process,
the fix is to build features that can. We tried two:

- **`dR/dt`**, how fast resistance is changing — a sense of the oxidation's
  momentum.
- **`cumR`**, the running integral of resistance over time, `∫₀ᵗ R dτ` (in Ω·s).
  This is a cumulative resistive-exposure quantity, and the materials group
  confirmed it has a genuine physical reading as accumulated oxidation. Unlike a
  single resistance sample, it's monotonic and it remembers everything that came
  before.

Adding them back one at a time:

| Features              | Best model        | R²      | MAE in the window | MAE elsewhere |
|-----------------------|-------------------|---------|-------------------|---------------|
| `R`                   | DecisionTree      | 0.80    | 0.114 mg          | 0.007 mg      |
| `R, dR/dt`            | KNN (k=5)         | 0.92    | 0.064 mg          | 0.003 mg      |
| `R, dR/dt, cumR`      | GradientBoosting  | 0.9996  | 0.005 mg          | 0.0002 mg     |
| `cumR` alone          | GradientBoosting  | 0.9996  | 0.005 mg          | 0.0002 mg     |

The climb from 0.80 to 0.92 to 0.9996 closes the gap all the way back to the
temperature ceiling — with nothing but resistance and things derived from it. And
`cumR` is plainly the piece that matters: on its own it already reaches R²=0.9996.

The signal plot makes the intuition concrete. Resistance itself wanders up and
down and is genuinely ambiguous inside the window; `dR/dt` is noisy and sits near
zero there; but `cumR` is a clean, monotonic curve that holds onto the cumulative
oxidation history a single resistance reading throws away.

## The arc, in a sentence

Resistance alone fails in the oxidation window because an instantaneous reading
loses its one-to-one link with the process there — which is also why it can't
track temperature in the same window. Restore that lost history with the
physically grounded `cumR` feature, and near-perfect prediction comes back from
resistance alone.

## Running it again

```bash
python soot_mass_resistance_only/01_temperature_reveal/run.py
python soot_mass_resistance_only/02_predict_T_from_R/run.py
python soot_mass_resistance_only/03_resistance_derived_features/run.py
```

Each script stands on its own, reads only `datasets/native_379.csv`, writes its
figures and CSVs beside itself, and is deterministic at `seed=0`.

A note on the data: `native_379.csv` was aligned from `12-02-2025 raw sensor
data.xlsx` (time, temperature and resistance from `Sheet1`, soot mass from the
`calculation` sheet), with the no-signal tail past 17500 s dropped. The original
build script and the earlier time-split and dense-interpolation experiments are
parked in [`_archive/`](_archive/) if you ever need them.
