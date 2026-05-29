# Soot Mass from Resistance Only (dense interpolation)

Self-contained bundle of everything we built for the "predict Soot left-mg
using **only** Resistance, with dense per-second interpolation to expand the
training set" experiment.

## What's in the Excel file (and which columns we used)

The source spreadsheet `12-02-2025 raw sensor data.xlsx` (in the repo root)
has two sheets:

| Sheet         | Shape       | Notes                                                    |
|---------------|-------------|----------------------------------------------------------|
| `Sheet1`      | 18000 × 42  | All raw sensor streams, packed side-by-side              |
| `calculation` | 412 × 3     | `CO2-DC-ppm`, `Soot left-%`, `Soot left-mg` (derived)    |

There are SEVERAL `Resistance`-like columns inside `Sheet1`. To be unambiguous:

| Sheet1 column                       | Non-null rows | What it is                                |
|-------------------------------------|---------------|-------------------------------------------|
| `NO-600-Resistance (ohm)-Right`     | 10000         | DC NOx sensor (used by `dc_random.py`)    |
| `NO-600-Resistance (ohm)-left`      | 10000         | DC NOx sensor (used by `dc_random.py`)    |
| **`Resistance`** (with `TIME-sensor`) | **1898**    | **DC soot sensor — what we use here**     |
| `Resistance.1`                       | 11000        | A duplicate-named later block             |
| ` Z/ohm`, ` Z/ohm.1`                 | 9000 / 5491  | AC impedance (used by `ac_*` scripts)     |

We use exactly the same columns as [dc_soot_mass.py](../dc_soot_mass.py) and
[dc_soot_mass_report.py](../dc_soot_mass_report.py):

- **Time / target alignment:** `Sheet1.Time-DC` (412 jointly-non-null with
  `Temp-DC`, `CO2-DC`) ↔ `calculation.Soot left-mg` (412 rows), aligned
  row-by-row. Cross-check: `Sheet1.CO2-DC` and `calculation.CO2-DC-ppm` have
  max abs diff of `0.0` over all 412 rows.
- **Feature source:** `Sheet1.TIME-sensor` + `Sheet1.Resistance` (1898 rows).
  Linearly interpolated onto the soot time axis.
- **Tail cutoff:** drop `Time-DC > 17500 s` (no-signal tail, same as the
  report). 412 → 379 native rows.

## Dense interpolation

Mirroring [dc_soot_random_interpolation.py](../dc_soot_random_interpolation.py)
and [dc_random_interpolation.py](../dc_random_interpolation.py), we build a
per-second time grid over the intersection of the resistance and soot time
ranges, then `np.interp` both signals onto it:

```
t_grid spans [47, 17471] s  →  17425 dense rows
```

Important: this is **linear interpolation between known points**. It does NOT
add new physical measurements. It just gives tree / kernel models more
support points, which is why dense interp helps non-linear models a bit and
helps linear models nothing.

## Files

```
soot_mass_resistance_only/
├── README.md                       # this file
├── dc_soot_mass_interpolation.py   # the script that generated everything
├── data/
│   ├── native_dataset.csv          # 379 rows after the 17500s cutoff
│   ├── dense_dataset.csv           # 17425 rows after per-second interpolation
│   ├── random_split_models.csv     # 10-model results under random 50/50 split
│   └── time_split_models.csv       # 10-model results under time 70/30 split
└── figures/
    ├── random_split_best.png       # best model: HistGradientBoosting, R²=0.852
    ├── time_split_best.png         # best model: KNN(k=11), R²=-20
    └── diagnostic_resistance_vs_soot.png   # WHY the results look like this
```

## Results — and why "time split" looks so bad

### Random 50/50 split (17425 dense rows)
| Model                  | R²    | MAE (mg) |
|-----------------------|-------|----------|
| HistGradientBoosting  | **0.852** | 0.039 |
| GradientBoosting       | 0.850 | 0.040 |
| DecisionTree           | 0.848 | 0.041 |
| KNN(k=5)               | 0.823 | 0.041 |
| RandomForest           | 0.822 | 0.041 |
| SVR(RBF)               | 0.744 | 0.071 |
| Linear / Ridge / Lasso | 0.355 | 0.168 |

Compared to native 412-row baseline (DecisionTree R²=0.80 in
`soot_mass_report/data/r2_wide.csv`), dense interp lifts the best Resistance-only
model from **0.80 → 0.85**. Modest but real. Linear models stay flat at 0.35
because the relationship is strongly non-linear.

### Time 70/30 split (early 12198 rows train, late 5227 rows test)
| Model                | R²       | MAE (mg) |
|---------------------|----------|----------|
| KNN(k=5)             | **-19.87** | 0.0084 |
| RandomForest         | -19.92   | 0.0084 |
| DecisionTree         | -22.42   | 0.0089 |
| Linear / Ridge / Lasso | **-30430** | 0.328 |

**Yes, all R² are negative — but read the MAE first.** MAE is only 0.008 mg
in the late test region. R² blows up because the held-out time window has
plateaued — the soot has finished oxidizing and barely changes (test stddev
is tiny). Any small bias divided by that tiny stddev produces a large
negative R². For a constant target, R² is a brittle metric.

## Why Resistance alone (even with 17000 rows) can't do better

See `figures/diagnostic_resistance_vs_soot.png`:

- **Pearson r(Resistance, Soot) = +0.60** — only moderate linear correlation
- **Spearman ρ(Resistance, Soot) = +0.93** — strong monotonic correlation
  → relationship is monotonic but heavily non-linear (why linear models cap at
  0.35 and trees reach 0.85)
- **Hysteresis:** in some resistance bins (e.g. R ≈ 180–290 Ω), the same R
  value corresponds to soot values ranging across ~0.2 mg. The system passes
  through these R values both during early heating and later oxidation, so
  R → Soot is not a single-valued function in those regions.
- The original 2-feature report (`soot_mass_report/`) found
  Temperature took 98.5% of the RandomForest importance because Temperature
  is nearly monotonic in time and serves as a proxy for "where in the
  experiment we are" — which Resistance alone cannot resolve in the
  hysteretic region.

**Bottom line:** dense interpolation is doing what it can (lifts random-split
R² from 0.80 to 0.85), but it cannot manufacture physical information that
isn't in the resistance signal. Resistance alone is fundamentally a noisier,
non-injective predictor of remaining soot mass than Resistance + Temperature.

## Reproducing

```bash
python dc_soot_mass_interpolation.py
```

Reads `12-02-2025 raw sensor data.xlsx` from the repo root and writes
`outputs/dc_soot_mass_interpolation_*` files. Random seed 0, deterministic.
The artifacts in this folder are copies of those outputs.
