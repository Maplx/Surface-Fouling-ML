# Surface-Fouling-ML

Machine-learning analysis of a SiC-ZnO/LSCO electrochemical sensor. The main
question: **can the sensor's own electrical signal tell us how much soot is on
it**, without a thermocouple or a CO₂ analyzer?

The sensor runs in two modes: DC (resistance / current at 1 V) and AC (impedance
at 0.2 V).

## Main finding

A single reading of the sensor signal is not enough, but its running integral
over time is. This holds for every soot experiment in the repo:

| Experiment | Folder | Raw signal (R²) | Integrated signal (R²) |
|---|---|---|---|
| DC temperature ramp | [soot_mass_sensor_signals/](soot_mass_sensor_signals/) | `R`: 0.80 | `cumR = ∫R dt`: 0.9996 |
| AC temperature ramp | [ac_soot_oxidation_si/](ac_soot_oxidation_si/) | `Z`: 0.93 | `cumZ = ∫Z dt`: 0.9998 |
| DC isothermal steps, run 1 | [isotherm_exploration/](isotherm_exploration/) | `I`: 0.97 | `cumI = ∫I dt`: 0.998 |
| DC isothermal steps, run 2 | [isotherm_exploration_2/](isotherm_exploration_2/) | `I`: 0.86 | `cumI = ∫I dt`: 0.998 |

Each number is the best of a 10-model zoo on a random 50/50 train/test split
(seed 0) **within a single run**. The integrated signal increases monotonically
with time, so these scores show that the soot curve can be interpolated inside
one run. A model trained on one run has not yet been tested on another.

## Where to start

To estimate soot mass from resistance, read
[soot_mass_sensor_signals/](soot_mass_sensor_signals/). Its README walks through
four steps:

1. `01_temperature_reveal`: `R` alone reaches R² 0.80; temperature alone reaches 0.9997.
2. `02_predict_T_from_R`: `R` is flat during the main oxidation window
   (6000–12000 s), so one reading cannot separate different soot loadings.
3. `03_resistance_derived_features`: adding `dR/dt` and `cumR` recovers the
   accuracy; `cumR` alone is enough.
4. `04_R_T_cumR_comparison`: `R`, `T` and `cumR` compared across all 10 models.

## What "soot conversion" means in each folder

The soot target is not defined the same way everywhere:

| Folder | Target | Range | Meaning |
|---|---|---|---|
| `soot_mass_sensor_signals` | `Soot left-mg` | 0.5 → 0 mg | soot **remaining** |
| `ac_soot_oxidation_si` | `Soot conversion-ac` | 100 → 0 % | soot **remaining**, despite the column name |
| `isotherm_exploration`, `isotherm_exploration_2` | `extent` | 0 → 100 % | soot **burned** (normalized cumulative CO₂) |

To convert between them:

```
soot remaining (%)  = 100 − soot burned (%)
soot remaining (mg) = initial mass × soot remaining (%) / 100
```

The initial mass in the DC ramp is 0.5 mg.

## Repository layout

| Path | Contents |
|---|---|
| [soot_mass_sensor_signals/](soot_mass_sensor_signals/) | DC ramp: remaining soot mass from `R`, `T`, `cumR` |
| [ac_soot_oxidation_si/](ac_soot_oxidation_si/) | AC ramp: remaining soot from `Z`, `T`, `cumZ` |
| [isotherm_exploration/](isotherm_exploration/), [isotherm_exploration_2/](isotherm_exploration_2/) | Two isothermal runs: soot burned and oxidation rate from DC current |
| [thickness_from_Z/](thickness_from_Z/) | Separate study: coating thickness from AC impedance (17.5 kHz, 400 °C) |
| [soot_mass_report/](soot_mass_report/) | Earlier LaTeX report on `R` versus `R + T`; superseded by `soot_mass_sensor_signals` |
| [legacy_scripts/](legacy_scripts/), [outputs/](outputs/) | Earlier scripts and figures, including NOx (NO/NO₂) prediction at 600 °C |
| `12-02-2025 raw sensor data.xlsx` | Raw data for the NOx and DC ramp experiments |
| `12-02-2025 sensor data discussion (1) copy.pdf` | Background on the sensor and the experiments |

Each study folder has its own README, a `run.py` that regenerates its results,
and `data/`, `datasets/` and `figures/` subfolders.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running

The DC ramp study runs from the data in this repo:

```bash
python soot_mass_sensor_signals/01_temperature_reveal/run.py
python soot_mass_sensor_signals/02_predict_T_from_R/run.py
python soot_mass_sensor_signals/03_resistance_derived_features/run.py
python soot_mass_sensor_signals/04_R_T_cumR_comparison/run.py
```

The other `run.py` scripts read Excel files that are **not in this repo**. Place
them in the directory that contains the `Surface-Fouling-ML` folder:

| Script | Needs |
|---|---|
| `ac_soot_oxidation_si/run.py`, `isotherm_exploration/run.py` | `Copy of AC soot oxidation ramping ML data.xlsx` |
| `isotherm_exploration_2/run.py` | `New isothermal soot oxidation date.xlsx` |
| `thickness_from_Z/run.py` | `three different tests results.xlsx` |

The aligned tables these scripts produce are already saved in each folder's
`datasets/`, so the results can be inspected without the Excel files.
