# Surface-Fouling-ML

## Overview
This workspace contains experiments for AC and DC sensing. The temperature is fixed at 600C.

## New lagged model (no interpolation)
Run the lagged AC-to-NO/NO2 model:

```bash
.venv/bin/python ac_lagged_interpolation.py
```

Optional flags:

```bash
.venv/bin/python ac_lagged_interpolation.py --lags 0,5,10,30,60 --split chrono --test-size 0.5
```

## Dependencies
Install from requirements:

```bash
.venv/bin/python -m pip install -r requirements.txt
```
