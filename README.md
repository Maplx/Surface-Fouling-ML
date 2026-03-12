# Surface-Fouling-ML

## Overview
This workspace contains experiments for AC and DC sensing. The temperature is fixed at 600C.

What we did so far:
- AC NOx modeling: align gas time to AC time and predict NO/NO2 from AC right/left.
- DC NOx modeling: align gas time to DC resistance time and predict NO/NO2 from DC right/left.
- Dense interpolation variants: build per-second grids for AC/DC to test random-split performance.

## Setup
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Scripts
AC alignment + random split model zoo:

```bash
.venv/bin/python ac_random.py
```

AC dense interpolation + random split:

```bash
.venv/bin/python ac_random_interpolation.py
```

DC alignment + random split model zoo:

```bash
.venv/bin/python dc_random.py
```

DC dense interpolation + random split:

```bash
.venv/bin/python dc_random_interpolation.py
```

## Dependencies
Install from requirements:

```bash
.venv/bin/python -m pip install -r requirements.txt
```
