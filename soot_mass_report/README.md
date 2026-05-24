# Soot Mass Report Bundle

Everything Overleaf needs to compile the report, plus the raw split data used by every model in it.

## Layout

```
soot_mass_report/
├── main.tex                # the LaTeX report (compile with pdflatex)
├── README.md               # this file
├── data/                   # all CSV inputs (committable, ~50 KB total)
│   ├── full_dataset.csv               # 379 aligned rows after t > 17500s cutoff
│   ├── train_resistance_only.csv      # 189 rows, X = [Resistance]
│   ├── test_resistance_only.csv       # 190 rows, X = [Resistance]
│   ├── train_resistance_temp.csv      # 189 rows, X = [Resistance, Temp-DC]
│   ├── test_resistance_temp.csv       # 190 rows, X = [Resistance, Temp-DC]
│   ├── ablation_metrics_long.csv      # per-model MAE/RMSE/R^2, long form
│   ├── r2_wide.csv                    # R^2 side-by-side + delta
│   ├── mae_wide.csv                   # MAE side-by-side
│   ├── rmse_wide.csv                  # RMSE side-by-side
│   ├── correlations.csv               # pairwise Pearson + Spearman (+ p-values)
│   ├── correlation_matrix.csv         # 3x3 Pearson matrix
│   └── interpretability.txt           # RF/GB feature importance, scaled Ridge coefs
└── figures/                # PNGs referenced by main.tex
    ├── timeseries.png
    ├── correlations.png
    ├── correlation_heatmap.png
    ├── ablation_r2.png
    ├── best_pred_resistance_only.png
    ├── best_pred_resistance_temp.png
    ├── feature_importance.png
    └── partial_dependence.png
```

## Regenerating everything

```bash
.venv/bin/python dc_soot_mass_report.py
```

The script reads `12-02-2025 raw sensor data.xlsx` from the repo root and writes the entire `data/` + `figures/` content here. Random seed `0`, deterministic output.

## Uploading to Overleaf

The simplest path is to zip this folder and upload it as a new Overleaf project — Overleaf will compile `main.tex` directly with all relative paths intact (`figures/...` and `\graphicspath{{figures/}}`).
