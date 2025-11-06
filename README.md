# LATAM — Inflación ↔ PIB usando Random Forest

Proyecto reproducible con datos del Banco Mundial.

Se crean features (rezagos y diferencias), se valida temporalmente y se evalúa con hold-out.


## Países incluidos

ARG, BRA, CHL, COL, MEX, PER

## Resultados (RMSE/R² en hold-out)

- **COL** → RMSE=5.166 | R²=0.296
- **MEX** → RMSE=2.644 | R²=0.722
- **BRA** → RMSE=1.823 | R²=0.582
- **CHL** → RMSE=4.736 | R²=0.279
- **PER** → RMSE=6.476 | R²=0.310
- **ARG** → RMSE=3.515 | R²=0.742

## Visualizaciones

### ARG — Series de tiempo: Inflación y PIB

![Series de tiempo: Inflación y PIB](reports/figures/series_ARG.png)

### ARG — Relación Inflación ↔ PIB

![Relación Inflación ↔ PIB](reports/figures/scatter_ARG.png)

### ARG — Hold-out: Real vs Predicho

![Hold-out: Real vs Predicho](reports/figures/real_vs_predicho_ARG_random_forest.png)

### ARG — RMSE (hold-out)

![RMSE (hold-out)](reports/figures/rmse_ARG.png)

### BRA — Series de tiempo: Inflación y PIB

![Series de tiempo: Inflación y PIB](reports/figures/series_BRA.png)

### BRA — Relación Inflación ↔ PIB

![Relación Inflación ↔ PIB](reports/figures/scatter_BRA.png)

### BRA — Hold-out: Real vs Predicho

![Hold-out: Real vs Predicho](reports/figures/real_vs_predicho_BRA_random_forest.png)

### BRA — RMSE (hold-out)

![RMSE (hold-out)](reports/figures/rmse_BRA.png)

### CHL — Series de tiempo: Inflación y PIB

![Series de tiempo: Inflación y PIB](reports/figures/series_CHL.png)

### CHL — Relación Inflación ↔ PIB

![Relación Inflación ↔ PIB](reports/figures/scatter_CHL.png)

### CHL — Hold-out: Real vs Predicho

![Hold-out: Real vs Predicho](reports/figures/real_vs_predicho_CHL_random_forest.png)

### CHL — RMSE (hold-out)

![RMSE (hold-out)](reports/figures/rmse_CHL.png)

### COL — Series de tiempo: Inflación y PIB

![Series de tiempo: Inflación y PIB](reports/figures/series_COL.png)

### COL — Relación Inflación ↔ PIB

![Relación Inflación ↔ PIB](reports/figures/scatter_COL.png)

### COL — Hold-out: Real vs Predicho

![Hold-out: Real vs Predicho](reports/figures/real_vs_predicho_COL_random_forest.png)

### COL — RMSE (hold-out)

![RMSE (hold-out)](reports/figures/rmse_COL.png)

### MEX — Series de tiempo: Inflación y PIB

![Series de tiempo: Inflación y PIB](reports/figures/series_MEX.png)

### MEX — Relación Inflación ↔ PIB

![Relación Inflación ↔ PIB](reports/figures/scatter_MEX.png)

### MEX — Hold-out: Real vs Predicho

![Hold-out: Real vs Predicho](reports/figures/real_vs_predicho_MEX_random_forest.png)

### MEX — RMSE (hold-out)

![RMSE (hold-out)](reports/figures/rmse_MEX.png)

### PER — Series de tiempo: Inflación y PIB

![Series de tiempo: Inflación y PIB](reports/figures/series_PER.png)

### PER — Relación Inflación ↔ PIB

![Relación Inflación ↔ PIB](reports/figures/scatter_PER.png)

### PER — Hold-out: Real vs Predicho

![Hold-out: Real vs Predicho](reports/figures/real_vs_predicho_PER_random_forest.png)

### PER — RMSE (hold-out)

![RMSE (hold-out)](reports/figures/rmse_PER.png)


## Cómo reproducir

```bash
pip install -r requirements.txt
python main.py fetch --countries COL MEX BRA CHL PER ARG --start_year 1991
python main.py bench --countries COL MEX BRA CHL PER ARG
python main.py report
```