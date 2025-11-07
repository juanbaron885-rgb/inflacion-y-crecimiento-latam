LATAM: Inflación ↔ PIB con Random Forest (Economía + ML)

Autor: Juan Baron
Fecha: Noviembre 2025

Propósito del Proyecto

El propósito de este proyecto es predecir el crecimiento del Producto Interno Bruto real (PIB %) de los países latinoamericanos utilizando información histórica de inflación y crecimiento del PIB, combinando el análisis económico tradicional con herramientas modernas de Machine Learning.

Para lograrlo, se emplea un modelo Random Forest Regressor, entrenado y validado con datos anuales del Banco Mundial (1991–2024). El modelo busca capturar patrones temporales y no lineales entre la inflación y el crecimiento económico, ofreciendo una herramienta predictiva para análisis macroeconómicos en la región.

Metodología y Flujo de Trabajo
1. Obtención de datos (API del Banco Mundial)

Se descargaron los indicadores:

Inflation, consumer prices (annual %) → FP.CPI.TOTL.ZG

GDP growth (annual %) → NY.GDP.MKTP.KD.ZG

Países incluidos: COL, BRA, CHL, MEX, ARG, PER

El proceso maneja paginación, limpieza y formateo de datos con pandas.

2. Limpieza y construcción de features

Se rellenaron valores faltantes por país (ffill/bfill).

Se generaron lags (rezagos) y diferencias anuales para capturar dinámica temporal:

inflation_lag1, inflation_lag2, inflation_lag3

gdp_growth_lag1, gdp_growth_lag2, gdp_growth_lag3

dinflation, dgdp

3. Entrenamiento del modelo

Se utilizó Random Forest Regressor con:

300 árboles, random_state=42

Validación cruzada temporal (TimeSeriesSplit, 5 folds)

Evaluación hold-out (15% final)

Métricas obtenidas:

RMSE (Root Mean Squared Error)

R² (Coeficiente de determinación)

4. Visualizaciones

El proyecto genera automáticamente:

Series de tiempo Inflación vs PIB

Dispersión Inflación ↔ Crecimiento PIB

Real vs Predicho (hold-out)

RMSE por país

Además, incluye un Dashboard Interactivo con dos vistas principales:

Mapa interactivo LATAM: evolución anual de la inflación

PIB vs Inflación (Todos los países): animación comparativa 1991–2024

También contiene un selector por país que muestra las 4 gráficas clave.

Estructura del Proyecto
-data/

-raw/               # datos crudos descargados

-processed/         # datos limpios y features (CSV)

-reports/

-figures/           # figuras generadas para README

- docs/

-index.html         # dashboard principal (GitHub Pages)

-mapa.html          # mapa interactivo LATAM

- pib_vs_inflacion.html

-insights/          # páginas por país (4 gráficas)

-figures/           # imágenes públicas para Pages

-.nojekyll

-main.py                # script principal con CLI (fetch, bench, report, report_html)

- README.md              # este documento

Comandos Principales
# 1. Descargar y procesar datos del Banco Mundial
python main.py fetch --countries COL MEX BRA CHL PER ARG --start_year 1991

# 2. Entrenar y evaluar el modelo Random Forest
python main.py bench --countries COL MEX BRA CHL PER ARG

# 3. Generar figuras estáticas y README visual
python main.py report

# 4. Crear dashboard interactivo (Plotly + HTML para GitHub Pages)
python main.py report_html

Resultados Resumidos
País	RMSE (Hold-Out)	R² (Hold-Out)
ARG
BRA	
CHL	
COL	
MEX	
PER	

(Los valores reales se generan automáticamente en benchmark_metrics.csv.)

Dashboard Interactivo en Línea

GitHub Pages:
https://juanbaron885-rgb.github.io/inflacion-y-crecimiento-latam/

Mapa LATAM: https://juanbaron885-rgb.github.io/inflacion-y-crecimiento-latam/mapa.html

PIB vs Inflación (todos): https://juanbaron885-rgb.github.io/inflacion-y-crecimiento-latam/pib_vs_inflacion.html

Análisis por país (4 gráficas): https://juanbaron885-rgb.github.io/inflacion-y-crecimiento-latam/insights/

Conclusiones

Se logró automatizar todo el pipeline económico, desde la descarga de datos hasta la predicción y visualización.

El modelo Random Forest muestra una capacidad sólida de generalización para series anuales pequeñas.

La integración con Plotly y GitHub Pages permite comunicar resultados de forma interactiva y transparente.

Este trabajo demuestra el potencial del Machine Learning aplicado a la economía regional, facilitando el análisis de políticas macroeconómicas basadas en evidencia.