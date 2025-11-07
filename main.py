# ============================================================================
# Proyecto Final (Economía + ML) — LATAM: Inflación ↔ PIB con Random Forest
# Autor: Juan Baron
# Fecha: Nov-2025
# ----------------------------------------------------------------------------
# PROPÓSITO
# Predecir el crecimiento del PIB real (GDP growth, %) usando información
# histórica del propio PIB y la inflación, con datos abiertos del Banco Mundial.
# Se aplica un método de Machine Learning: Random Forest Regressor.
# ============================================================================
# Este archivo puede ejecutarse con los comandos:
# python main.py fetch  -> descarga datos y crea features
# python main.py bench  -> entrena y evalúa el modelo
# python main.py report -> genera visualizaciones y README.md para GitHub
# ============================================================================


# ============================================================
# Importación de librerías necesarias
# ============================================================
import argparse  # permite crear comandos desde consola
import os        # manejo de rutas y archivos en el sistema
import time      # control de tiempo y pausas en los bucles
from dataclasses import dataclass  # define configuraciones estructuradas
from typing import List, Dict, Any, Tuple, Optional  # tipado de variables y funciones

# Librerías de análisis y visualización
import requests           # solicitudes HTTP para consultar la API del Banco Mundial
import numpy as np        # cálculos numéricos y matrices
import pandas as pd       # manipulación y limpieza de datos
import matplotlib.pyplot as plt  # creación de gráficos

# Librerías de Machine Learning (scikit-learn)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score  # validación cruzada temporal
from sklearn.metrics import mean_squared_error, r2_score               # métricas de evaluación
from sklearn.pipeline import Pipeline                                 # flujo de pasos del modelo
from sklearn.ensemble import RandomForestRegressor                    # modelo Random Forest
import joblib                                                         # para guardar el modelo entrenado


# ============================================================
# Configuración de rutas del proyecto (estructura de carpetas)
# ============================================================
BASE_DIR = "."                                      # carpeta base del proyecto
DATA_DIR = os.path.join(BASE_DIR, "data")           # carpeta para datos
RAW_DIR  = os.path.join(DATA_DIR, "raw")            # datos crudos descargados
PROC_DIR = os.path.join(DATA_DIR, "processed")      # datos procesados
REP_DIR  = os.path.join(BASE_DIR, "reports")        # reportes y resultados
FIG_DIR  = os.path.join(REP_DIR, "figures")         # carpeta de figuras gráficas

def ensure_dirs() -> None:
    """Crea toda la estructura de carpetas necesarias para ejecutar el proyecto."""
    for d in (DATA_DIR, RAW_DIR, PROC_DIR, REP_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)  # crea cada directorio si no existe


# ============================================================
# Configuración general del dataset (países e indicadores)
# ============================================================
# Lista explícita de países a analizar (códigos ISO3)
COUNTRIES: List[str] = ["COL", "MEX", "BRA", "CHL", "PER", "ARG"]

# Diccionario de indicadores del Banco Mundial (clave -> código API)
# Contiene inflación y crecimiento del PIB real.
INDICATORS: Dict[str, Tuple[str, str]] = {
    "inflation":  ("FP.CPI.TOTL.ZG",  "Inflation, consumer prices (annual %)"),
    "gdp_growth": ("NY.GDP.MKTP.KD.ZG", "GDP growth (annual %)")
}

# URL base del endpoint del Banco Mundial, usando formato JSON v2
WB_URL = "https://api.worldbank.org/v2/country/{iso}/indicator/{series}?format=json&per_page=1000&page={page}"


# ============================================================================
# CLASE 1: Cliente de la API del Banco Mundial
# Descarga datos de inflación y PIB usando la API, manejando paginación y reintentos.
# ============================================================================
class WorldBankClient:
    def __init__(self, iso3: str, max_retries: int = 3, timeout: int = 30):
        # Inicializa el cliente con el código del país (ISO3)
        self.iso3 = iso3.upper()        # país (ejemplo: "COL")
        self.max_retries = max_retries  # número máximo de reintentos por fallo
        self.timeout = timeout          # tiempo máximo de espera por solicitud (segundos)

    def _get_json(self, url: str) -> List[Any]:
        """Hace una llamada GET a la API del Banco Mundial con manejo de errores."""
        for i in range(1, self.max_retries + 1):  # reintenta varias veces
            try:
                r = requests.get(url, timeout=self.timeout)  # realiza la solicitud
                r.raise_for_status()                         # lanza error si falla
                js = r.json()                                # convierte la respuesta a JSON
                if not isinstance(js, list) or len(js) < 2:  # valida estructura
                    raise RuntimeError("Estructura inesperada de la API del Banco Mundial.")
                return js
            except Exception:
                if i == self.max_retries:
                    raise  # si falla el último intento, lanza el error
                time.sleep(0.6 * i)  # espera un poco antes de reintentar

    def fetch_indicator(self, series_code: str) -> List[Dict[str, Any]]:
        """Descarga todas las páginas del indicador solicitado para el país actual."""
        rows, page = [], 1
        while True:
            url = WB_URL.format(iso=self.iso3.lower(), series=series_code, page=page)
            meta, items = self._get_json(url)  # obtiene metadatos y registros
            rows.extend(items or [])           # añade datos obtenidos
            # manejo del paginado de la API
            m0 = meta[0] if isinstance(meta, list) and meta else {}
            page_num = int(m0.get("page", 0) or 0)
            per_page = int(m0.get("per_page", 0) or 0)
            total = int(m0.get("total", 0) or 0)
            if page_num * per_page >= total:  # si se descargaron todas las páginas
                break
            page += 1
            time.sleep(0.15)  # pausa para evitar límite de peticiones
        return rows


# ============================================================================
# CLASE 2: Dataset LATAM (descarga + limpieza)
# Combina los datos de inflación y PIB de varios países en un solo DataFrame.
# ============================================================================
class LatinMacroDataset:
    def __init__(self, countries: List[str]):
        # Inicializa con lista de países (en mayúsculas)
        self.countries = [c.upper() for c in countries]

    def fetch_all(self) -> pd.DataFrame:
        """Descarga ambos indicadores (inflación y PIB) y los une por país y año."""
        frames: List[pd.DataFrame] = []  # lista para almacenar cada país
        for iso in self.countries:       # itera sobre cada país
            client = WorldBankClient(iso)
            per_indicator: List[pd.DataFrame] = []
            for key, (code, _title) in INDICATORS.items():  # para cada indicador
                rows = client.fetch_indicator(code)
                data = []
                for d in rows:
                    iso3 = d.get("countryiso3code")
                    y = d.get("date")
                    v = d.get("value")
                    if iso3 != iso or y is None:
                        continue
                    try:
                        year = int(y)
                    except:
                        continue
                    try:
                        val = float(v) if v is not None else None
                    except:
                        val = None
                    data.append({"country": iso, "year": year, key: val})
                # crea un DataFrame para el indicador actual
                df = pd.DataFrame(data).drop_duplicates(subset=["country", "year"]).sort_values(["country", "year"])
                per_indicator.append(df)
            # une inflación y PIB por país
            out = per_indicator[0]
            for k in range(1, len(per_indicator)):
                out = out.merge(per_indicator[k], on=["country", "year"], how="outer")
            frames.append(out)
        # concatena todos los países en un solo DataFrame
        big = pd.concat(frames, ignore_index=True).sort_values(["country", "year"]).reset_index(drop=True)
        return big

    def clean(self, df: pd.DataFrame, start_year: Optional[int] = None, end_year: Optional[int] = None) -> pd.DataFrame:
        """Limpia los datos: recorta rango de años y completa valores faltantes."""
        if start_year is not None:
            df = df[df["year"] >= start_year]
        if end_year is not None:
            df = df[df["year"] <= end_year]
        df = df.sort_values(["country", "year"]).copy()
        # Rellena huecos por país (ffill/bfill)
        df["inflation"]  = df.groupby("country")["inflation"].transform(lambda s: s.ffill().bfill())
        df["gdp_growth"] = df.groupby("country")["gdp_growth"].transform(lambda s: s.ffill().bfill())
        return df.reset_index(drop=True)
# ============================================================================
# Clase 3: FeatureBuilder (creación de variables)
# ============================================================================
class FeatureBuilder:
    def __init__(self, lags: int = 3):
        self.lags = lags  # número de rezagos a crear

    def make_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea rezagos y diferencias por país y elimina filas con NaN."""
        df = df.copy()
        for k in range(1, self.lags + 1):
            df[f"inflation_lag{k}"]  = df.groupby("country")["inflation"].shift(k)
            df[f"gdp_growth_lag{k}"] = df.groupby("country")["gdp_growth"].shift(k)
        df["dinflation"] = df.groupby("country")["inflation"].diff()
        df["dgdp"]       = df.groupby("country")["gdp_growth"].diff()
        df = df.dropna().reset_index(drop=True)
        return df

# ============================================================================
# Clase 4: Entrenamiento con Random Forest (único método de ML)
# ============================================================================
@dataclass
class TrainerConfig:
    scoring: str = "neg_root_mean_squared_error"  # métrica para CV temporal
    n_splits: int = 5                              # número de folds en TimeSeriesSplit
    holdout_ratio: float = 0.15                    # tamaño del conjunto de prueba final

class RandomForestTrainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg  # almacena configuración

    def _split_country(self, df_feat: pd.DataFrame, country: str, target: str = "gdp_growth"):
        """Partición temporal por país: devuelve X_train, y_train, X_test, y_test, df_filtrado."""
        d = df_feat[df_feat["country"] == country].sort_values("year").reset_index(drop=True)
        X = d.drop(columns=[target, "year", "country"], errors="ignore")
        y = d[target]
        cutoff = int(len(d) * (1.0 - self.cfg.holdout_ratio))  # índice de corte temporal
        if cutoff <= 1:
            raise ValueError(f"No hay suficientes datos para {country}")
        return X.iloc[:cutoff], y.iloc[:cutoff], X.iloc[cutoff:], y.iloc[cutoff:], d

    def cross_validate(self, X: pd.DataFrame, y: pd.Series):
        """Aplica validación cruzada temporal y retorna media y std del scoring."""
        model = Pipeline([("model", RandomForestRegressor(n_estimators=300, random_state=42))])
        cv = TimeSeriesSplit(n_splits=self.cfg.n_splits)
        scores = cross_val_score(model, X, y, scoring=self.cfg.scoring, cv=cv, n_jobs=-1)
        return float(np.mean(scores)), float(np.std(scores))

    def fit_eval_country(self, df_feat: pd.DataFrame, country: str) -> Dict[str, Any]:
        """Entrena RF por país, evalúa con hold-out, guarda figuras y el modelo."""
        X_tr, y_tr, X_te, y_te, _ = self._split_country(df_feat, country)
        cv_mean, cv_std = self.cross_validate(X_tr, y_tr)  # CV temporal

        pipe = Pipeline([("model", RandomForestRegressor(n_estimators=300, random_state=42))])
        pipe.fit(X_tr, y_tr)  # ajuste final en train
        y_pred = pipe.predict(X_te)  # predicciones en hold-out

        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))  # RMSE hold-out
        r2   = float(r2_score(y_te, y_pred))                     # R² hold-out

        joblib.dump(pipe, os.path.join(PROC_DIR, f"model_random_forest_{country}.joblib"))  # guarda modelo

        # Figura: Real vs Predicho en hold-out
        os.makedirs(FIG_DIR, exist_ok=True)
        plt.figure()
        plt.scatter(y_te, y_pred, alpha=0.8)
        plt.xlabel("Real (gdp_growth)")
        plt.ylabel("Predicho")
        plt.title(f"{country} — RandomForest — Real vs Predicho (hold-out)")
        fig_rvp = os.path.join(FIG_DIR, f"real_vs_predicho_{country}_random_forest.png")
        plt.savefig(fig_rvp, bbox_inches="tight"); plt.close()

        # Figura: Importancia de variables (si aplica)
        rf = pipe.named_steps["model"]
        if hasattr(rf, "feature_importances_"):
            importances = rf.feature_importances_
            idx = np.argsort(importances)[::-1]
            names = X_tr.columns[idx]
            plt.figure()
            plt.bar(range(len(idx)), importances[idx])
            plt.xticks(range(len(idx)), names, rotation=90)
            plt.title(f"{country}: Importancia de variables (Random Forest)")
            plt.tight_layout()
            fig_imp = os.path.join(FIG_DIR, f"importances_{country}.png")
            plt.savefig(fig_imp, bbox_inches="tight"); plt.close()
        else:
            fig_imp = ""

        return {
            "country": country,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "rmse_test": rmse,
            "r2_test": r2,
            "figure_rvp": fig_rvp,
            "figure_imp": fig_imp
        }

# ============================================================================
# Visualizaciones para el README (series, dispersión, barras)
# ============================================================================
def plot_series(df_clean: pd.DataFrame, countries: List[str]) -> None:
    """Grafica series temporales de inflación y PIB por país y guarda PNG."""
    for c in countries:
        d = df_clean[df_clean["country"] == c].sort_values("year")
        if d.empty: 
            continue
        plt.figure()
        plt.plot(d["year"], d["inflation"], label="Inflación (%)")
        plt.plot(d["year"], d["gdp_growth"], label="PIB real (%)")
        plt.title(f"{c}: Inflación y PIB (anual)")
        plt.xlabel("Año"); plt.ylabel("%"); plt.legend(); plt.grid(True)
        out = os.path.join(FIG_DIR, f"series_{c}.png")
        plt.savefig(out, bbox_inches="tight"); plt.close()

def plot_scatter(df_clean: pd.DataFrame, countries: List[str]) -> None:
    """Grafica dispersión Inflación vs PIB por país y guarda PNG."""
    for c in countries:
        d = df_clean[df_clean["country"] == c].dropna(subset=["inflation", "gdp_growth"])
        if d.empty:
            continue
        plt.figure()
        plt.scatter(d["inflation"], d["gdp_growth"], alpha=0.7)
        plt.title(f"{c}: Inflación ↔ Crecimiento PIB")
        plt.xlabel("Inflación (% anual)"); plt.ylabel("PIB real (% anual)")
        plt.grid(True)
        out = os.path.join(FIG_DIR, f"scatter_{c}.png")
        plt.savefig(out, bbox_inches="tight"); plt.close()

def plot_rmse_bars(bench: pd.DataFrame) -> None:
    """Grafica barra de RMSE por país (a partir del benchmark)."""
    for c in bench["country"].unique():
        b = bench[bench["country"] == c]
        if b.empty: 
            continue
        plt.figure()
        plt.bar(["Random Forest"], [b["rmse_test"].values[0]])
        plt.title(f"{c}: RMSE (hold-out)")
        plt.ylabel("RMSE"); plt.grid(axis="y")
        out = os.path.join(FIG_DIR, f"rmse_{c}.png")
        plt.savefig(out, bbox_inches="tight"); plt.close()
# ============================================================================
# Comandos CLI
# - fetch : datos crudos → limpios → features (CSV) # - bench : entrena+evalúa Random Forest por país, guarda métricas (CSV) 
# - report: genera todas las figuras + README.md visual en la raíz
# ============================================================================
def cmd_fetch(args) -> None:
    """Descarga, limpia y construye features; guarda CSVs en processed/."""
    ensure_dirs()
    ds = LatinMacroDataset(args.countries)         # instancia dataset multi-país
    df = ds.fetch_all()                            # descarga inflación y PIB
    df = ds.clean(df, start_year=args.start_year, end_year=args.end_year)  # limpieza
    clean_path = os.path.join(PROC_DIR, "latam_macro_clean.csv")
    df.to_csv(clean_path, index=False)             # guarda datos limpios
    fb = FeatureBuilder(lags=3)                    # builder de características
    df_feat = fb.make_features(df)                 # crea lags y diferencias
    feat_path = os.path.join(PROC_DIR, "latam_macro_features.csv")
    df_feat.to_csv(feat_path, index=False)         # guarda features
    print(f"[OK] Guardado: {clean_path}")
    print(f"[OK] Features: {feat_path}")

def cmd_bench(args) -> None:
    """Entrena y evalúa Random Forest por país; exporta benchmark_metrics.csv."""
    ensure_dirs()
    feat_path = os.path.join(PROC_DIR, "latam_macro_features.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError("No se encontró latam_macro_features.csv. Ejecuta 'fetch' primero.")
    df_feat = pd.read_csv(feat_path)               # carga features
    rows = []                                      # acumula métricas por país
    trainer = RandomForestTrainer(TrainerConfig(scoring="neg_root_mean_squared_error",
                                                n_splits=5, holdout_ratio=0.15))
    for c in args.countries:
        print(f"[{c}] Entrenando y evaluando Random Forest...")
        r = trainer.fit_eval_country(df_feat, c)   # entrena y evalúa
        rows.append(r)
    bench = pd.DataFrame(rows)
    out_path = os.path.join(PROC_DIR, "benchmark_metrics.csv")
    bench.to_csv(out_path, index=False)            # guarda benchmark
    print(f"[OK] Benchmark guardado en {out_path}")

def cmd_report(args) -> None:
    """Genera visualizaciones y README.md con resultados y figuras embebidas."""
    ensure_dirs()
    clean_path = os.path.join(PROC_DIR, "latam_macro_clean.csv")
    feat_path  = os.path.join(PROC_DIR, "latam_macro_features.csv")
    bench_path = os.path.join(PROC_DIR, "benchmark_metrics.csv")
    if not (os.path.exists(clean_path) and os.path.exists(feat_path) and os.path.exists(bench_path)):
        raise FileNotFoundError("Faltan archivos. Ejecuta 'fetch' y 'bench' antes de 'report'.")

    df_clean = pd.read_csv(clean_path)             # datos limpios (para series/scatter)
    bench    = pd.read_csv(bench_path)             # resultados de benchmark
    countries = sorted(df_clean["country"].unique().tolist())

    # Genera figuras para README
    plot_series(df_clean, countries)
    plot_scatter(df_clean, countries)
    plot_rmse_bars(bench)

    # Construye README.md con imágenes y resumen
    readme_lines: List[str] = []
    readme_lines.append("# LATAM — Inflación ↔ PIB usando Random Forest\n")
    readme_lines.append("Proyecto reproducible con datos del Banco Mundial.\n")
    readme_lines.append("Se crean features (rezagos y diferencias), se valida temporalmente y se evalúa con hold-out.\n\n")
    readme_lines.append("## Países incluidos\n")
    readme_lines.append(", ".join(countries) + "\n")
    readme_lines.append("## Resultados (RMSE/R² en hold-out)\n")
    for _, r in bench.iterrows():
        readme_lines.append(f"- **{r['country']}** → RMSE={r['rmse_test']:.3f} | R²={r['r2_test']:.3f}")
    readme_lines.append("\n## Visualizaciones\n")
    for c in countries:
        for tag, title in [
            (f"series_{c}.png", "Series de tiempo: Inflación y PIB"),
            (f"scatter_{c}.png", "Relación Inflación ↔ PIB"),
            (f"real_vs_predicho_{c}_random_forest.png", "Hold-out: Real vs Predicho"),
            (f"rmse_{c}.png", "RMSE (hold-out)")
        ]:
            fig_path = os.path.join("reports", "figures", tag)
            if os.path.exists(fig_path):
                readme_lines.append(f"### {c} — {title}\n")
                readme_lines.append(f"![{title}]({fig_path})\n")
    readme_lines.append("\n## Cómo reproducir\n")
    readme_lines.append("```bash\npip install -r requirements.txt\n"
                        "python main.py fetch --countries COL MEX BRA CHL PER ARG --start_year 1991\n"
                        "python main.py bench --countries COL MEX BRA CHL PER ARG\n"
                        "python main.py report\n```")
    with open("README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines))
    print("[OK] README.md visual generado en la raíz.")

# ============================================================================
# Parser CLI y ejecución principal
# ============================================================================
def build_parser():
    """Define subcomandos y argumentos de línea de comandos."""
    p = argparse.ArgumentParser(description="LATAM Inflación ↔ PIB (WB API) + Random Forest + Reportes")
    sub = p.add_subparsers(dest="cmd", required=True)

    # subcomando: fetch
    f = sub.add_parser("fetch", help="Descargar/limpiar y construir features")
    f.add_argument("--countries", nargs="+", default=COUNTRIES)  # lista de países
    f.add_argument("--start_year", type=int, default=1991)       # año inicial
    f.add_argument("--end_year", type=int, default=None)         # año final (None = último)
    f.set_defaults(func=cmd_fetch)

    # subcomando: bench
    b = sub.add_parser("bench", help="Entrenar/Evaluar SOLO Random Forest por país")
    b.add_argument("--countries", nargs="+", default=COUNTRIES)
    b.set_defaults(func=cmd_bench)

    # subcomando: report
    r = sub.add_parser("report", help="Generar gráficas y README visual para GitHub")
    r.set_defaults(func=cmd_report)
    return p

def main():
    """Punto de entrada: parsea argumentos y ejecuta el subcomando seleccionado."""
    ensure_dirs()               # garantiza estructura de carpetas
    parser = build_parser()     # construye parser
    args = parser.parse_args()  # parsea argumentos
    args.func(args)             # ejecuta la función asociada

if __name__ == "__main__":
    main()
# ====================== USO DE LOS COMANDOS ======================
# !python main.py fetch  → descarga y prepara datos del Banco Mundial.
# !python main.py bench  → entrena y evalúa el modelo Random Forest.
# !python main.py report → genera gráficas y crea el README con resultados.
# ================================================================
# ============================================================================
# NUEVO COMANDO: report_html → genera dashboard interactivo (Plotly + HTML)
# ============================================================================

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def cmd_report_html(args):
    """Genera un reporte interactivo HTML con Plotly (para GitHub Pages)."""
    ensure_dirs()
    import pandas as pd, os

    clean_path = os.path.join(PROC_DIR, "latam_macro_clean.csv")
    bench_path = os.path.join(PROC_DIR, "benchmark_metrics.csv")

    df = pd.read_csv(clean_path)
    bench = pd.read_csv(bench_path)

    # Gráfico interactivo principal: Inflación ↔ PIB animado por año
    df = df.dropna(subset=["inflation", "gdp_growth"])
    df["inflation_pct"] = df["inflation"].astype(float).round(2)
    df["gdp_growth_pct"] = df["gdp_growth"].astype(float).round(2)

    fig = px.scatter(
        df,
        x="inflation_pct",
        y="gdp_growth_pct",
        animation_frame="year",
        animation_group="country",
        color="country",
        hover_name="country",
        size_max=30,
        title="Inflación vs Crecimiento del PIB (LATAM, animado por año)",
        labels={
            "inflation_pct": "Inflación (%)",
            "gdp_growth_pct": "Crecimiento PIB (%)"
        },
    )

    fig.update_traces(marker=dict(sizemode="area", sizeref=0.5))
    fig.update_layout(width=900, height=600, template="plotly_white")

    os.makedirs("docs", exist_ok=True)
    fig.write_html("docs/index.html", include_plotlyjs="cdn")
    print("[OK] Interactivo listo en docs/index.html (GitHub Pages).")

# Añadir el nuevo subcomando al parser
def build_parser_with_html():
    parser = build_parser()
    r_html = parser.add_subparser = parser.add_subparsers(dest="cmd_html", required=False)
    parser.add_argument("--html", action="store_true", help="Generar dashboard HTML interactivo")
    return parser

# Sobrescribir main para incluirlo
if __name__ == "__main__":
    import sys
    ensure_dirs()
    if len(sys.argv) > 1 and sys.argv[1] == "report_html":
        cmd_report_html(None)
    else:
        parser = build_parser()
        args = parser.parse_args()
        args.func(args)
# ============================================================================
# SUBCOMANDO NUEVO: report_html → dashboard interactivo (Plotly) en docs/
# ============================================================================

def cmd_report_html(args):
    """Genera docs/index.html con un scatter animado (Inflación vs PIB por año)."""
    import os, pandas as pd
    import plotly.express as px
    from pathlib import Path

    ensure_dirs()
    clean_path = os.path.join(PROC_DIR, "latam_macro_clean.csv")
    if not os.path.exists(clean_path):
        raise FileNotFoundError("Falta data procesada. Corre antes: 'fetch' y 'bench'.")

    df = pd.read_csv(clean_path).dropna(subset=["inflation","gdp_growth"]).copy()
    # columnas en % (texto) para ejes/hover
    df["inflation_pct"]  = df["inflation"].astype(float).round(2)
    df["gdp_growth_pct"] = df["gdp_growth"].astype(float).round(2)

    fig = px.scatter(
        df, x="inflation_pct", y="gdp_growth_pct",
        animation_frame="year", animation_group="country",
        color="country", hover_name="country", size_max=28,
        labels={"inflation_pct":"Inflación (%)", "gdp_growth_pct":"Crecimiento PIB (%)"},
        title="LATAM: Inflación vs Crecimiento del PIB (animado por año)"
    )
    fig.update_xaxes(ticksuffix="%"); fig.update_yaxes(ticksuffix="%")

    Path("docs").mkdir(parents=True, exist_ok=True)
    Path("docs/.nojekyll").write_text("", encoding="utf-8")  # evita que Jekyll bloquee recursos
    fig.write_html("docs/index.html", include_plotlyjs="cdn")
    print("[OK] Interactivo listo en docs/index.html (GitHub Pages).")

# --- Re-defino el parser para incluir 'report_html' además de fetch/bench/report ---
def build_parser():
    import argparse
    p = argparse.ArgumentParser(description="LATAM Inflación ↔ PIB (WB API) + Random Forest + Reportes")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="Descargar/limpiar y construir features")
    f.add_argument("--countries", nargs="+", default=COUNTRIES)
    f.add_argument("--start_year", type=int, default=1991)
    f.add_argument("--end_year", type=int, default=None)
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("bench", help="Entrenar/Evaluar SOLO Random Forest por país")
    b.add_argument("--countries", nargs="+", default=COUNTRIES)
    b.set_defaults(func=cmd_bench)

    r = sub.add_parser("report", help="Generar gráficas y README visual (estático)")
    r.set_defaults(func=cmd_report)

    rh = sub.add_parser("report_html", help="Generar dashboard interactivo (Plotly) en docs/")
    rh.set_defaults(func=cmd_report_html)

    return p
# ============================================================================
# NUEVO SUBCOMANDO: report_html → genera dashboard interactivo (Plotly)
# ============================================================================

import plotly.express as px
import pandas as pd, os
from pathlib import Path

def cmd_report_html(args):
    """Genera un dashboard interactivo (HTML) con Plotly animado por año."""
    ensure_dirs()
    clean_path = os.path.join(PROC_DIR, "latam_macro_clean.csv")
    if not os.path.exists(clean_path):
        raise FileNotFoundError("Faltan datos procesados. Corre antes: fetch y bench.")
    
    # Cargar datos y limpiar
    df = pd.read_csv(clean_path).dropna(subset=["inflation", "gdp_growth"]).copy()
    df["inflation_pct"]  = df["inflation"].round(2)
    df["gdp_growth_pct"] = df["gdp_growth"].round(2)

    # Crear gráfico interactivo animado
    fig = px.scatter(
        df,
        x="inflation_pct", y="gdp_growth_pct",
        animation_frame="year", animation_group="country",
        color="country", hover_name="country",
        labels={
            "inflation_pct": "Inflación (%)",
            "gdp_growth_pct": "Crecimiento PIB (%)"
        },
        title="Inflación vs Crecimiento PIB — LATAM (Animado por Año)"
    )
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(ticksuffix="%")
    fig.update_layout(width=900, height=600, template="plotly_white")

    # Guardar en docs/
    Path("docs").mkdir(parents=True, exist_ok=True)
    Path("docs/.nojekyll").write_text("")  # evita que GitHub Pages bloquee recursos
    fig.write_html("docs/index.html", include_plotlyjs="cdn")
    print("[OK] Interactivo listo en docs/index.html (GitHub Pages).")

# Reescribimos el parser para incluir el nuevo comando
def build_parser():
    p = argparse.ArgumentParser(description="LATAM Inflación ↔ PIB + Random Forest + Reportes")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="Descargar/limpiar y construir features")
    f.add_argument("--countries", nargs="+", default=COUNTRIES)
    f.add_argument("--start_year", type=int, default=1991)
    f.add_argument("--end_year", type=int, default=None)
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("bench", help="Entrenar/Evaluar SOLO Random Forest por país")
    b.add_argument("--countries", nargs="+", default=COUNTRIES)
    b.set_defaults(func=cmd_bench)

    r = sub.add_parser("report", help="Generar gráficas y README visual (estático)")
    r.set_defaults(func=cmd_report)

    rh = sub.add_parser("report_html", help="Generar dashboard interactivo (Plotly)")
    rh.set_defaults(func=cmd_report_html)

    return p
