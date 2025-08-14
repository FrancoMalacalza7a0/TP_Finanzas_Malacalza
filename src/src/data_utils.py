# data_utils.py
# Utilidades de datos: descarga, limpieza y métricas básicas de riesgo.

from __future__ import annotations
import warnings
from typing import Iterable, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import jarque_bera as sp_jarque_bera

warnings.filterwarnings("ignore")


# ----------------------------- Descarga & Limpieza ----------------------------

def download_prices(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    auto_adjust: bool = True,
    progress: bool = False,
    how: str = "inner",
) -> pd.DataFrame:
    """
    Descarga precios con yfinance y devuelve un DataFrame de CIERRE.
    - Une por 'how': 'inner' = solo fechas comunes (recomendado).
    """
    tickers = list(tickers)
    panel = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=progress,
        group_by="column",
        threads=True,
    )
    if panel.empty:
        raise ValueError("No se pudieron descargar datos con yfinance.")

    # Manejar MultiIndex vs columnas simples
    if isinstance(panel.columns, pd.MultiIndex):
        if ("Adj Close" in panel.columns.get_level_values(0)) and auto_adjust:
            px = panel["Adj Close"].copy()
        else:
            px = panel["Close"].copy()
    else:
        # Un solo ticker: devolver como DataFrame
        px = panel[["Close"]].copy()
        px.columns = [tickers[0]]

    # Unir por fechas (inner por defecto)
    px = px.sort_index().tz_localize(None)
    if how == "inner":
        px = px.dropna(how="any")
    elif how == "outer":
        px = px
    else:
        raise ValueError("Parámetro 'how' debe ser 'inner' u 'outer'.")

    return px


def fill_small_gaps(df: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """
    Rellena con forward-fill huecos de hasta 'max_gap' días; gaps mayores quedan NaN.
    Útil cuando faltan pocos días en alguna serie.
    """
    df = df.copy()
    gaps = df.isna().astype(int).groupby((~df.isna()).cumsum()).cumsum()
    df[gaps <= max_gap] = df.ffill()[gaps <= max_gap]
    return df


# --------------------------------- Retornos -----------------------------------

def calc_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """
    Calcula retornos diarios; logarítmicos si log=True.
    """
    if log:
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    return rets.dropna(how="all")


# ------------------------------- Métricas riesgo ------------------------------

def var_cvar(
    series: pd.Series, alpha: float = 0.95
) -> Tuple[float, float]:
    """
    VaR y CVaR (Expected Shortfall) históricos a 1 día, confianza 'alpha'.
    Devuelve números NEGATIVOS (pérdida), ej: -0.048 = -4.8%.
    """
    s = series.dropna()
    if s.empty:
        return np.nan, np.nan
    losses = -s.values  # pérdidas positivas
    var = np.quantile(losses, alpha)
    cvar = losses[losses >= var].mean() if np.isfinite(var) else np.nan
    return -var, -cvar  # signo negativo para consistencia


def max_drawdown(price_series: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Máximo drawdown sobre precios (no retornos).
    Retorna: drawdown_min (negativo), fecha pico, fecha valle.
    """
    p = price_series.dropna()
    if p.empty:
        return np.nan, pd.NaT, pd.NaT
    cummax = p.cummax()
    dd = p / cummax - 1.0
    end = dd.idxmin()
    start = p.loc[:end].idxmax()
    return float(dd.loc[end]), start, end


def jarque_bera(series: pd.Series) -> Tuple[float, float]:
    """
    Test de normalidad Jarque–Bera. Devuelve (estadístico, p-valor).
    """
    s = series.dropna()
    if s.empty:
        return np.nan, np.nan
    stat, p = sp_jarque_bera(s)
    return float(stat), float(p)
