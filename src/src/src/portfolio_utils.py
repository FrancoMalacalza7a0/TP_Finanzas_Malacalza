# portfolio_utils.py
# Simulación Montecarlo, métricas de cartera y utilidades.

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd


# ------------------------------ Métricas de cartera ---------------------------

def portfolio_metrics(
    w: np.ndarray,
    mean_returns: pd.Series,
    cov_annual: pd.DataFrame,
    rf: float = 0.02,
    trans_cost: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Retorno esperado, volatilidad y Sharpe anualizados para pesos 'w'.
    Resta un costo de transacción 'trans_cost' (por única vez).
    """
    ret = float(np.dot(w, mean_returns)) - float(trans_cost)
    vol = float(np.sqrt(w.T @ cov_annual.values @ w))
    shp = (ret - rf) / vol if vol > 1e-9 else np.nan
    return ret, vol, shp


def pretty_weights(w: np.ndarray, tickers: list[str]) -> pd.Series:
    """Convierte pesos a Series con etiquetas."""
    return pd.Series(w, index=tickers)


# ----------------------------- Frontera (simulación) --------------------------

def efficient_frontier_sim(
    returns: pd.DataFrame,
    rf: float = 0.02,
    n_ports: int = 20000,
    w_min: float = 0.05,
    w_max: float = 0.40,
    trans_cost: float = 0.005,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    """
    Simula 'n_ports' carteras con límites de pesos (w_min/w_max) y costo inicial.
    Devuelve:
      - df_port: DataFrame con Return, Vol, Sharpe y Pesos (array)
      - p_max: dict portafolio Máx Sharpe
      - p_min: dict portafolio Mín Vol
      - p_tgt: dict portafolio más cercano a σ ≈ 28%
    """
    rng = np.random.default_rng(seed)
    tickers = list(returns.columns)
    mu = returns.mean() * 252.0
    cov = returns.cov() * 252.0
    n = len(tickers)

    out = {"Return": [], "Vol": [], "Sharpe": [], "Pesos": []}

    for _ in range(n_ports):
        w = rng.dirichlet(np.ones(n))
        # límites por activo
        w = np.clip(w, w_min, w_max)
        w = w / w.sum()
        ret, vol, shp = portfolio_metrics(w, mu, cov, rf=rf, trans_cost=trans_cost)
        out["Return"].append(ret)
        out["Vol"].append(vol)
        out["Sharpe"].append(shp)
        out["Pesos"].append(w)

    df_port = pd.DataFrame(out)
    i_max = df_port["Sharpe"].idxmax()
    i_min = df_port["Vol"].idxmin()
    i_tgt = (df_port["Vol"] - 0.28).abs().idxmin()

    def _pack(idx: int) -> Dict:
        return {
            "Return": float(df_port.at[idx, "Return"]),
            "Vol": float(df_port.at[idx, "Vol"]),
            "Sharpe": float(df_port.at[idx, "Sharpe"]),
            "Pesos": pretty_weights(df_port.at[idx, "Pesos"], tickers),
        }

    return df_port, _pack(i_max), _pack(i_min), _pack(i_tgt)
