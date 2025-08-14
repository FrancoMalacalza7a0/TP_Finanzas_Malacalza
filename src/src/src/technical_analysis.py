# technical_analysis.py
# Indicadores técnicos y backtesting simple.

from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import pandas as pd


# ------------------------------- Indicadores base -----------------------------

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.clip(0, 100)


def adx_plus_minus(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    ADX con +DI y -DI (Wilder).
    """
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx, plus_di, minus_di


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    width_pct = (upper - lower) / mid * 100.0
    return upper, mid, lower, width_pct


# --------------------------------- Backtesting --------------------------------

def _max_drawdown_from_returns(r: pd.Series) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def _calmar_ratio(r: pd.Series) -> float:
    mdd = _max_drawdown_from_returns(r)
    if mdd >= 0:
        return np.nan
    ann_ret = r.mean() * 252.0
    return float(ann_ret / abs(mdd))


def _sharpe_ratio(r: pd.Series, rf: float = 0.0) -> float:
    mu = r.mean() * 252.0
    sig = r.std() * np.sqrt(252.0)
    return float((mu - rf) / sig) if sig > 1e-9 else np.nan


def backtest_sma_cross(close: pd.Series, fast: int = 50, slow: int = 200
) -> Dict[str, object]:
    """
    Backtest LONG-only: posición 1 si SMA(fast) > SMA(slow), 0 si no.
    Devuelve: dict con series (position, equity) y métricas (Sharpe, Calmar, DD, WinRate).
    """
    df = pd.DataFrame({"Close": close}).copy()
    df[f"SMA_{fast}"] = sma(df["Close"], fast)
    df[f"SMA_{slow}"] = sma(df["Close"], slow)
    df = df.dropna()

    pos = (df[f"SMA_{fast}"] > df[f"SMA_{slow}"]).astype(int)
    ret = df["Close"].pct_change().fillna(0.0)
    strat_ret = ret * pos.shift(1).fillna(0)

    equity_bh = (1 + ret).cumprod()
    equity_st = (1 + strat_ret).cumprod()

    # métricas
    sharpe = _sharpe_ratio(strat_ret)
    calmar = _calmar_ratio(strat_ret)
    dd = _max_drawdown_from_returns(strat_ret)

    # win rate por “trade” (cambios de posición)
    chg = pos.diff().fillna(0)
    # “retornos en el trade” aproximado: sumamos desde cada entrada hasta salida
    trade_idx = chg[chg != 0].index
    wins = 0
    tot = 0
    last = None
    for t in trade_idx:
        if chg.loc[t] == 1:
            last = t
        elif chg.loc[t] == -1 and last is not None:
            tot += 1
            rsum = strat_ret.loc[last:t].sum()
            if rsum > 0:
                wins += 1
            last = None
    win_rate = wins / tot if tot > 0 else np.nan

    return {
        "position": pos,
        "bh_equity": equity_bh,
        "strat_equity": equity_st,
        "metrics": {
            "Sharpe": sharpe,
            "Calmar": calmar,
            "MaxDD": dd,
            "WinRate": win_rate,
        },
    }
