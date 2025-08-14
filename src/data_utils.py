src/data_utils.py

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

# fundamental_analysis.py
# Ratios comparativos con yfinance y un DCF simple y transparente.

from __future__ import annotations
from typing import Dict, Iterable
import numpy as np
import pandas as pd
import yfinance as yf


# --------------------------------- Helpers ------------------------------------

def _safe_div(a, b):
    try:
        if a is None or b in (None, 0):
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan


def _get_info_safe(t: yf.Ticker, key: str):
    info = getattr(t, "get_info", None) or getattr(t, "info", {})
    try:
        return info.get(key, None) if isinstance(info, dict) else None
    except Exception:
        return None


# ------------------------------ Ratios comparativos ---------------------------

def get_sector_dashboard(tickers: Iterable[str]) -> pd.DataFrame:
    """
    Construye una tabla de ratios estandarizados con yfinance (robusta a faltantes).
      EV/EBITDA, EV/Sales, Forward/Trailing P/E, PEG, P/B, P/S,
      Operating Margin, Profit Margin, ROA, ROE,
      Debt/Equity, Interest Coverage (si hay datos),
      Market Cap, 52W High/Low, 52W Change, Beta.
    """
    rows = []
    for tk in tickers:
        t = yf.Ticker(tk)

        # info rápido
        finfo = getattr(t, "fast_info", {})
        mcap = finfo.get("market_cap") or _get_info_safe(t, "marketCap")
        beta = _get_info_safe(t, "beta") or finfo.get("beta")

        # info detallada
        ev = _get_info_safe(t, "enterpriseValue")
        ebitda = _get_info_safe(t, "ebitda")
        rev = _get_info_safe(t, "totalRevenue")
        fpe = _get_info_safe(t, "forwardPE")
        tpe = _get_info_safe(t, "trailingPE")
        peg = _get_info_safe(t, "pegRatio")
        pb = _get_info_safe(t, "priceToBook")
        ps = _get_info_safe(t, "priceToSalesTrailing12Months")
        opm = _get_info_safe(t, "operatingMargins")    # proporción (0.13 = 13%)
        pm = _get_info_safe(t, "profitMargins")
        roa = _get_info_safe(t, "returnOnAssets")
        roe = _get_info_safe(t, "returnOnEquity")
        de  = _get_info_safe(t, "debtToEquity")        # %
        year_high = finfo.get("year_high") or _get_info_safe(t, "fiftyTwoWeekHigh")
        year_low  = finfo.get("year_low")  or _get_info_safe(t, "fiftyTwoWeekLow")
        chg_52w   = _get_info_safe(t, "52WeekChange")  # proporción

        # EV/Sales & EV/EBITDA
        ev_sales  = _safe_div(ev, rev)
        ev_ebitda = _safe_div(ev, ebitda)

        # Interest coverage: EBIT / InterestExpense (si hay financials)
        interest_cov = np.nan
        try:
            fin = t.financials  # DataFrame anual (EBIT, Interest Expense pueden estar)
            if isinstance(fin, pd.DataFrame) and not fin.empty:
                ebit = fin.loc[fin.index.str.contains("Ebit", case=False)].iloc[0].iloc[0]
                int_exp = abs(fin.loc[fin.index.str.contains("InterestExpense", case=False)].iloc[0].iloc[0])
                interest_cov = _safe_div(ebit, int_exp)
        except Exception:
            pass

        rows.append({
            "Ticker": tk,
            "EV/EBITDA": ev_ebitda,
            "EV/Sales": ev_sales,
            "Fwd P/E": fpe,
            "Trailing P/E": tpe,
            "PEG": peg,
            "P/B": pb,
            "P/S": ps,
            "Op Margin": opm,
            "Profit Margin": pm,
            "ROA": roa,
            "ROE": roe,
            "Debt/Equity": de,
            "Interest Cover": interest_cov,
            "Market Cap": mcap,
            "52W High": year_high,
            "52W Low": year_low,
            "52W Chg": chg_52w,
            "Beta": beta,
        })

    df = pd.DataFrame(rows).set_index("Ticker")
    # Formateo suave (que luego podés .style.format en el notebook)
    return df


# ----------------------------------- DCF --------------------------------------

def dcf_intrinsic_value(
    current_revenue: float,
    fcf_margin: float = 0.12,
    growth_rate: float = 0.12,
    years: int = 5,
    terminal_growth: float = 0.04,
    wacc: float = 0.10,
    net_debt: float = 0.0,
    shares_outstanding: float | None = None,
) -> Dict[str, float]:
    """
    DCF muy simple: proyecta FCF = Revenue * fcf_margin con crecimiento 'growth_rate'
    por 'years' años y valor terminal Gordon con 'terminal_growth'.
    Devuelve valor empresa (EV), equity y valor por acción (si se provee 'shares_outstanding').
    """
    rev = float(current_revenue)
    g = float(growth_rate)
    m = float(fcf_margin)
    tg = float(terminal_growth)
    k = float(wacc)

    if k <= tg:
        raise ValueError("WACC debe ser mayor que el crecimiento terminal.")

    # Proyección de FCF explícitos
    fcfs = []
    for t in range(1, years + 1):
        rev = rev * (1 + g)
        fcfs.append(rev * m)

    # Valor terminal (perpetuidad a t=years)
    vt = fcfs[-1] * (1 + tg) / (k - tg)

    # Traer a valor presente
    disc = [(1 / ((1 + k) ** t)) for t in range(1, years + 1)]
    pv_fcfs = sum(f * d for f, d in zip(fcfs, disc))
    pv_vt = vt / ((1 + k) ** years)

    ev = pv_fcfs + pv_vt
    equity = ev - float(net_debt)
    out = {"EV": ev, "Equity": equity}

    if shares_outstanding and shares_outstanding > 0:
        out["Value_per_Share"] = equity / float(shares_outstanding)

    return out


def dcf_from_yfinance(
    ticker: str,
    fcf_margin: float = 0.12,
    growth_rate: float = 0.12,
    years: int = 5,
    terminal_growth: float = 0.04,
    wacc: float = 0.10,
) -> Dict[str, float]:
    """
    Conveniencia: arma inputs desde yfinance (.info/fast_info). Usa:
     - totalRevenue, totalDebt, totalCash, sharesOutstanding
    Si faltan datos críticos, levanta ValueError para que los completes manualmente.
    """
    t = yf.Ticker(ticker)
    info = getattr(t, "get_info", None) or getattr(t, "info", {})
    finfo = getattr(t, "fast_info", {})

    revenue = info.get("totalRevenue", None)
    total_debt = info.get("totalDebt", None)
    cash = info.get("totalCash", None)
    shares = info.get("sharesOutstanding", None) or finfo.get("shares_outstanding")

    if revenue in (None, 0) or shares in (None, 0):
        raise ValueError(
            "Faltan datos para DCF (totalRevenue o sharesOutstanding). "
            "Completar manualmente los inputs."
        )

    net_debt = (total_debt or 0.0) - (cash or 0.0)

    return dcf_intrinsic_value(
        current_revenue=float(revenue),
        fcf_margin=fcf_margin,
        growth_rate=growth_rate,
        years=years,
        terminal_growth=terminal_growth,
        wacc=wacc,
        net_debt=float(net_debt),
        shares_outstanding=float(shares),
    )

