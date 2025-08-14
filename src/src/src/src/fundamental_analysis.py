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
