from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance_cache as yf
from curl_cffi import requests as curl_requests

_session = curl_requests.Session(impersonate="chrome")


class FetchError(Exception):
    """Raised when we cannot get usable financials for a ticker."""


def _make_session() -> requests.Session:
    """Plain requests session for yfinance.

    yfinance 1.5.x defaults to curl_cffi, which fails SSL verification on
    corporate Windows machines behind TLS-inspecting proxies. Passing our
    own requests.Session sidesteps that entirely.
    """
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )
    return s


_SESSION = _make_session()


@dataclass
class Financials:
    ticker: str
    revenue: pd.Series
    net_income: pd.Series
    eps: pd.Series
    equity: pd.Series
    shares: pd.Series
    operating_cash_flow: pd.Series
    capex: pd.Series
    free_cash_flow: pd.Series
    long_term_debt: pd.Series
    tax_rate: pd.Series
    ebit: pd.Series
    current_price: float | None
    market_cap: float | None
    pe_ratio_ttm: float | None
    analyst_5yr_growth: float | None
    shares_outstanding: float | None = None
    dividend_yield: float | None = None
    beta: float | None = None
    book_value_per_share: float | None = None
    bvps_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    analyst_rec_key: str | None = None          # "strong_buy" / "buy" / "hold" / "sell" / "strong_sell"
    analyst_rec_mean: float | None = None       # 1.0 (strong buy) to 5.0 (strong sell)
    analyst_count: int | None = None
    analyst_target_mean: float | None = None
    analyst_target_high: float | None = None
    analyst_target_low: float | None = None
    company_name: str = ""
    data_source: str = "yfinance"           # "edgar+yfinance" or "yfinance"
    data_source_note: str = ""              # UI-facing message about data source
    raw: dict[str, Any] = field(default_factory=dict)
    exchange: str = ""                      # "NSE", "BSE", "NYSE", etc. 

    @property
    def years_available(self) -> int:
        return int(self.revenue.dropna().shape[0])


_ROW_ALIASES = {
    "revenue": ["Total Revenue", "TotalRevenue", "Revenue", "Operating Revenue",
                "Revenue From Operations", "Total Income", "Income From Operations",
        "Revenue from operations"],
    "net_income": [
        "Net Income",
        "NetIncome",
        "Net Income Common Stockholders",
        "Net Income From Continuing Operation Net Minority Interest",
         "Profit After Tax", "PAT", "Net Profit", "Net Profit After Tax", 
        "Profit for the Year"
    ],
    "eps": ["Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS", "Earnings Per Share", "EPS - Basic", "EPS - Diluted"],
    "shares": [
        "Diluted Average Shares",
        "DilutedAverageShares",
        "Basic Average Shares",
        "BasicAverageShares",
        "Share Issued",
        "Number of Shares", "Shares Outstanding", "Weighted Average Shares"
    ],
    "ebit": ["EBIT", "Operating Income", "OperatingIncome", "Earnings Before Interest and Tax", "Operating Profit"],
    "tax_provision": ["Tax Provision", "Income Tax Expense", "TaxProvision", "Tax Expense", "Provision for Taxation", "Income Tax"],
    "pretax_income": ["Pretax Income", "PretaxIncome", "Income Before Tax",  "Profit Before Tax", "PBT", "Income Before Taxation"],
    "equity": [
        "Stockholders Equity",
        "StockholdersEquity",
        "Total Stockholder Equity",
        "Common Stock Equity",
        "Total Equity", "Shareholders Funds", "Shareholders Equity",
        "Total Shareholders Equity"
    ],
    "long_term_debt": [
        "Long Term Debt",
        "LongTermDebt",
        "Long Term Debt And Capital Lease Obligation",
        "Long Term Borrowings", "Non-Current Liabilities", "Debt - Long Term"
    ],
    "ocf": [
        "Operating Cash Flow",
        "OperatingCashFlow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
        "Net Cash From Operating Activities", "Cash Flow from Operating Activities"
    ],
    "capex": ["Capital Expenditure", "CapitalExpenditure", "Capital Expenditures",  "Purchase of Fixed Assets", "Capital Expenditure on Fixed Assets",
        "Purchase of Property, Plant and Equipment"],
    "fcf": ["Free Cash Flow", "FreeCashFlow",  "Free Cash Flow", "Net Cash Flow"],
}

def _normalize_ticker(ticker: str) -> tuple[str, str]:
    """
    Normalize a user-supplied ticker and detect its exchange.

    Rules (in order):
      1. If the ticker already ends in .NS or .BO, keep it and mark the exchange.
      2. If the ticker is in the explicit indian_tickers allowlist, append .NS.
      3. Otherwise, leave the ticker as-is and let the caller try it bare first.

    Note: this function does NOT auto-append .NS to "any short alpha ticker"
    — that heuristic matched almost every US ticker (PGR, AON, AA, AER, ...)
    and silently misrouted them to the Indian exchange.
    """
    ticker = ticker.strip().upper()

    if ticker.endswith(".NS"):
        return ticker, "NSE"
    if ticker.endswith(".BO"):
        return ticker, "BSE"

    # Explicit allowlist of NSE symbols that are commonly typed without
    # a suffix. Only these get auto-suffixed.
    indian_tickers = {
        "RELIANCE", "TCS", "HDFC", "HDFCBANK", "INFY", "ICICIBANK",
        "ITC", "KOTAKBANK", "SBIN", "BHARTIARTL", "HINDUNILVR",
        "LT", "AXISBANK", "WIPRO", "ASIANPAINT", "MARUTI", "HCLTECH",
        "SUNPHARMA", "TITAN", "ULTRACEMCO", "BAJFINANCE", "ADANIENT",
        "ADANIPORTS", "NTPC", "ONGC", "POWERGRID", "TATASTEEL",
        "INDUSINDBK", "BRITANNIA", "GRASIM", "JSWSTEEL", "NESTLEIND",
        "HDFCLIFE", "SBILIFE", "BAJAJFINSV", "TECHM", "COALINDIA",
        "DRREDDY", "EICHERMOT", "HINDALCO", "M&M", "SHREECEM",
        "TATAMOTORS", "UPL", "BAJAJ-AUTO", "BHARATFORGE", "DIVISLAB",
        "ZOMATO", "DMART", "JIOFIN",
        "ABFRL", "ACC", "ADANIGREEN", "ADANITRANS", "ALKEM", "AMBUJACEM",
        "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", "AUROPHARMA", "BANDHANBNK",
        "BANKBARODA", "BERGEPAINT", "BIOCON", "BOSCHLTD", "CANBK",
        "CHOLAFIN", "CIPLA", "COLPAL", "DABUR", "DALBHARAT", "DIXON",
        "DLF", "GODREJCP", "GODREJPROP", "HAVELLS", "HEROMOTOCO",
        "HINDZINC", "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB",
        "INDIGO", "IOC", "IRCTC", "JINDALSTEL", "JUBLFOOD", "LICHSGFIN",
        "MCDOWELL-N", "MUTHOOTFIN", "NAUKRI", "NAVINFLUOR", "NMDC",
        "PAGEIND", "PEL", "PIDILITIND", "PIIND", "PNB", "POLYCAB",
        "POWERINDIA", "RBLBANK", "RECLTD", "SAIL", "SIEMENS", "SRTRANSFIN",
        "SUNTV", "TATACONSUM", "TATAPOWER", "TORNTPHARM", "TRENT",
        "TVSMOTOR", "UBL", "VEDL", "VOLTAS", "WHIRLPOOL", "YESBANK",
        "ZEEL",
    }

    if ticker in indian_tickers:
        return f"{ticker}.NS", "NSE"

    return ticker, ""

def _pick_row(df: pd.DataFrame, keys: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return None


def _to_year_series(row: pd.Series | None) -> pd.Series:
    if row is None:
        return pd.Series(dtype=float)
    s = row.copy()
    # yfinance columns are Timestamps; map to fiscal year int
    try:
        s.index = [pd.Timestamp(c).year for c in s.index]
    except Exception:
        pass
    s = s[~s.index.duplicated(keep="first")]
    s = s.sort_index()
    return pd.to_numeric(s, errors="coerce").dropna()


def fetch(ticker: str) -> Financials:
    """Pull normalized annual financials for `ticker` via yfinance.

    Raises FetchError with a user-friendly message if the ticker is invalid
    or returns no usable data.

    Resolution order:
      1. If the ticker already has .NS/.BO, use it as-is.
      2. If it's in the explicit Indian allowlist, try <ticker>.NS first.
      3. Otherwise try the bare ticker first.
      4. If the first candidate fails, try the alternate form (bare vs suffixed).
    """
    if not ticker or not ticker.strip():
        raise FetchError("Please enter a ticker symbol.")

    original = ticker.strip().upper()
    symbol, exchange = _normalize_ticker(original)

    # Build an ordered list of candidates to try.
    # The first candidate is what _normalize_ticker resolved to; the second
    # is the alternate form (bare vs suffixed) as a fallback.
    candidates: list[tuple[str, str]] = [(symbol, exchange)]

    if "." not in original:
        # User typed a bare ticker. If the resolver auto-suffixed it,
        # also try the bare form as a fallback.
        if symbol != original:
            candidates.append((original, ""))
    else:
        # User typed an explicit suffix. If it fails, try without the suffix.
        bare = original.split(".")[0]
        candidates.append((bare, ""))

    income = None
    balance = None
    cash = None
    yft = None
    last_error: Exception | None = None

    for cand_symbol, cand_exchange in candidates:
        try:
            cand_yft = yf.Ticker(cand_symbol, session=_SESSION)
            cand_income = cand_yft.income_stmt
            cand_balance = cand_yft.balance_sheet
            cand_cash = cand_yft.cashflow
        except Exception as e:
            last_error = e
            continue

        has_income = cand_income is not None and not cand_income.empty
        has_balance = cand_balance is not None and not cand_balance.empty

        if has_income or has_balance:
            yft = cand_yft
            income = cand_income
            balance = cand_balance
            cash = cand_cash
            symbol = cand_symbol
            exchange = cand_exchange or exchange
            break
        # Empty result — try the next candidate.

    if yft is None:
        detail = f" ({last_error})" if last_error else ""
        raise FetchError(
            f"No financial statements available for {original}. "
            f"Check the ticker symbol or try a US-listed company.{detail}"
        )

    # ---------- Extract annual series ----------
    revenue = _to_year_series(_pick_row(income, _ROW_ALIASES["revenue"]))
    net_income = _to_year_series(_pick_row(income, _ROW_ALIASES["net_income"]))
    eps = _to_year_series(_pick_row(income, _ROW_ALIASES["eps"]))
    shares = _to_year_series(_pick_row(income, _ROW_ALIASES["shares"]))
    ebit = _to_year_series(_pick_row(income, _ROW_ALIASES["ebit"]))
    tax_prov = _to_year_series(_pick_row(income, _ROW_ALIASES["tax_provision"]))
    pretax = _to_year_series(_pick_row(income, _ROW_ALIASES["pretax_income"]))

    equity = _to_year_series(_pick_row(balance, _ROW_ALIASES["equity"]))
    long_term_debt = _to_year_series(_pick_row(balance, _ROW_ALIASES["long_term_debt"]))

    ocf = _to_year_series(_pick_row(cash, _ROW_ALIASES["ocf"]))
    capex = _to_year_series(_pick_row(cash, _ROW_ALIASES["capex"]))
    fcf_direct = _to_year_series(_pick_row(cash, _ROW_ALIASES["fcf"]))

    # Prefer Yahoo's direct FCF row; fall back to OCF - |CapEx|.
    if not fcf_direct.empty:
        fcf = fcf_direct
    elif not ocf.empty and not capex.empty:
        aligned = pd.concat([ocf, capex], axis=1, join="inner")
        aligned.columns = ["ocf", "capex"]
        fcf = aligned["ocf"] - aligned["capex"].abs()
    else:
        fcf = pd.Series(dtype=float)

    # EPS fallback: net income / shares.
    if eps.empty and not net_income.empty and not shares.empty:
        aligned = pd.concat([net_income, shares], axis=1, join="inner")
        aligned.columns = ["ni", "sh"]
        aligned = aligned[aligned["sh"] > 0]
        eps = aligned["ni"] / aligned["sh"]

    # Effective tax rate per year, defaulting to 21% when unavailable.
    if not tax_prov.empty and not pretax.empty:
        aligned = pd.concat([tax_prov, pretax], axis=1, join="inner")
        aligned.columns = ["tax", "pretax"]
        aligned = aligned[aligned["pretax"] != 0]
        tax_rate = (aligned["tax"] / aligned["pretax"]).clip(lower=0, upper=0.5)
    else:
        tax_rate = pd.Series(dtype=float)

    # If we somehow got nothing at all, bail.
    if revenue.empty and net_income.empty and equity.empty:
        raise FetchError(
            f"No usable financials returned for {symbol}. The ticker may be "
            "delisted, non-US, or recently IPO'd with no annual filings yet."
        )

    # ---------- Snapshot info: price, market cap, PE, analyst growth ----------
    current_price = None
    market_cap = None
    company_name = ""
    try:
        fi = yft.fast_info
        current_price = float(fi.get("last_price") or fi.get("lastPrice") or np.nan)
        if np.isnan(current_price):
            current_price = None
        market_cap = fi.get("market_cap") or fi.get("marketCap")
        if market_cap is not None:
            market_cap = float(market_cap)
    except Exception:
        pass

    pe_ratio_ttm = None
    analyst_5yr_growth = None
    shares_outstanding = None
    dividend_yield = None
    beta = None
    book_value_per_share = None
    analyst_rec_key = None
    analyst_rec_mean = None
    analyst_count = None
    analyst_target_mean = None
    analyst_target_high = None
    analyst_target_low = None
    info_exchange = ""

    try:
        info = yft.info or {}
        company_name = info.get("longName") or info.get("shortName") or symbol
        info_exchange = info.get("exchange", "") or info.get("fullExchangeName", "")
        if info_exchange:
            exchange = info_exchange

        if current_price is None:
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            if current_price is not None:
                current_price = float(current_price)
        if market_cap is None:
            market_cap = info.get("marketCap")
            if market_cap is not None:
                market_cap = float(market_cap)
        pe_ratio_ttm = info.get("trailingPE")
        if pe_ratio_ttm is not None:
            pe_ratio_ttm = float(pe_ratio_ttm)
        g = info.get("earningsGrowth") or info.get("revenueGrowth")
        if g is not None:
            analyst_5yr_growth = float(g)
        so = info.get("sharesOutstanding")
        if so is not None:
            shares_outstanding = float(so)
        dy = info.get("dividendYield")
        if dy is not None:
            # yfinance returns dividend yield as a percentage (e.g. 2.44 = 2.44%).
            dividend_yield = float(dy) / 100
        b = info.get("beta")
        if b is not None:
            beta = float(b)
        bvps = info.get("bookValue")
        if bvps is not None:
            book_value_per_share = float(bvps)
        rec_key = info.get("recommendationKey")
        rec_mean = info.get("recommendationMean")
        num_ops = info.get("numberOfAnalystOpinions")
        tgt_mean = info.get("targetMeanPrice")
        tgt_high = info.get("targetHighPrice")
        tgt_low = info.get("targetLowPrice")
        analyst_rec_key = rec_key if isinstance(rec_key, str) and rec_key != "none" else None
        analyst_rec_mean = float(rec_mean) if rec_mean is not None else None
        analyst_count = int(num_ops) if num_ops is not None else None
        analyst_target_mean = float(tgt_mean) if tgt_mean is not None else None
        analyst_target_high = float(tgt_high) if tgt_high is not None else None
        analyst_target_low = float(tgt_low) if tgt_low is not None else None
    except Exception:
        if not company_name:
            company_name = symbol

    # Fallback: derive shares outstanding from market cap / price.
    if shares_outstanding is None and market_cap and current_price:
        shares_outstanding = market_cap / current_price

    # ---------- Currency normalization ----------
    fin_ccy = None
    quote_ccy = None
    fx_rate = None
    fx_note = ""
    try:
        info_ccy = yft.info or {}
        fin_ccy = (info_ccy.get("financialCurrency") or "").upper() or None
        quote_ccy = (info_ccy.get("currency") or "").upper() or None
    except Exception:
        pass

    if fin_ccy and quote_ccy and fin_ccy != quote_ccy:
        try:
            if fin_ccy == "USD" or quote_ccy == "USD":
                other = quote_ccy if fin_ccy == "USD" else fin_ccy
                fx_pair = f"{other}=X"
                fx_ticker = yf.Ticker(fx_pair, session=_SESSION)
                hist = fx_ticker.history(period="5d")
                if not hist.empty:
                    rate_other_per_usd = float(hist["Close"].iloc[-1])
                    if fin_ccy == "USD":
                        fx_rate = rate_other_per_usd
                    else:
                        fx_rate = 1.0 / rate_other_per_usd
                    fx_note = (
                        f"Statements in {fin_ccy} converted to {quote_ccy} at "
                        f"1 {fin_ccy} = {fx_rate:.6f} {quote_ccy}."
                    )
            else:
                fx1 = yf.Ticker(f"{fin_ccy}=X", session=_SESSION).history(period="5d")
                fx2 = yf.Ticker(f"{quote_ccy}=X", session=_SESSION).history(period="5d")
                if not fx1.empty and not fx2.empty:
                    fin_per_usd = float(fx1["Close"].iloc[-1])
                    quote_per_usd = float(fx2["Close"].iloc[-1])
                    fx_rate = quote_per_usd / fin_per_usd
                    fx_note = (
                        f"Statements in {fin_ccy} converted to {quote_ccy} at "
                        f"1 {fin_ccy} = {fx_rate:.6f} {quote_ccy}."
                    )
        except Exception:
            fx_rate = None

    if fx_rate is not None:
        revenue = revenue * fx_rate if not revenue.empty else revenue
        net_income = net_income * fx_rate if not net_income.empty else net_income
        eps = eps * fx_rate if not eps.empty else eps
        equity = equity * fx_rate if not equity.empty else equity
        ocf = ocf * fx_rate if not ocf.empty else ocf
        capex = capex * fx_rate if not capex.empty else capex
        fcf = fcf * fx_rate if not fcf.empty else fcf
        long_term_debt = long_term_debt * fx_rate if not long_term_debt.empty else long_term_debt
        ebit = ebit * fx_rate if not ebit.empty else ebit
        if book_value_per_share is not None:
            book_value_per_share = book_value_per_share * fx_rate

    return Financials(
        ticker=symbol,
        revenue=revenue,
        net_income=net_income,
        eps=eps,
        equity=equity,
        shares=shares,
        operating_cash_flow=ocf,
        capex=capex,
        free_cash_flow=fcf,
        long_term_debt=long_term_debt,
        tax_rate=tax_rate,
        ebit=ebit,
        current_price=current_price,
        market_cap=market_cap,
        pe_ratio_ttm=pe_ratio_ttm,
        analyst_5yr_growth=analyst_5yr_growth,
        shares_outstanding=shares_outstanding,
        dividend_yield=dividend_yield,
        beta=beta,
        book_value_per_share=book_value_per_share,
        analyst_rec_key=analyst_rec_key,
        analyst_rec_mean=analyst_rec_mean,
        analyst_count=analyst_count,
        analyst_target_mean=analyst_target_mean,
        analyst_target_high=analyst_target_high,
        analyst_target_low=analyst_target_low,
        company_name=company_name or symbol,
        data_source_note=fx_note,
        raw={
            "income": income,
            "balance": balance,
            "cash": cash,
            "fx_rate": fx_rate,
            "fin_ccy": fin_ccy,
            "quote_ccy": quote_ccy,
            "candidate_tried": [c[0] for c in candidates],
        },
        exchange=exchange,
    )