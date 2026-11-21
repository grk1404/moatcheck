from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
import requests
import yfinance_cache as yf

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class FetchError(Exception):
    """Base for all fetch failures. Existing `except FetchError` handlers still work."""


class StructuralNoDataError(FetchError):
    """The security cannot provide Big 5 data (SPAC warrants/units/rights, CEFs,
    foreign private issuers, funds, tickers not listed on a supported exchange, etc.).
    The screener should route these to SKIP / SEC_NA, not Error."""


class TransientFetchError(FetchError):
    """Network / rate-limit / empty-response failure that may succeed on retry.
    The screener may retry these or route to RETRYABLE."""


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Spaces calls across threads to reduce Yahoo Finance burst throttling."""

    def __init__(self, per_second: float = 4.0):
        self.set_rate(per_second)
        self._lock = threading.Lock()
        self._next = 0.0

    def set_rate(self, per_second: float) -> None:
        rate = max(0.25, float(per_second))
        self._interval = 1.0 / rate

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self._interval
        if start > now:
            time.sleep(start - now)


_YF_LIMITER = _RateLimiter(per_second=4.0)


def set_yf_rate(per_second: float) -> None:
    """Adjust global Yahoo request pace (e.g., lower during large screener runs)."""
    _YF_LIMITER.set_rate(per_second)


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------
def _make_session() -> requests.Session:
    """Plain requests session for yfinance.

    yfinance 1.5.x defaults to curl_cffi, which can fail SSL verification on
    corporate Windows machines behind TLS-inspecting proxies.
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


# ---------------------------------------------------------------------------
# Yahoo call wrapper
# ---------------------------------------------------------------------------
_TRANSIENT_TOKENS = (
    "429", "too many requests", "timeout", "timed out",
    "connection", "502", "503", "504", "reset by peer", "econnreset",
)


def _yf_call(fn: Callable[[], T], retries: int = 3, base_delay: float = 0.8) -> T:
    """Execute a Yahoo call with rate limiting and jittered backoff on transient errors.

    NOTE: yfinance can silently return an empty DataFrame under rate limiting
    *without* raising. This wrapper only retries on raised exceptions; the
    caller should pair it with `_fetch_stmt_with_empty_retry` for statements.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        _YF_LIMITER.wait()
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            is_transient = any(tok in msg for tok in _TRANSIENT_TOKENS)
            if attempt == retries - 1 or not is_transient:
                raise
            sleep_s = min(15.0, base_delay * (2 ** attempt) + random.uniform(0.1, 0.5))
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def _fetch_stmt_with_empty_retry(
    fetcher: Callable[[], pd.DataFrame | None],
    retries: int = 2,
    delay: float = 2.0,
) -> pd.DataFrame | None:
    """Fetch a statement, retrying if yfinance returns empty without raising.

    yfinance silently returns empty on some rate-limit / stale-session paths.
    A single delayed retry recovers most of those cases without the caller
    having to distinguish them from genuinely missing data.
    """
    last: pd.DataFrame | None = None
    for attempt in range(retries):
        df = _yf_call(fetcher)  # raises on transient exceptions
        if df is not None and not df.empty:
            return df
        last = df
        if attempt < retries - 1:
            time.sleep(delay * (attempt + 1))
    return last


# ---------------------------------------------------------------------------
# Financials container
# ---------------------------------------------------------------------------
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
    headline_eps_ttm: float | None = None
    analyst_5yr_growth: float | None = None
    analyst_5yr_growth_raw: float | None = None
    analyst_forward_growth: float | None = None
    analyst_forward_growth_source: str = ""
    shares_outstanding: float | None = None
    dividend_yield: float | None = None
    beta: float | None = None
    book_value_per_share: float | None = None
    bvps_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    analyst_rec_key: str | None = None
    analyst_rec_mean: float | None = None
    analyst_count: int | None = None
    analyst_target_mean: float | None = None
    analyst_target_high: float | None = None
    analyst_target_low: float | None = None
    company_name: str = ""
    data_source: str = "yfinance"
    data_source_note: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    exchange: str = ""
    week52_high: float | None = None
    week52_low: float | None = None
    company_summary: str = ""
    company_sector: str = ""
    company_industry: str = ""
    company_country: str = ""

    @property
    def years_available(self) -> int:
        return int(self.revenue.dropna().shape[0])

    def has_big5_history(self, min_years: int = 5) -> bool:
        """True if enough annual data exists to attempt a Big 5 calculation."""
        return self.years_available >= min_years


# ---------------------------------------------------------------------------
# Row aliases
# ---------------------------------------------------------------------------
_ROW_ALIASES = {
    "revenue": [
        "Total Revenue", "TotalRevenue", "Revenue", "Operating Revenue",
        "Revenue From Operations", "Total Income", "Income From Operations",
        "Revenue from operations",
    ],
    "net_income": [
        "Net Income", "NetIncome", "Net Income Common Stockholders",
        "Net Income From Continuing Operation Net Minority Interest",
        "Profit After Tax", "PAT", "Net Profit", "Net Profit After Tax",
        "Profit for the Year",
    ],
    "eps": [
        "Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS",
        "Earnings Per Share", "EPS - Basic", "EPS - Diluted",
    ],
    "shares": [
        "Diluted Average Shares", "DilutedAverageShares",
        "Basic Average Shares", "BasicAverageShares",
        "Share Issued", "Number of Shares", "Shares Outstanding",
        "Weighted Average Shares",
    ],
    "ebit": [
        "EBIT", "Operating Income", "OperatingIncome",
        "Earnings Before Interest and Tax", "Operating Profit",
    ],
    "tax_provision": [
        "Tax Provision", "Income Tax Expense", "TaxProvision",
        "Tax Expense", "Provision for Taxation", "Income Tax",
    ],
    "pretax_income": [
        "Pretax Income", "PretaxIncome", "Income Before Tax",
        "Profit Before Tax", "PBT", "Income Before Taxation",
    ],
    "equity": [
        "Stockholders Equity", "StockholdersEquity",
        "Total Stockholder Equity", "Common Stock Equity",
        "Total Equity", "Shareholders Funds", "Shareholders Equity",
        "Total Shareholders Equity",
    ],
    "long_term_debt": [
        "Long Term Debt", "LongTermDebt",
        "Long Term Debt And Capital Lease Obligation",
        "Long Term Borrowings", "Non-Current Liabilities", "Debt - Long Term",
    ],
    "ocf": [
        "Operating Cash Flow", "OperatingCashFlow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
        "Net Cash From Operating Activities", "Cash Flow from Operating Activities",
    ],
    "capex": [
        "Capital Expenditure", "CapitalExpenditure", "Capital Expenditures",
        "Purchase of Fixed Assets", "Capital Expenditure on Fixed Assets",
        "Purchase of Property, Plant and Equipment",
    ],
    "fcf": ["Free Cash Flow", "FreeCashFlow", "Net Cash Flow"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_ticker(ticker: str) -> tuple[str, str]:
    """Normalize a user-supplied ticker and detect its exchange."""
    ticker = ticker.strip().upper()

    if ticker.endswith(".NS"):
        return ticker, "NSE"
    if ticker.endswith(".BO"):
        return ticker, "BSE"

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


def _pick_row(df: pd.DataFrame | None, keys: list[str]) -> pd.Series | None:
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
    try:
        s.index = [pd.Timestamp(c).year for c in s.index]
    except Exception:
        pass
    s = s[~s.index.duplicated(keep="first")]
    s = s.sort_index()
    return pd.to_numeric(s, errors="coerce").dropna()


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip()
    if s in ("", "None", "nan", "NaN", "inf", "-inf"):
        return None
    try:
        f = float(s)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return None


def _fast_info_get(fi: Any, *keys: str) -> Any:
    """Best-effort getter for yfinance FastInfo across versions/forks."""
    if fi is None:
        return None
    for k in keys:
        try:
            v = fi[k]
        except (KeyError, TypeError, AttributeError):
            v = getattr(fi, k, None)
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def fetch(ticker: str) -> Financials:
    """Pull normalized annual financials for `ticker` via yfinance.

    Raises
    ------
    StructuralNoDataError
        The security cannot provide Big 5 data (SPAC warrant/unit/right,
        CEF, foreign private issuer with no usable statements, etc.).
    TransientFetchError
        Yahoo was rate limited / timed out / returned transiently empty
        across all retries. Safe to retry the whole fetch.
    """
    if not ticker or not ticker.strip():
        raise FetchError("Please enter a ticker symbol.")

    original = ticker.strip().upper()
    symbol, exchange = _normalize_ticker(original)

    candidates: list[tuple[str, str]] = [(symbol, exchange)]
    if "." not in original:
        if symbol != original:
            candidates.append((original, ""))
    else:
        bare = original.split(".")[0]
        candidates.append((bare, ""))

    income: pd.DataFrame | None = None
    balance: pd.DataFrame | None = None
    cash: pd.DataFrame | None = None
    yft = None
    last_error: Exception | None = None
    transient_seen = False
    balance_only_seen = False

    for cand_symbol, cand_exchange in candidates:
        try:
            cand_yft = yf.Ticker(cand_symbol, session=_SESSION)
            cand_income = _fetch_stmt_with_empty_retry(lambda: cand_yft.income_stmt)
            cand_balance = _fetch_stmt_with_empty_retry(lambda: cand_yft.balance_sheet)
            cand_cash = _fetch_stmt_with_empty_retry(lambda: cand_yft.cashflow)
        except Exception as e:
            last_error = e
            transient_seen = True
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
            if not has_income and has_balance:
                balance_only_seen = True
            break

    if yft is None:
        detail = f" ({last_error})" if last_error else ""
        if transient_seen:
            raise TransientFetchError(
                f"Yahoo Finance unavailable or rate-limited for {original} after retries.{detail}"
            )
        raise StructuralNoDataError(
            f"No financial statements available for {original}. "
            f"Check the ticker symbol or try a US-listed company.{detail}"
        )

    # ------------------------------------------------------------------
    # Statement row extraction
    # ------------------------------------------------------------------
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

    if not fcf_direct.empty:
        fcf = fcf_direct
    elif not ocf.empty and not capex.empty:
        aligned = pd.concat([ocf, capex], axis=1, join="inner")
        aligned.columns = ["ocf", "capex"]
        fcf = aligned["ocf"] - aligned["capex"].abs()
    else:
        fcf = pd.Series(dtype=float)

    if eps.empty and not net_income.empty and not shares.empty:
        aligned = pd.concat([net_income, shares], axis=1, join="inner")
        aligned.columns = ["ni", "sh"]
        aligned = aligned[aligned["sh"] > 0]
        eps = aligned["ni"] / aligned["sh"]

    if not tax_prov.empty and not pretax.empty:
        aligned = pd.concat([tax_prov, pretax], axis=1, join="inner")
        aligned.columns = ["tax", "pretax"]
        aligned = aligned[aligned["pretax"] != 0]
        tax_rate = (aligned["tax"] / aligned["pretax"]).clip(lower=0, upper=0.5)
    else:
        tax_rate = pd.Series(dtype=float)

    if revenue.empty and net_income.empty and equity.empty:
        if transient_seen:
            raise TransientFetchError(
                f"Yahoo returned empty financials for {symbol} after retries."
            )
        raise StructuralNoDataError(
            f"No usable financials returned for {symbol}. The ticker may be "
            "delisted, non-US, or recently IPO'd with no annual filings yet."
        )

    # ------------------------------------------------------------------
    # Quote / profile
    # ------------------------------------------------------------------
    current_price: float | None = None
    market_cap: float | None = None
    company_name = ""
    fast_info_cache: Any = None

    try:
        fast_info_cache = _yf_call(lambda: yft.fast_info)
        current_price = _safe_float(
            _fast_info_get(fast_info_cache, "last_price", "lastPrice")
        )
        market_cap = _safe_float(
            _fast_info_get(fast_info_cache, "market_cap", "marketCap")
        )
    except Exception:
        pass

    pe_ratio_ttm = None
    headline_eps_ttm = None
    analyst_5yr_growth = None
    analyst_5yr_growth_raw = None
    analyst_forward_growth = None
    analyst_forward_growth_source = ""
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
    week52_high = None
    week52_low = None
    company_summary = ""
    company_sector = ""
    company_industry = ""
    company_country = ""
    info: dict[str, Any] = {}
    info_fetch_failed = False

    try:
        info = _yf_call(lambda: yft.info) or {}
        company_name = info.get("longName") or info.get("shortName") or symbol
        info_exchange = info.get("exchange", "") or info.get("fullExchangeName", "")
        if info_exchange:
            exchange = info_exchange

        week52_high = _safe_float(info.get("fiftyTwoWeekHigh"))
        week52_low = _safe_float(info.get("fiftyTwoWeekLow"))

        company_summary = info.get("longBusinessSummary") or ""
        company_sector = info.get("sector") or ""
        company_industry = info.get("industry") or ""
        company_country = info.get("country") or ""

        if current_price is None:
            current_price = _safe_float(
                info.get("currentPrice") or info.get("regularMarketPrice")
            )
        if market_cap is None:
            market_cap = _safe_float(info.get("marketCap"))

        pe_ratio_ttm = _safe_float(info.get("trailingPE"))
        headline_eps_ttm = _safe_float(info.get("trailingEps"))

        g = _safe_float(info.get("earningsGrowth") or info.get("revenueGrowth"))
        if g is not None:
            analyst_5yr_growth = g
            analyst_5yr_growth_raw = g

        try:
            ge = _yf_call(lambda: yft.growth_estimates)
            if ge is not None and not ge.empty and "stockTrend" in ge.columns:
                for period in ("+1y", "0y"):
                    if period in ge.index:
                        v = _safe_float(ge.loc[period, "stockTrend"])
                        if v is not None:
                            analyst_forward_growth = v
                            analyst_forward_growth_source = period
                            break
        except Exception:
            pass

        shares_outstanding = _safe_float(info.get("sharesOutstanding"))
        dy = _safe_float(info.get("dividendYield"))
        if dy is not None:
            dividend_yield = dy / 100.0
        beta = _safe_float(info.get("beta"))
        book_value_per_share = _safe_float(info.get("bookValue"))

        rec_key = info.get("recommendationKey")
        analyst_rec_key = (
            rec_key if isinstance(rec_key, str) and rec_key != "none" else None
        )
        analyst_rec_mean = _safe_float(info.get("recommendationMean"))
        num_ops = _safe_float(info.get("numberOfAnalystOpinions"))
        analyst_count = int(num_ops) if num_ops is not None else None
        analyst_target_mean = _safe_float(info.get("targetMeanPrice"))
        analyst_target_high = _safe_float(info.get("targetHighPrice"))
        analyst_target_low = _safe_float(info.get("targetLowPrice"))
    except Exception:
        info_fetch_failed = True
        if not company_name:
            company_name = symbol

    if shares_outstanding is None and market_cap and current_price:
        shares_outstanding = market_cap / current_price

    # ------------------------------------------------------------------
    # Currency normalization
    # ------------------------------------------------------------------
    fin_ccy = (info.get("financialCurrency") or "").upper() or None
    quote_ccy = (info.get("currency") or "").upper() or None
    fx_rate = None
    fx_note = ""

    if fin_ccy and quote_ccy and fin_ccy != quote_ccy:
        try:
            if fin_ccy == "USD" or quote_ccy == "USD":
                other = quote_ccy if fin_ccy == "USD" else fin_ccy
                hist = _yf_call(
                    lambda: yf.Ticker(f"{other}=X", session=_SESSION).history(period="5d")
                )
                if hist is not None and not hist.empty:
                    rate_other_per_usd = float(hist["Close"].iloc[-1])
                    fx_rate = (
                        rate_other_per_usd if fin_ccy == "USD"
                        else (1.0 / rate_other_per_usd)
                    )
                    fx_note = (
                        f"Statements in {fin_ccy} converted to {quote_ccy} at "
                        f"1 {fin_ccy} = {fx_rate:.6f} {quote_ccy}."
                    )
            else:
                fx1 = _yf_call(
                    lambda: yf.Ticker(f"{fin_ccy}=X", session=_SESSION).history(period="5d")
                )
                fx2 = _yf_call(
                    lambda: yf.Ticker(f"{quote_ccy}=X", session=_SESSION).history(period="5d")
                )
                if fx1 is not None and not fx1.empty and fx2 is not None and not fx2.empty:
                    fin_per_usd = float(fx1["Close"].iloc[-1])
                    quote_per_usd = float(fx2["Close"].iloc[-1])
                    fx_rate = quote_per_usd / fin_per_usd
                    fx_note = (
                        f"Statements in {fin_ccy} converted to {quote_ccy} at "
                        f"1 {fin_ccy} = {fx_rate:.6f} {quote_ccy}."
                    )
        except Exception:
            fx_rate = None

    # FIX: locals() writes do not rebind function locals — reassign explicitly.
    if fx_rate is not None:
        if not revenue.empty:
            revenue = revenue * fx_rate
        if not net_income.empty:
            net_income = net_income * fx_rate
        if not eps.empty:
            eps = eps * fx_rate
        if not equity.empty:
            equity = equity * fx_rate
        if not ocf.empty:
            ocf = ocf * fx_rate
        if not capex.empty:
            capex = capex * fx_rate
        if not fcf.empty:
            fcf = fcf * fx_rate
        if not long_term_debt.empty:
            long_term_debt = long_term_debt * fx_rate
        if not ebit.empty:
            ebit = ebit * fx_rate
        if book_value_per_share is not None:
            book_value_per_share = book_value_per_share * fx_rate

    # ------------------------------------------------------------------
    # 52-week range fallback
    # ------------------------------------------------------------------
    if week52_high is None:
        week52_high = _safe_float(
            _fast_info_get(fast_info_cache, "year_high", "yearHigh", "fiftyTwoWeekHigh")
        )
    if week52_low is None:
        week52_low = _safe_float(
            _fast_info_get(fast_info_cache, "year_low", "yearLow", "fiftyTwoWeekLow")
        )

    if week52_high is None or week52_low is None:
        try:
            _hist = _yf_call(lambda: yft.history(period="1y"))
            if _hist is not None and not _hist.empty:
                if week52_high is None:
                    week52_high = _safe_float(_hist["High"].max())
                if week52_low is None:
                    week52_low = _safe_float(_hist["Low"].min())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Compose notes
    # ------------------------------------------------------------------
    notes: list[str] = []
    if fx_note:
        notes.append(fx_note)
    if info_fetch_failed:
        notes.append(
            "Yahoo profile/quote data unavailable; price, market cap, "
            "analyst estimates, and 52-week range may be missing."
        )
    if balance_only_seen:
        notes.append(
            "Income statement unavailable; Big 5 revenue/EPS metrics may be incomplete."
        )
    data_source_note = " ".join(notes)

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
        headline_eps_ttm=headline_eps_ttm,
        analyst_5yr_growth=analyst_5yr_growth,
        analyst_5yr_growth_raw=analyst_5yr_growth_raw,
        analyst_forward_growth=analyst_forward_growth,
        analyst_forward_growth_source=analyst_forward_growth_source,
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
        data_source_note=data_source_note,
        raw={
            "income": income,
            "balance": balance,
            "cash": cash,
            "fx_rate": fx_rate,
            "fin_ccy": fin_ccy,
            "quote_ccy": quote_ccy,
            "candidate_tried": [c[0] for c in candidates],
            "info_fetch_failed": info_fetch_failed,
            "balance_only": balance_only_seen,
        },
        exchange=exchange,
        week52_high=week52_high,
        week52_low=week52_low,
        company_summary=company_summary,
        company_sector=company_sector,
        company_industry=company_industry,
        company_country=company_country,
    )