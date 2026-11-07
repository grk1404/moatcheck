"""Stock Screener — Uses built-in universe + MoatCheck deep analysis (No API Key Needed)."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
import time

import pandas as pd
import streamlit as st
from data_provider import get_ticker
from datetime import datetime, timedelta

from moatcheck import compute_big5, fetch, value_price
from moatcheck.big5 import resolve_thresholds
from moatcheck.edgar import EdgarTransientError
from moatcheck.fetcher import FetchError
from moatcheck.tickerlist import (
    get_combined_tickers,
    get_us_stock_tickers,
    get_us_stock_universe,
    get_indian_stock_tickers,
)

import json
from pathlib import Path

SCREENER_CACHE = Path("data/screener_results.json")
SCREENER_TTL = 7 * 24 * 3600      # a successful analysis stays valid for 7 days (prices are refreshed every run)
ERROR_TTL = 3 * 24 * 3600         # a hard failure (no SEC filings / bad data) is re-tried after 3 days
CACHE_VERSION = 3                 # bump when scoring/valuation logic changes; older rows are recomputed
SAVE_EVERY = 100                  # write progress to disk every N finished tickers
UNIVERSE_CACHE = Path("data/screener_universe_v2.csv")      # v2: includes NASDAQ + quote type
_OLD_UNIVERSE_CACHE = Path("data/screener_universe_cache.csv")
UNIVERSE_CACHE_TTL = timedelta(days=7)
UNIVERSE_WORKERS = 12
UNIVERSE_MAX_BLANK = 0.10                      # more than this share of empty rows = incomplete list
UNIVERSE_RETRY_AFTER = timedelta(hours=1)      # how soon an incomplete list may be re-fetched
MAX_VALUATION_GROWTH = 0.30       # growth fed to the value price is capped here (same bound the alert service uses)

# Yahoo exchange codes -> the names shown in the Exchange dropdown
EXCHANGE_DISPLAY = {
    "NSI": "NSE", "BSE": "BSE",
    "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "NASDAQ": "NASDAQ",
    "NYQ": "NYSE", "NYSE": "NYSE", "PCX": "NYSE Arca", "ASE": "AMEX", "AMEX": "AMEX",
    "BTS": "BATS", "BATS": "BATS", "OQB": "OTC", "OQX": "OTC", "PNK": "OTC",
}


def _fetch_meta(sym: str) -> dict:
    """Sector / exchange / market cap / quote type for one ticker (one cheap Yahoo info call)."""
    from data_provider import get_ticker

    info = None
    for attempt in range(4):
        try:
            info = get_ticker(sym).info or {}
            break
        except Exception as e:
            throttled = "ratelimit" in type(e).__name__.lower() or "too many" in str(e).lower() or "429" in str(e)
            if not throttled:
                break
            time.sleep(2 ** attempt + 1)      # 2s, 3s, 5s, 9s
    if info is not None:
        return {
            "ticker": sym,
            "company": info.get("longName") or info.get("shortName") or sym,
            "exchange": info.get("exchange") or "",
            "sector": info.get("sector") or "",
            "industry": info.get("industry") or "",
            "market_cap": info.get("marketCap") or 0,
            "quote_type": info.get("quoteType") or "",
        }
    return {"ticker": sym, "company": sym, "exchange": "", "sector": "", "industry": "",
            "market_cap": 0, "quote_type": ""}


def _blank_share(df: pd.DataFrame) -> float:
    """Share of rows with neither an exchange nor a market cap (Yahoo returned nothing for them)."""
    if df is None or df.empty:
        return 0.0
    no_exch = df["exchange"].fillna("").astype(str).str.strip() == ""
    no_cap = pd.to_numeric(df["market_cap"], errors="coerce").fillna(0) <= 0
    return float((no_exch & no_cap).mean())


def _seed_rows() -> dict[str, dict]:
    """Rows from earlier universe files that already have usable metadata (so a rebuild only
    fetches tickers it has never seen, or ones that failed last time)."""
    seed: dict[str, dict] = {}
    for path in (UNIVERSE_CACHE, _OLD_UNIVERSE_CACHE):
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "quote_type" not in df.columns:
            df["quote_type"] = ""
        for r in df.to_dict(orient="records"):
            ok = (r.get("market_cap") or 0) > 0 or str(r.get("exchange") or "").strip()
            if ok and r.get("ticker") not in seed:
                seed[r["ticker"]] = {k: ("" if pd.isna(v) else v) for k, v in r.items()}
    return seed


def _build_universe_cache(force: bool = False) -> pd.DataFrame:
    """Build or refresh the screener universe (sector, exchange, market cap, quote type).

    Metadata is fetched in parallel and reused from earlier builds, so after the first build
    a refresh only fetches new tickers (about 3 minutes for 7,600 tickers cold, seconds warm).
    """
    if not force and UNIVERSE_CACHE.exists():
        age = datetime.now() - datetime.fromtimestamp(UNIVERSE_CACHE.stat().st_mtime)
        cached = pd.read_csv(UNIVERSE_CACHE)
        blank = _blank_share(cached)
        # A list built while Yahoo was throttling has many empty rows (no exchange, no size), which
        # no filter can match. Do not trust it for a week: re-fetch just the empty rows, but not more
        # often than once an hour so a throttled Yahoo is not hammered on every page load.
        incomplete = blank > UNIVERSE_MAX_BLANK and age > UNIVERSE_RETRY_AFTER
        if age < UNIVERSE_CACHE_TTL and not incomplete:
            return cached

    with st.spinner("📊 Loading raw ticker list from SEC..."):
        raw = get_us_stock_universe()
        if isinstance(raw, pd.DataFrame):
            tickers = raw["ticker"].tolist() if "ticker" in raw.columns else raw.iloc[:, 0].tolist()
        else:
            tickers = list(raw)

    seed = {} if force else _seed_rows()
    todo = [t for t in tickers if t not in seed]
    st.info(f"Universe: {len(tickers):,} tickers ({len(tickers) - len(todo):,} reused, {len(todo):,} to fetch).")

    fetched: dict[str, dict] = {}
    if todo:
        progress = st.progress(0.0, text="Fetching metadata…")
        with ThreadPoolExecutor(max_workers=UNIVERSE_WORKERS) as ex:
            futs = {ex.submit(_fetch_meta, t): t for t in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    row = fut.result()
                except Exception:
                    t = futs[fut]
                    row = {"ticker": t, "company": t, "exchange": "", "sector": "", "industry": "",
                           "market_cap": 0, "quote_type": ""}
                fetched[row["ticker"]] = row
                if i % 25 == 0 or i == len(todo):
                    progress.progress(i / len(todo), text=f"Fetched {i:,}/{len(todo):,}")
        progress.empty()

    cols = ["ticker", "company", "exchange", "sector", "industry", "market_cap", "quote_type"]
    rows = [{c: (seed.get(t) or fetched.get(t) or {}).get(c, "") for c in cols} for t in tickers]
    df = pd.DataFrame(rows, columns=cols)
    UNIVERSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = UNIVERSE_CACHE.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(UNIVERSE_CACHE)
    failed = sum(1 for t in todo if not (fetched.get(t) or {}).get("exchange") and not (fetched.get(t) or {}).get("market_cap"))
    if todo and failed / len(todo) > 0.10:
        st.warning(f"⚠️ {failed:,} of {len(todo):,} new tickers returned no metadata (Yahoo throttling?). "
                   "Click 'Rebuild universe cache' again later; only those tickers are re-fetched.")
    st.success(f"✅ Cached {len(df):,} tickers → {UNIVERSE_CACHE}")
    return df


_CACHE_LOCK = threading.Lock()


def _load_screener_cache() -> dict[str, dict]:
    if not SCREENER_CACHE.exists():
        return {}
    try:
        data = json.loads(SCREENER_CACHE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_screener_cache(cache: dict[str, dict]) -> None:
    """Atomic write (temp file + replace), so a stop mid-save can't corrupt the cache."""
    SCREENER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SCREENER_CACHE.with_suffix(f".{os.getpid()}.tmp")
    with _CACHE_LOCK:
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        tmp.replace(SCREENER_CACHE)


_METRIC_LABELS = {
    "roic": "ROIC", "sales": "Sales", "eps": "EPS", "equity": "Equity (BVPS)", "fcf": "Free Cash Flow",
    "cash_conversion": "Cash Conversion Ratio", "discount_fcf": "Discount to Fair Value (FCF)",
    "discount_np": "Discount to Fair Value (NP)",
}


def _big5_values(b) -> dict:
    """Growth numbers per metric and window. They do not depend on the pass/fail thresholds, so
    they are cached and the score is recomputed from them whenever the thresholds change."""
    out: dict = {}
    for key in _METRIC_LABELS:
        m = getattr(b, key, None)
        if m is not None:
            out[key] = {str(w): v for w, v in m.values.items()}
    return out


def score_from_values(vals: dict | None, thresholds: dict | None) -> float | None:
    """Wonderfulness score from cached growth numbers under the given thresholds (microseconds)."""
    from moatcheck.big5 import Big5Result, MetricResult

    if not vals:
        return None

    def mk(key: str) -> MetricResult:
        v = vals.get(key) or {}
        return MetricResult(label=_METRIC_LABELS[key],
                            values={w: v.get(str(w)) for w in (10, 5, 3, 1)}, passes=False)

    b = Big5Result(
        roic=mk("roic"), sales=mk("sales"), eps=mk("eps"), equity=mk("equity"), fcf=mk("fcf"),
        cash_conversion=mk("cash_conversion") if "cash_conversion" in vals else None,
        discount_fcf=mk("discount_fcf") if "discount_fcf" in vals else None,
        discount_np=mk("discount_np") if "discount_np" in vals else None,
        thresholds=resolve_thresholds(thresholds),
    )
    return b.wonderfulness().overall


def _cache_entry_valid(entry: dict | None, now: float) -> bool:
    if not entry or entry.get("v") != CACHE_VERSION:
        return False
    row = entry.get("row") or {}
    ttl = ERROR_TTL if row.get("error") else SCREENER_TTL
    return now - entry.get("cached_at", 0) < ttl

def _row_to_dict(r: ScreenerRow) -> dict:
    return {
        "company": r.company, "ticker": r.ticker, "score": r.score,
        "stock_price": r.stock_price, "value_price": r.value_price,
        "mos_price": r.mos_price, "verdict": r.verdict,
        "pe_ratio": r.pe_ratio, "exchange": r.exchange,
        "sector": r.sector, "industry": r.industry,
        "market_cap": r.market_cap, "error": r.error,
        "big5_values": r.big5_values,
    }


def _row_from_dict(d: dict) -> ScreenerRow:
    """Reconstruct a ScreenerRow, coercing numeric fields to float."""
    def _f(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return ScreenerRow(
        company=d.get("company", ""),
        ticker=d.get("ticker", ""),
        score=_f(d.get("score")),
        stock_price=_f(d.get("stock_price")),
        value_price=_f(d.get("value_price")),
        mos_price=_f(d.get("mos_price")),
        verdict=d.get("verdict", "Unknown"),
        pe_ratio=_f(d.get("pe_ratio")),
        exchange=d.get("exchange"),
        sector=d.get("sector"),
        industry=d.get("industry"),
        market_cap=_f(d.get("market_cap")),
        error=d.get("error"),
        big5_values=d.get("big5_values"),
    )

# ==================== CONFIGURATION ====================
DEFAULT_MAX_WORKERS = 80          # total concurrent fetches (processes x threads)
PROCESS_MIN_BATCH = 150           # below this, threads in one process are quicker than starting workers
THREADS_PER_PROCESS = 8
DEFAULT_MAX_CANDIDATES = 100

# Filter Options
MARKET_CAP_RANGES = {
    "All Sizes": (0, float("inf")),
    "Micro Cap": (0, 300_000_000),
    "Small Cap": (300_000_000, 2_000_000_000),
    "Mid Cap": (2_000_000_000, 10_000_000_000),
    "Large Cap": (10_000_000_000, 100_000_000_000),
    "Mega Cap": (100_000_000_000, float("inf")),
}

EXCHANGES = ["All Exchanges", "NYSE", "NASDAQ", "AMEX", "BATS", "OTC", "NSE", "BSE"]

SECTORS = [
    "All Sectors",
    "Technology",
    "Financial Services",
    "Healthcare",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Industrials",
    "Energy",
    "Real Estate",
    "Utilities",
    "Communication Services",
    "Basic Materials",
]


# ==================== DATA CLASSES ====================
@dataclass
class ScreenerRow:
    company: str
    ticker: str
    score: float | None
    stock_price: float | None
    value_price: float | None
    mos_price: float | None
    verdict: str
    pe_ratio: float | None = None
    exchange: str | None = None
    week52_high: float | None = None
    week52_low: float | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    error: str | None = None
    transient: bool = False    # SEC/Yahoo hiccup: retry, never cache
    big5_values: dict | None = None   # growth numbers per metric/window (independent of thresholds)


_VERDICT_COLORS = {
    "BARGAIN BUY": "background-color: #0d3d1f; color: #00E676;",
    "BUY": "background-color: #1e4620; color: #b6f0b6;",
    "WATCH": "background-color: #4a3a1e; color: #f0d8a6;",
    "AVOID": "background-color: #4b1e1e; color: #f0b6b6;",
    "Unknown": "background-color: #2a2a2a; color: #9aa0a6;",
    "Error": "background-color: #4b1e1e; color: #f0b6b6;",
}


# ==================== HELPERS ====================
def _currency_symbol(ticker: str) -> str:
    """Return the currency symbol for a ticker based on its suffix."""
    t = (ticker or "").upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return "₹"
    return "$"


def fmt_price(v: float | None, ticker: str = "") -> str:
    """Format a per-share price with the correct currency symbol."""
    if v is None:
        return "n/a"
    symbol = _currency_symbol(ticker)
    return f"{symbol}{v:,.2f}"


def fmt_money(v: float | None, ticker: str = "") -> str:
    if v is None:
        return "n/a"
    symbol = _currency_symbol(ticker)
    if abs(v) >= 1e12:
        return f"{symbol}{v / 1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"{symbol}{v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"{symbol}{v / 1e6:.2f}M"
    return f"{symbol}{v:,.2f}"

# ==================== LOAD UNIVERSE (NO API KEY) ====================
@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_universe() -> pd.DataFrame:
    """Load the enriched stock universe: exchange names normalized to the dropdown values and
    funds / ETFs removed (they have no financial statements and used to be 15% of all errors)."""
    df = _build_universe_cache(force=False).copy()
    df["exchange"] = df["exchange"].fillna("").astype(str).map(lambda x: EXCHANGE_DISPLAY.get(x.strip().upper(), x.strip()))
    if "quote_type" in df.columns:
        qt = df["quote_type"].fillna("").astype(str).str.upper()
        df = df[(qt == "") | (qt == "EQUITY")]
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce").fillna(0)
    return df.reset_index(drop=True)

# ==================== DEEP ANALYSIS ====================
def big5_eps_growth(big5) -> float | None:
    windows = big5.eps.values
    stable = [v for w, v in windows.items() if w in (5, 3, 10) and v is not None]
    if stable:
        stable.sort()
        n = len(stable)
        if n % 2:
            return stable[n // 2]
        return (stable[n // 2 - 1] + stable[n // 2]) / 2
    one = windows.get(1)
    return one if one is not None else None


def positive_eps(fin) -> float | None:
    """Return a plausible per-share EPS, or None if the data looks corrupted."""
    if fin.eps.empty:
        return None

    eps_ttm = float(fin.eps.iloc[-1])
    if eps_ttm > 0:
        if fin.current_price and fin.current_price > 0:
            # Tightened: a real per-share EPS is never more than 50% of the
            # stock price (that would be a P/E below 2, essentially impossible).
            if abs(eps_ttm) / fin.current_price > 0.5:
                return None
        return eps_ttm

    positive_history = fin.eps[fin.eps > 0]
    if positive_history.empty:
        return None

    candidate = float(positive_history.tail(3).mean())
    if fin.current_price and fin.current_price > 0:
        if abs(candidate) / fin.current_price > 0.5:
            return None
    return candidate


def verdict_price(current: float | None, mos: float | None, value: float | None) -> str:
    if current is None or mos is None or value is None or value <= 0 or mos <= 0:
        return "Unknown"
    if current <= mos * 0.5:
        return "BARGAIN BUY"
    if current <= mos:
        return "BUY"
    if current <= value:
        return "WATCH"
    return "AVOID"


def _error_row(ticker: str, err: Exception, transient: bool) -> ScreenerRow:
    """Row for a failed analysis, with whatever basic Yahoo info is available."""
    company, price, cap, exch, sector, industry, pe = ticker, None, None, None, None, None, None
    try:
        info = get_ticker(ticker).info or {}
        company = info.get("longName") or info.get("shortName") or ticker
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        cap = info.get("marketCap")
        exch = EXCHANGE_DISPLAY.get(str(info.get("exchange", "")).upper(), info.get("exchange"))
        sector, industry, pe = info.get("sector"), info.get("industry"), info.get("trailingPE")
    except Exception:
        pass
    return ScreenerRow(
        company=company, ticker=ticker, score=None, stock_price=price, value_price=None,
        mos_price=None, verdict="Retry" if transient else "Error", pe_ratio=pe, exchange=exch,
        sector=sector, industry=industry, market_cap=cap, error=str(err)[:300], transient=transient,
    )


def analyze_ticker(ticker: str, thresholds: dict | None) -> ScreenerRow:
    """Full MoatCheck analysis of one ticker. Pure (no Streamlit), so it can be tested and reused."""
    try:
        fin = fetch(ticker, light=True, persist=False)
        big5 = compute_big5(fin, thresholds=thresholds)
        score = big5.wonderfulness().overall

        current_eps = positive_eps(fin)

        historical_pe = fin.pe_ratio_ttm
        if historical_pe is not None and (historical_pe <= 0 or historical_pe > 200):
            historical_pe = None

        def _clamp(g):
            return None if g is None else min(g, MAX_VALUATION_GROWTH)

        val = value_price(
            current_eps=current_eps,
            big5_eps_growth=_clamp(big5_eps_growth(big5)),
            analyst_growth=_clamp(fin.analyst_5yr_growth),
            historical_pe=historical_pe,
        )
        value_price_val = val.value_price if val else None
        mos_price_val = val.mos_price if val else None

        # An intrinsic value more than 5x the market price is almost always bad input data.
        if value_price_val is not None and fin.current_price and fin.current_price > 0:
            if value_price_val / fin.current_price > 5.0:
                value_price_val = None
                mos_price_val = None
        verdict = verdict_price(fin.current_price, mos_price_val, value_price_val)

        exch = fin.exchange
        t_up = ticker.upper()
        if t_up.endswith(".NS"):
            exch = "NSE"
        elif t_up.endswith(".BO"):
            exch = "BSE"
        else:
            exch = EXCHANGE_DISPLAY.get(str(exch).upper(), exch)

        return ScreenerRow(
            company=fin.company_name or ticker,
            ticker=fin.ticker,
            score=score,
            stock_price=fin.current_price,
            value_price=value_price_val,
            mos_price=mos_price_val,
            verdict=verdict,
            pe_ratio=fin.pe_ratio_ttm,
            exchange=exch,
            sector=fin.company_sector or None,
            industry=fin.company_industry or None,
            market_cap=fin.market_cap,
            error=None,
            big5_values=_big5_values(big5),
        )
    except EdgarTransientError as e:
        return _error_row(ticker, e, transient=True)
    except FetchError as e:
        return _error_row(ticker, e, transient=False)
    except Exception as e:      # unexpected bug: report it, but do not let one ticker stop the scan
        return _error_row(ticker, f"{type(e).__name__}: {e}", transient=False)


def _worker_init(sec_rate: float) -> None:
    """Runs once in each worker process: quiet warnings and take this process's SEC rate share."""
    import warnings
    warnings.filterwarnings("ignore")
    from moatcheck.edgar import set_sec_rate
    set_sec_rate(sec_rate)


def _analyze_chunk(tickers: list[str], thresholds: dict | None, threads: int) -> list[ScreenerRow]:
    """Analyze a chunk of tickers inside a worker process, overlapping the network waits with
    threads. Processes give real CPU parallelism (the parsing is CPU-bound; threads share one core)."""
    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        return list(ex.map(lambda t: analyze_ticker(t, thresholds), tickers))


def deep_analyze_ticker(ticker: str, thresholds_tuple: tuple) -> ScreenerRow:
    """Backward-compatible wrapper (takes the hashable thresholds tuple)."""
    return analyze_ticker(ticker, dict(thresholds_tuple) if thresholds_tuple else None)


def refresh_prices(rows: list[ScreenerRow], chunk: int = 200) -> int:
    """Replace each row's price with the latest close and recompute its verdict.

    Cached rows are up to 7 days old and Yahoo's `info` price can lag by days, so the
    verdict (price vs MOS) must not come from them. One batch download covers 200 tickers
    in about 5 seconds. Rows whose download fails keep their previous price. Returns the
    number of rows updated.
    """
    import yfinance as yf
    from data_provider import _shared_session

    # Only stocks with a valuation need a fresh price (their verdict is price vs MOS). That is
    # about a quarter of the list, which cuts a 7,600-stock refresh from ~3 min to under a minute.
    targets = [r for r in rows if r.error is None and r.value_price and r.mos_price]
    by_sym = {r.ticker: r for r in targets}
    syms = list(by_sym)

    def _one(part: list[str]) -> int:
        try:
            d = yf.download(part, period="5d", progress=False, threads=True,
                            session=_shared_session, auto_adjust=False)
            close = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d[["Close"]].rename(columns={"Close": part[0]})
        except Exception:
            return 0
        n = 0
        for sym in part:
            if sym not in close.columns:
                continue
            col = close[sym].dropna()
            if col.empty:
                continue
            last = pd.Timestamp(col.index[-1])
            if last.tzinfo is not None:
                last = last.tz_localize(None)
            if (pd.Timestamp.now() - last).days > 7:
                continue
            px = float(col.iloc[-1])
            if px <= 0:
                continue
            r = by_sym[sym]
            r.stock_price = px
            r.verdict = verdict_price(px, r.mos_price, r.value_price)
            n += 1
        return n

    chunks = [syms[i:i + chunk] for i in range(0, len(syms), chunk)]
    if not chunks:
        return 0
    with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as ex:
        return sum(ex.map(_one, chunks))


def deep_analyze_batch(
    tickers: list[str],
    max_workers: int = DEFAULT_MAX_WORKERS,
    show_live: bool = True,
    thresholds: dict | None = None,
) -> list[ScreenerRow]:
    """Analyze tickers in parallel with live results, periodic saves and one retry pass for
    transient failures. Cached rows are reused only if their thresholds and version match."""
    if not tickers:
        return []

    now = time.time()
    cache = {k: v for k, v in _load_screener_cache().items() if _cache_entry_valid(v, now)}   # drops stale/old-format rows

    to_run: list[str] = []
    from_cache: list[ScreenerRow] = []
    for t in tickers:
        entry = cache.get(t)
        if entry:
            row = _row_from_dict(entry["row"])
            if row.big5_values:                      # re-score under the CURRENT thresholds
                row.score = score_from_values(row.big5_values, thresholds)
            from_cache.append(row)
        else:
            to_run.append(t)

    def _finish(rows_: list[ScreenerRow]) -> list[ScreenerRow]:
        n = refresh_prices(rows_)
        if n:
            st.caption(f"💲 Refreshed latest prices for {n:,} stocks")
        rows_.sort(key=lambda r: (r.score is None, -(r.score or 0)))
        return rows_

    if not to_run:
        st.info("✅ All results loaded from cache")
        return _finish(list(from_cache))

    st.write(f"🔄 Deep analysis of **{len(to_run):,}** of {len(tickers):,} stocks "
             f"({len(from_cache):,} reused from cache)...")

    live_container = st.empty() if show_live else None
    results: list[ScreenerRow] = []
    total = len(to_run)
    progress_bar = st.progress(0)
    status_text = st.empty()
    new_cache = dict(cache)
    since_save = 0

    def _store(row: ScreenerRow) -> None:
        nonlocal since_save
        if row.transient:
            return                       # never cache a hiccup
        new_cache[row.ticker] = {"cached_at": time.time(), "v": CACHE_VERSION, "row": _row_to_dict(row)}
        since_save += 1
        if since_save >= SAVE_EVERY:
            _save_screener_cache(new_cache)
            since_save = 0

    def _show_live() -> None:
        good = sorted([r for r in results if r.error is None], key=lambda x: -(x.score or 0))
        if not good:
            live_container.info(f"⏳ Waiting for first results... ({len(results)}/{total})")
            return
        df_live = pd.DataFrame([
            {
                "Company": r.company[:30] + "..." if len(r.company) > 30 else r.company,
                "Ticker": r.ticker,
                "Score": f"{_safe_float(r.score):.1f}/10" if _safe_float(r.score) is not None else "n/a",
                "Price": fmt_price(_safe_float(r.stock_price), r.ticker),
                "Value": fmt_price(_safe_float(r.value_price), r.ticker),
                "MOS Price": fmt_price(_safe_float(r.mos_price), r.ticker),
                "Verdict": r.verdict,
            }
            for r in good[:20]
        ])
        live_container.dataframe(
            df_live.style.apply(lambda row: [_VERDICT_COLORS.get(row["Verdict"], "")] * len(row), axis=1),
            use_container_width=True, hide_index=True, height=min(400, 25 * len(df_live) + 40),
        )

    def _record(row: ScreenerRow, out: list, label: str) -> None:
        out.append(row)
        results.append(row)
        _store(row)
        if len(results) % 10 == 0 or len(results) == total:
            progress_bar.progress(min(1.0, len(results) / total))
            status_text.text(f"{label}: {len(results)}/{total} - {row.ticker}")
        if show_live and live_container is not None and len(results) % 25 == 0:
            _show_live()

    def _run_threads(batch: list[str], workers: int, label: str, out: list) -> None:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futs = {executor.submit(analyze_ticker, t, thresholds): t for t in batch}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    row = _error_row(t, e, transient=False)
                _record(row, out, label)

    def _run_processes(batch: list[str], workers: int, label: str, out: list) -> None:
        """Worker processes (real CPU parallelism) each running several fetch threads."""
        from concurrent.futures import ProcessPoolExecutor

        procs = max(2, min((os.cpu_count() or 4) - 2, -(-workers // THREADS_PER_PROCESS), 12))
        threads = max(2, -(-workers // procs))
        chunks = [batch[i:i + 12] for i in range(0, len(batch), 12)]
        with ProcessPoolExecutor(max_workers=procs, initializer=_worker_init, initargs=(8.0 / procs,)) as ex:
            futs = [ex.submit(_analyze_chunk, c, thresholds, threads) for c in chunks]
            for fut in as_completed(futs):
                for row in fut.result():
                    _record(row, out, label)

    def _run_pass(batch: list[str], workers: int, label: str) -> list[ScreenerRow]:
        out: list[ScreenerRow] = []
        if len(batch) >= PROCESS_MIN_BATCH:
            try:
                _run_processes(batch, workers, label, out)
                return out
            except Exception as e:           # pool could not start or broke: finish with threads
                st.caption(f"ℹ️ Process pool unavailable ({type(e).__name__}); continuing with threads.")
        done = {r.ticker for r in out}
        _run_threads([t for t in batch if t not in done], workers, label, out)
        return out

    try:
        first = _run_pass(to_run, max_workers, "Analyzing")
        retry = [r.ticker for r in first if r.transient]
        if retry:
            status_text.text(f"Retrying {len(retry)} tickers that hit a temporary SEC/Yahoo error...")
            time.sleep(5)
            results[:] = [r for r in results if not r.transient]
            total = len(results) + len(retry)
            _run_pass(retry, max(2, max_workers // 5), "Retrying")
        if show_live and live_container is not None:
            _show_live()
    finally:
        _save_screener_cache(new_cache)      # keep progress even if the run is interrupted

    progress_bar.empty()
    status_text.empty()
    still = sum(1 for r in results if r.transient)
    if still:
        st.warning(f"⚠️ {still} stock(s) could not be fetched (SEC/Yahoo busy). They were not cached; run again to retry them.")
    return _finish(from_cache + results)


# ==================== DISPLAY ====================
def _safe_float(v) -> float | None:
    """Coerce a value to float, returning None if it can't be converted."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        f = float(v)
        return f if f == f else None   # reject NaN
    except (TypeError, ValueError):
        return None


def rows_to_dataframe(rows):                    # kept so older callers do not break
    return rows_to_frame(rows)


# ==================== SAVED RESULTS (CSV in the data folder) ====================
RESULTS_CSV = Path("data/screener_results_latest.csv")
RESULTS_META = Path("data/screener_results_latest.json")
RESULTS_STALE_DAYS = 7

# Session key -> default (in percent). Same keys the page has always used.
_THRESHOLD_DEFAULTS = {
    "scr_rev_min": 10.0, "scr_rev_max": 50.0,
    "scr_eps_min": 10.0, "scr_eps_max": 50.0,
    "scr_eq_min": 10.0, "scr_eq_max": 50.0,
    "scr_ocf_min": 10.0, "scr_ocf_max": 50.0,
    "scr_roic_min": 10.0, "scr_roic_max": 50.0,
    "scr_cash_conv": 80.0, "scr_disc_fcf": 30.0, "scr_disc_np": 30.0,
}
_BIG5_ROWS = [
    ("Revenue", "scr_rev_min", "scr_rev_max"),
    ("EPS", "scr_eps_min", "scr_eps_max"),
    ("Equity (BVPS)", "scr_eq_min", "scr_eq_max"),
    ("Operating cash flow", "scr_ocf_min", "scr_ocf_max"),
    ("ROIC", "scr_roic_min", "scr_roic_max"),
]

_FRAME_COLS = ["ticker", "company", "verdict", "score", "stock_price", "value_price", "mos_price",
               "pe_ratio", "market_cap", "sector", "industry", "exchange", "error"]

_VERDICT_BADGE = {
    "BARGAIN BUY": "💎 BARGAIN BUY", "BUY": "🟢 BUY", "WATCH": "🟡 WATCH", "AVOID": "🔴 AVOID",
    "Unknown": "⚪ Unknown", "Error": "⚠️ Error", "Retry": "🔁 Retry",
}
_MAIN_VERDICTS = ["BARGAIN BUY", "BUY", "WATCH", "AVOID", "Unknown"]


def rows_to_frame(rows: list[ScreenerRow]) -> pd.DataFrame:
    """Results as a plain numeric table (this is what is saved to CSV)."""
    recs = [{
        "ticker": r.ticker, "company": r.company, "verdict": r.verdict,
        "score": _safe_float(r.score), "stock_price": _safe_float(r.stock_price),
        "value_price": _safe_float(r.value_price), "mos_price": _safe_float(r.mos_price),
        "pe_ratio": _safe_float(r.pe_ratio), "market_cap": _safe_float(r.market_cap),
        "sector": r.sector or "", "industry": r.industry or "", "exchange": r.exchange or "",
        "error": r.error or "",
    } for r in rows]
    return pd.DataFrame(recs, columns=_FRAME_COLS)


def frame_to_rows(df: pd.DataFrame) -> list[ScreenerRow]:
    def f(v):
        return None if v is None or (isinstance(v, float) and v != v) else float(v)

    def s(v):
        return "" if v is None or (isinstance(v, float) and v != v) else str(v)

    rows = []
    for rec in df.to_dict("records"):
        rows.append(ScreenerRow(
            company=s(rec.get("company")) or s(rec.get("ticker")),
            ticker=s(rec.get("ticker")),
            score=f(rec.get("score")), stock_price=f(rec.get("stock_price")),
            value_price=f(rec.get("value_price")), mos_price=f(rec.get("mos_price")),
            verdict=s(rec.get("verdict")) or "Unknown",
            pe_ratio=f(rec.get("pe_ratio")),
            exchange=s(rec.get("exchange")) or None, sector=s(rec.get("sector")) or None,
            industry=s(rec.get("industry")) or None, market_cap=f(rec.get("market_cap")),
            error=s(rec.get("error")) or None,
        ))
    return rows


def save_results(rows: list[ScreenerRow], meta: dict) -> bool:
    """Write the results CSV and its settings file atomically. Returns False if a file is locked
    (for example open in Excel), so the caller can say so instead of failing silently."""
    try:
        RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        tmp = RESULTS_CSV.with_suffix(f".{os.getpid()}.tmp")
        rows_to_frame(rows).to_csv(tmp, index=False, encoding="utf-8")
        tmp.replace(RESULTS_CSV)
        tmpm = RESULTS_META.with_suffix(f".{os.getpid()}.tmp")
        tmpm.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmpm.replace(RESULTS_META)
        return True
    except OSError:
        return False


def load_saved_results() -> tuple[list[ScreenerRow], dict] | None:
    """The last saved screen as (rows, meta), or None if there is none (or it cannot be read)."""
    if not RESULTS_CSV.exists():
        return None
    try:
        df = pd.read_csv(
            RESULTS_CSV, keep_default_na=False, na_values=[""],
            dtype={c: str for c in ("ticker", "company", "verdict", "sector", "industry", "exchange", "error")},
        )
        rows = frame_to_rows(df)
    except Exception:
        return None
    if not rows:
        return None
    try:
        meta = json.loads(RESULTS_META.read_text(encoding="utf-8")) if RESULTS_META.exists() else {}
    except (OSError, ValueError):
        meta = {}
    if not meta.get("finished_at"):
        meta["finished_at"] = datetime.fromtimestamp(RESULTS_CSV.stat().st_mtime).isoformat(timespec="seconds")
    return rows, meta


# ==================== PAGE ====================
def _init_settings_state(meta: dict) -> None:
    """First visit in a session: put every control back to what produced the saved list."""
    ss = st.session_state
    if ss.get("scr_init"):
        return
    saved = meta.get("settings") or {}
    pct = meta.get("thresholds_pct") or {}
    size = saved.get("size")
    ss.setdefault("scr_size", size if size in MARKET_CAP_RANGES else "All Sizes")
    exch = saved.get("exchange")
    ss.setdefault("scr_exchange", exch if exch in EXCHANGES else "All Exchanges")
    secs = [x for x in (saved.get("sectors") or []) if x in SECTORS]
    ss.setdefault("scr_sectors", secs)
    ss.setdefault("scr_india", bool(saved.get("include_india", False)))
    ss.setdefault("scr_workers", DEFAULT_MAX_WORKERS)
    ss.setdefault("scr_live", True)
    ss.setdefault("scr_force", False)
    for key, default in _THRESHOLD_DEFAULTS.items():
        try:
            ss.setdefault(key, float(pct.get(key, default)))
        except (TypeError, ValueError):
            ss.setdefault(key, default)
    ss.setdefault("scr_v_verdicts", list(_MAIN_VERDICTS))
    ss.setdefault("scr_v_min", 0.0)
    ss.setdefault("scr_v_search", "")
    ss["scr_init"] = True


def _thresholds_from_state() -> dict:
    ss = st.session_state
    g = lambda k: float(ss[k]) / 100.0
    return {
        "revenue": {"min": g("scr_rev_min"), "max": g("scr_rev_max")},
        "eps": {"min": g("scr_eps_min"), "max": g("scr_eps_max")},
        "equity": {"min": g("scr_eq_min"), "max": g("scr_eq_max")},
        "ocf": {"min": g("scr_ocf_min"), "max": g("scr_ocf_max")},
        "roic": {"min": g("scr_roic_min"), "max": g("scr_roic_max")},
        "cash_conversion": {"min": g("scr_cash_conv"), "max": 999.0},
        "discount_fcf": {"min": g("scr_disc_fcf"), "max": 999.0},
        "discount_np": {"min": g("scr_disc_np"), "max": 999.0},
    }


def _describe_settings(meta: dict) -> str:
    s = meta.get("settings") or {}
    parts = [s.get("size", "All Sizes"), s.get("exchange", "All Exchanges")]
    secs = s.get("sectors") or []
    parts.append(", ".join(secs) if secs else "All sectors")
    if s.get("include_india"):
        parts.append("incl. India")
    return " · ".join(parts)


def _run_screen(thresholds: dict) -> tuple[list[ScreenerRow], dict] | None:
    """Scan the stocks that match the current filters, save the result, and return (rows, meta)."""
    ss = st.session_state
    size, exchange, sectors = ss["scr_size"], ss["scr_exchange"], list(ss["scr_sectors"])
    include_india = bool(ss["scr_india"])
    min_mcap, max_mcap = MARKET_CAP_RANGES[size]
    start = time.time()

    if ss["scr_force"]:
        st.cache_data.clear()
        if SCREENER_CACHE.exists():
            SCREENER_CACHE.unlink()

    universe_df = load_stock_universe()
    if not include_india:
        universe_df = universe_df[~universe_df["ticker"].str.endswith((".NS", ".BO"), na=False)]
    else:
        try:
            india_tickers = get_indian_stock_tickers()
            india_df = pd.DataFrame({
                "ticker": india_tickers,
                "exchange": ["NSE" if t.endswith(".NS") else "BSE" for t in india_tickers],
                "sector": [""] * len(india_tickers),
                "market_cap": [0] * len(india_tickers),
            })
            universe_df = pd.concat([universe_df, india_df], ignore_index=True)
        except Exception:
            pass                                    # carry on with US only

    if sectors:
        universe_df = universe_df[universe_df["sector"].isin(sectors)]
    if exchange and exchange != "All Exchanges":
        universe_df = universe_df[universe_df["exchange"] == exchange]
    if min_mcap > 0:
        universe_df = universe_df[universe_df["market_cap"] >= min_mcap]
    if max_mcap < float("inf"):
        universe_df = universe_df[universe_df["market_cap"] <= max_mcap]

    tickers = universe_df["ticker"].tolist()
    if not tickers:
        st.warning("No stocks match the company filters (size, exchange, sector). Loosen them and run again.")
        return None

    st.info(f"📊 Screening **{len(tickers):,}** stocks that match your company filters.")
    results = deep_analyze_batch(tickers, int(ss["scr_workers"]), bool(ss["scr_live"]), thresholds=thresholds)

    elapsed = time.time() - start
    meta = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed, 1),
        "n_tickers": len(tickers),
        "settings": {"size": size, "exchange": exchange, "sectors": sectors, "include_india": include_india},
        "thresholds_pct": {k: float(ss[k]) for k in _THRESHOLD_DEFAULTS},
    }
    if save_results(results, meta):
        st.success(f"✅ Screened {len(results):,} stocks in {elapsed:.0f}s and saved to `{RESULTS_CSV}`.")
    else:
        st.warning(f"Screened {len(results):,} stocks, but `{RESULTS_CSV}` could not be written "
                   "(is it open in Excel?). The results are shown below but were not saved.")
    return results, meta


def _render_settings() -> None:
    """Company filters, Big 5 thresholds and quality filters, in that order."""
    st.markdown("##### Company filters")
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Company size", options=list(MARKET_CAP_RANGES.keys()), key="scr_size")
    c2.selectbox("Exchange", options=EXCHANGES, key="scr_exchange")
    c3.multiselect(
        "Sector", options=[s for s in SECTORS if s != "All Sectors"], key="scr_sectors",
        placeholder="All sectors", help="Pick one or more sectors, or leave empty for all.",
    )

    st.markdown("##### Big 5 growth thresholds")
    st.caption("Annual growth a company must reach (**min**) and must not exceed (**max**), in %. "
               "A company passes a metric only if every window (10, 5, 3 and 1 year) falls inside the range.")
    cols = st.columns(5)
    for col, (label, kmin, kmax) in zip(cols, _BIG5_ROWS):
        with col:
            st.markdown(f"**{label}**")
            a, b = st.columns(2)
            a.number_input("Min %", key=kmin, min_value=0.0, max_value=100.0, step=1.0, format="%.0f")
            b.number_input("Max %", key=kmax, min_value=0.0, max_value=100.0, step=1.0, format="%.0f")

    st.markdown("##### Additional quality filters")
    st.caption("Minimums, in %. A company passes if its value is at or above the number.")
    q1, q2, q3 = st.columns(3)
    q1.number_input("Cash conversion at least", key="scr_cash_conv", min_value=0.0, max_value=500.0,
                    step=5.0, format="%.0f", help="Operating cash flow as a share of net income, 10-year average.")
    q2.number_input("Discount to fair value (FCF) at least", key="scr_disc_fcf", min_value=-100.0,
                    max_value=100.0, step=5.0, format="%.0f")
    q3.number_input("Discount to fair value (NP) at least", key="scr_disc_np", min_value=-100.0,
                    max_value=100.0, step=5.0, format="%.0f")


def _render_advanced() -> None:
    """Rarely used controls and data upkeep, kept out of the way."""
    a1, a2 = st.columns(2)
    with a1:
        st.slider("Parallel workers", min_value=8, max_value=128, step=8, key="scr_workers",
                  help="How many stocks are fetched at once. More is faster until Yahoo or SEC start throttling.")
        st.checkbox("Show results live while screening", key="scr_live")
    with a2:
        st.checkbox("Include Indian stocks (NSE)", key="scr_india",
                    help="Adds about 2,000 NSE tickers. SEC EDGAR has no data for them, so long-history metrics show n/a.")
        st.checkbox("Re-analyze everything (ignore cached analysis)", key="scr_force",
                    help="Throws away the cached per-stock analysis and starts from scratch. Much slower; "
                         "normally not needed, because changing a threshold re-scores the cache instantly.")
    st.divider()
    m1, m2 = st.columns(2)
    with m1:
        from moatcheck import edgar as _edgar_mod
        zp = _edgar_mod._BULK_ZIP
        zip_ok = zp.exists() and (time.time() - zp.stat().st_mtime) < _edgar_mod._BULK_MAX_AGE
        st.markdown("**SEC bulk data**")
        if zip_ok:
            st.caption(f"Present, {(time.time() - zp.stat().st_mtime) / 86400:.0f} day(s) old. "
                       "Full scans need no per-stock SEC requests.")
        else:
            st.caption("⚠️ Not downloaded. Scans fall back to one SEC request per stock (much slower).")
        if st.button("Download / refresh SEC data (1.4 GB)", key="scr_btn_sec"):
            bar = st.progress(0.0, text="Downloading SEC companyfacts.zip…")
            try:
                _edgar_mod.refresh_bulk_facts(progress=lambda d, t: bar.progress(
                    min(1.0, d / t) if t else 0.0, text=f"Downloading… {d / 1048576:,.0f} MB"))
                bar.empty()
                st.success("SEC bulk data updated.")
            except Exception as e:
                bar.empty()
                st.error(f"Download failed: {e}")
    with m2:
        st.markdown("**Stock list**")
        if UNIVERSE_CACHE.exists():
            age = (time.time() - UNIVERSE_CACHE.stat().st_mtime) / 86400
            st.caption(f"Sector, exchange and size for every ticker, {age:.0f} day(s) old. Refreshes itself weekly.")
        else:
            st.caption("Not built yet. It is built automatically on the first run (about 10 minutes).")
        if st.button("Rebuild stock list now", key="scr_btn_universe",
                     help="Re-fetches sector, exchange and market cap for every ticker."):
            _build_universe_cache(force=True)
            st.cache_data.clear()
            st.rerun()


def _view_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The display table (real numbers, so sorting works) and its column formats."""
    india = df["ticker"].astype(str).str.endswith((".NS", ".BO")).any()
    px_fmt = "%.2f" if india else "$%.2f"
    view = pd.DataFrame({
        "Ticker": df["ticker"],
        "Company": df["company"],
        "Verdict": df["verdict"].map(lambda v: _VERDICT_BADGE.get(v, v)),
        "Score": df["score"],
        "Price": df["stock_price"],
        "Value Price": df["value_price"],
        "MOS Price": df["mos_price"],
        "P/E": df["pe_ratio"],
        "Market Cap (B)": df["market_cap"] / 1e9,
        "Sector": df["sector"],
        "Industry": df["industry"],
        "Exchange": df["exchange"],
    })
    cfg = {
        "Ticker": st.column_config.TextColumn(width="small"),
        "Company": st.column_config.TextColumn(width="medium"),
        "Verdict": st.column_config.TextColumn(width="medium"),
        "Score": st.column_config.ProgressColumn("Score (0-10)", min_value=0.0, max_value=10.0, format="%.1f"),
        "Price": st.column_config.NumberColumn(format=px_fmt),
        "Value Price": st.column_config.NumberColumn(format=px_fmt),
        "MOS Price": st.column_config.NumberColumn(format=px_fmt),
        "P/E": st.column_config.NumberColumn(format="%.1f"),
        "Market Cap (B)": st.column_config.NumberColumn(format="%.2f"),
    }
    return view, cfg


def _render_results(rows: list[ScreenerRow], meta: dict, source: str) -> None:
    df = rows_to_frame(rows)
    ok = df["error"] == ""
    n_buy = int(df["verdict"].isin(["BUY", "BARGAIN BUY"]).sum())
    n_bargain = int((df["verdict"] == "BARGAIN BUY").sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks screened", f"{len(df):,}")
    m2.metric("With a valuation", f"{int(df['value_price'].notna().sum()):,}")
    m3.metric("Buy candidates", f"{n_buy:,}", help="Verdict BUY or BARGAIN BUY: price at or below the MOS buy price.")
    m4.metric("Bargain buys", f"{n_bargain:,}", help="Price at or below half the MOS buy price.")

    st.markdown("#### Results")
    present = [v for v in _MAIN_VERDICTS + ["Error", "Retry"] if (df["verdict"] == v).any()]
    f1, f2, f3, f4 = st.columns([3, 3, 2, 2])
    search = f1.text_input("Search", key="scr_v_search", placeholder="Ticker or company",
                           label_visibility="collapsed")
    verdicts = f2.multiselect("Verdict", options=present, key="scr_v_verdicts", label_visibility="collapsed",
                              placeholder="Verdict")
    min_score = f3.slider("Min score", 0.0, 10.0, step=0.5, key="scr_v_min", help="Hide companies scoring below this.")

    mask = df["verdict"].isin(verdicts) if verdicts else pd.Series(True, index=df.index)
    if min_score > 0:
        mask &= df["score"].fillna(-1) >= min_score
    if search.strip():
        q = search.strip().lower()
        mask &= (df["ticker"].str.lower().str.contains(q, regex=False)
                 | df["company"].str.lower().str.contains(q, regex=False))
    shown = df[mask]

    f4.download_button(
        "⬇️ Download CSV", data=shown.to_csv(index=False).encode("utf-8"),
        file_name=f"screener_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv", width="stretch",
        help="Downloads the rows currently shown, with plain numbers.",
    )
    st.caption(f"Showing **{len(shown):,}** of {len(df):,} stocks"
               + ("" if source == "fresh" else " · loaded from the saved file"))

    if shown.empty:
        st.info("No stocks match the view filters above.")
    else:
        view, cfg = _view_table(shown)
        st.dataframe(view, column_config=cfg, hide_index=True, width="stretch",
                     height=min(680, 36 * len(view) + 42))

    errs = df[~ok]
    if len(errs):
        with st.expander(f"⚠️ {len(errs):,} stocks could not be analyzed"):
            st.caption("Usually funds, trusts or foreign filers with no usable statements. "
                       "\"Retry\" means SEC or Yahoo was busy; run again to retry those.")
            st.dataframe(errs[["ticker", "company", "verdict", "error"]].head(300),
                         hide_index=True, width="stretch", height=300)


def render_stock_screener() -> None:
    """Stock Screener: shows the last saved list, with settings and a single Run button."""
    ss = st.session_state
    st.header("🏦 Stock Screener")
    st.caption("Scores every US-listed stock on the Big 5 growth rates and estimates its value. "
               "Your latest results are saved and shown whenever you open this page.")

    if "screener_results" not in ss:
        loaded = load_saved_results()
        if loaded:
            ss["screener_results"], ss["screener_meta"] = loaded
            ss["screener_source"] = "saved"
    rows = ss.get("screener_results")
    meta = ss.get("screener_meta") or {}
    _init_settings_state(meta)

    status_area = st.container()                    # filled in after the settings, so it can hold the Run button

    with st.expander("⚙️ Screener settings", expanded=not rows):
        _render_settings()
    with st.expander("🔧 Advanced options & data maintenance", expanded=False):
        _render_advanced()

    with status_area:
        s_col, b_col = st.columns([5, 1])
        with s_col:
            if meta.get("finished_at"):
                try:
                    done = datetime.fromisoformat(meta["finished_at"])
                    age_d = (datetime.now() - done).total_seconds() / 86400
                    stale = f" · ⚠️ {age_d:.0f} days old, consider re-running" if age_d > RESULTS_STALE_DAYS else ""
                    st.markdown(f"**Last screened:** {done:%b %d, %Y %H:%M} · "
                                f"{meta.get('n_tickers', len(rows or [])):,} stocks · {_describe_settings(meta)}{stale}")
                except ValueError:
                    st.markdown("**Saved results loaded.**")
            else:
                st.markdown("**No saved results yet.** Click **Run screener**.")
        run_clicked = b_col.button("▶ Run screener", type="primary", width="stretch", key="scr_run")

    if run_clicked:
        bad = [label for label, kmin, kmax in _BIG5_ROWS if ss[kmin] > ss[kmax]]
        if bad:
            st.error("Min is higher than Max for: " + ", ".join(bad) + ". Fix the Big 5 thresholds and run again.")
        else:
            out = _run_screen(_thresholds_from_state())
            if out:
                rows, meta = out
                ss["screener_results"], ss["screener_meta"], ss["screener_source"] = rows, meta, "fresh"

    if not rows:
        st.info("Nothing to show yet. The first run builds the stock list (about 10 minutes) and analyzes each stock; "
                "after that, runs reuse the cached analysis and finish in a few minutes at most.")
        return
    _render_results(rows, meta, ss.get("screener_source", "saved"))
