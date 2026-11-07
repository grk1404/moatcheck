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
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

SCREENER_CACHE = Path("data/screener_results.json")
SCREENER_TTL = 7 * 24 * 3600      # a successful analysis stays valid for 7 days (prices are refreshed every run)
ERROR_TTL = 3 * 24 * 3600         # a hard failure (no SEC filings / bad data) is re-tried after 3 days
CACHE_VERSION = 3                 # bump when scoring/valuation logic changes; older rows are recomputed
SAVE_EVERY = 100                  # write progress to disk every N finished tickers
UNIVERSE_CACHE = Path("data/screener_universe_v2.csv")      # v2: includes NASDAQ + quote type
_OLD_UNIVERSE_CACHE = Path("data/screener_universe_cache.csv")
UNIVERSE_CACHE_TTL = timedelta(days=7)
UNIVERSE_WORKERS = 12
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
        if age < UNIVERSE_CACHE_TTL:
            return pd.read_csv(UNIVERSE_CACHE)

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


def rows_to_dataframe(rows: list[ScreenerRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Company": r.company,
                "Ticker": r.ticker,
                "Sector": r.sector or "n/a",
                "Industry": r.industry or "n/a",
                "Exchange": r.exchange or "n/a",
                "Market Cap": fmt_money(_safe_float(r.market_cap), r.ticker),
                "Wonderfulness": f"{_safe_float(r.score):.1f}/10" if _safe_float(r.score) is not None else "n/a",
                "Stock Price": fmt_price(_safe_float(r.stock_price), r.ticker),
                "Value Price": fmt_price(_safe_float(r.value_price), r.ticker),
                "MOS Price": fmt_price(_safe_float(r.mos_price), r.ticker),
                "P/E": f"{_safe_float(r.pe_ratio):.1f}" if _safe_float(r.pe_ratio) is not None else "n/a",
                "Verdict": r.verdict,
            }
            for r in rows
        ]
    )


def style_screener_table(df: pd.DataFrame):
    def style_row(row):
        style = _VERDICT_COLORS.get(row["Verdict"], "")
        return [style] * len(row)

    return df.style.apply(style_row, axis=1)

def _render_aggrid(df: pd.DataFrame) -> None:
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        filter=True,
        sortable=True,
        resizable=True,
        floatingFilter=True,
    )
    gb.configure_column("Ticker", pinned="left", width=100)
    gb.configure_column("Company", pinned="left", width=200)
    gb.configure_column("Verdict", width=130)

    grid_options = gb.build()
    
    # --- MUTE CONSOLE DURING AGGrid RENDER ---
    # st_aggrid dumps massive amounts of debug info to the terminal. 
    # We temporarily redirect sys.stdout to silence it.
    import sys
    import os
    _original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        AgGrid(
            df,
            gridOptions=grid_options,
            height=600,
            theme="streamlit",
            update_mode=GridUpdateMode.NO_UPDATE,
            allow_unsafe_jscode=False,
        )
    finally:
        # ALWAYS restore stdout, even if AgGrid crashes
        sys.stdout.close()
        sys.stdout = _original_stdout

def display_screener_stats(rows: list[ScreenerRow]):
    col1, col2, col3, col4 = st.columns(4)

    total = len(rows)
    successful = len([r for r in rows if r.error is None])
    bargain_buys = len([r for r in rows if r.verdict == "BARGAIN BUY"])
    buys = len([r for r in rows if r.verdict in ["BARGAIN BUY", "BUY"]])

    with col1:
        st.metric("Total Analyzed", total)
    with col2:
        st.metric("Successful", successful)
    with col3:
        st.metric("Buy Candidates", buys)
    with col4:
        st.metric("Bargain Buys", bargain_buys)

# ==================== MAIN SCREENER UI ====================
def render_stock_screener() -> None:
    """Main screener UI — shows results LIVE as companies are analyzed."""

    st.markdown("### 🏦 Stock Screener - Find Wonderful Businesses")
    st.caption(
      "**Full-universe screening:** Scans every US-listed stock via SEC EDGAR + "
        "yFinance. Results are cached for 7 days — reruns are incremental."
    )

    with st.expander("⚙️ Screener Settings", expanded=True):
        # Elegant single-row filter layout
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Company Size**")
            mcap_preset = st.selectbox(
                "Market Cap Range",
                options=list(MARKET_CAP_RANGES.keys()),
                index=0,
                label_visibility="collapsed",
            )
            min_mcap, max_mcap = MARKET_CAP_RANGES[mcap_preset]

        with col2:
            st.markdown("**Exchange**")
            exchange = st.selectbox(
                "Exchange",
                options=EXCHANGES,
                index=0,
                label_visibility="collapsed",
            )

        with col3:
            st.markdown("**Sector**")
            # Multi-select for sectors
            sector_filter = st.multiselect(
                "Select Sectors",
                options=[s for s in SECTORS if s != "All Sectors"],
                default=[],
                label_visibility="collapsed",
                help="Select one or more sectors. Leave empty for all sectors.",
                placeholder="All Sectors",
            )        
        # Performance settings in a secondary row
        col4, col5 = st.columns([1, 3])
        with col4:
            max_workers = st.slider(
                "Workers",
                min_value=5,
                max_value=128,
                value=DEFAULT_MAX_WORKERS,
                step=8,
                help="Parallel processing speed"
            )        
        with col5:
            show_live_results = st.checkbox(
                "📡 Show LIVE results as they come in",
                value=True,
                help="Display each company immediately after analysis (recommended)"
            )
        include_india = st.checkbox(
            "🇮🇳 Include Indian stocks (NSE)",
            value=False,
            help=(
                "Adds ~2,000 NSE-listed tickers. Note: EDGAR doesn't cover Indian "
                "companies, so the Big 5 will show n/a for long-history metrics. "
                "Technical signals and intrinsic value still work."
            ),
        )

        from moatcheck import edgar as _edgar_mod
        _zip = _edgar_mod._BULK_ZIP
        _zip_ok = _zip.exists() and (time.time() - _zip.stat().st_mtime) < _edgar_mod._BULK_MAX_AGE
        if _zip_ok:
            _age_d = (time.time() - _zip.stat().st_mtime) / 86400
            st.caption(f"⚡ SEC bulk data present ({_age_d:.0f} days old): full scans need no per-stock SEC requests.")
        else:
            st.warning("SEC bulk data not downloaded: scans fall back to one rate-limited SEC request per stock "
                       "(much slower). Download it once (about 1.4 GB, 1-2 minutes).")
        if st.button("⬇️ Download / refresh SEC bulk data"):
            _bar = st.progress(0.0, text="Downloading SEC companyfacts.zip…")
            try:
                _edgar_mod.refresh_bulk_facts(
                    progress=lambda d, t: _bar.progress(min(1.0, d / t) if t else 0.0,
                                                        text=f"Downloading… {d / 1048576:,.0f} MB"))
                _bar.empty()
                st.success("SEC bulk data updated.")
            except Exception as _e:
                _bar.empty()
                st.error(f"Download failed: {_e}")

        force_refresh = st.checkbox(
            "🔄 Force fresh scan (ignore cache)",
            value=False,
            help="Ignore all cached results and re-analyze every ticker from scratch."
        )
        
        if st.button("🔨 Rebuild universe cache", help="Refresh sector/exchange metadata. Takes ~10 min."):
            _build_universe_cache(force=True)
            st.cache_data.clear()
            st.rerun()

        # --- Big 5 growth thresholds ---
        with st.expander("📊 Big 5 growth thresholds", expanded=False):
            st.caption(
                "Set a minimum and maximum acceptable growth rate for each metric. "
                "A company passes a metric only if its CAGR falls within [Min, Max] "
                "for **every** computed window. Changes take effect on the next run."
            )

            tcol1, tcol2, tcol3, tcol4, tcol5 = st.columns(5)

            with tcol1:
                st.markdown("**Revenue**")
                rev_min = st.number_input(
                    "Rev min", min_value=0.0, max_value=100.0,
                    value=10.0, step=1.0, format="%.0f",
                    key="scr_rev_min", label_visibility="collapsed",
                ) / 100
                rev_max = st.number_input(
                    "Rev max", min_value=0.0, max_value=100.0,
                    value=50.0, step=1.0, format="%.0f",
                    key="scr_rev_max", label_visibility="collapsed",
                ) / 100

            with tcol2:
                st.markdown("**EPS**")
                eps_min = st.number_input(
                    "EPS min", min_value=0.0, max_value=100.0,
                    value=10.0, step=1.0, format="%.0f",
                    key="scr_eps_min", label_visibility="collapsed",
                ) / 100
                eps_max = st.number_input(
                    "EPS max", min_value=0.0, max_value=100.0,
                    value=50.0, step=1.0, format="%.0f",
                    key="scr_eps_max", label_visibility="collapsed",
                ) / 100

            with tcol3:
                st.markdown("**Equity**")
                eq_min = st.number_input(
                    "Eq min", min_value=0.0, max_value=100.0,
                    value=10.0, step=1.0, format="%.0f",
                    key="scr_eq_min", label_visibility="collapsed",
                ) / 100
                eq_max = st.number_input(
                    "Eq max", min_value=0.0, max_value=100.0,
                    value=50.0, step=1.0, format="%.0f",
                    key="scr_eq_max", label_visibility="collapsed",
                ) / 100

            with tcol4:
                st.markdown("**OCF**")
                ocf_min = st.number_input(
                    "OCF min", min_value=0.0, max_value=100.0,
                    value=10.0, step=1.0, format="%.0f",
                    key="scr_ocf_min", label_visibility="collapsed",
                ) / 100
                ocf_max = st.number_input(
                    "OCF max", min_value=0.0, max_value=100.0,
                    value=50.0, step=1.0, format="%.0f",
                    key="scr_ocf_max", label_visibility="collapsed",
                ) / 100

            with tcol5:
                st.markdown("**ROIC**")
                roic_min = st.number_input(
                    "ROIC min", min_value=0.0, max_value=100.0,
                    value=10.0, step=1.0, format="%.0f",
                    key="scr_roic_min", label_visibility="collapsed",
                ) / 100
                roic_max = st.number_input(
                    "ROIC max", min_value=0.0, max_value=100.0,
                    value=50.0, step=1.0, format="%.0f",
                    key="scr_roic_max", label_visibility="collapsed",
                ) / 100

        # --- Additional quality filters ---
        with st.expander("📊 Additional quality filters", expanded=False):
            st.caption(
                "One-sided floors. A company passes if its metric is **at or above** "
                "the value entered here. Changes take effect on the next run."
            )

            acol1, acol2, acol3 = st.columns(3)

            with acol1:
                st.markdown("**Cash Conversion ≥**")
                cash_conv_min = st.number_input(
                    "Cash Conversion", min_value=0.0, max_value=500.0,
                    value=80.0, step=5.0, format="%.0f",
                    key="scr_cash_conv", label_visibility="collapsed",
                ) / 100

            with acol2:
                st.markdown("**Discount to FV (FCF) ≥**")
                disc_fcf_min = st.number_input(
                    "Discount FCF", min_value=-100.0, max_value=100.0,
                    value=30.0, step=5.0, format="%.0f",
                    key="scr_disc_fcf", label_visibility="collapsed",
                ) / 100

            with acol3:
                st.markdown("**Discount to FV (NP) ≥**")
                disc_np_min = st.number_input(
                    "Discount NP", min_value=-100.0, max_value=100.0,
                    value=30.0, step=5.0, format="%.0f",
                    key="scr_disc_np", label_visibility="collapsed",
                ) / 100

        # --- Combine into the thresholds dict (after both expanders) ---
        screener_thresholds = {
            "revenue":         {"min": rev_min,       "max": rev_max},
            "eps":             {"min": eps_min,       "max": eps_max},
            "equity":          {"min": eq_min,        "max": eq_max},
            "ocf":             {"min": ocf_min,       "max": ocf_max},
            "roic":            {"min": roic_min,      "max": roic_max},
            "cash_conversion": {"min": cash_conv_min, "max": 999.0},
            "discount_fcf":    {"min": disc_fcf_min,  "max": 999.0},
            "discount_np":     {"min": disc_np_min,   "max": 999.0},
        }

    st.caption(
        "**Active filters:** "
        f"Revenue {screener_thresholds['revenue']['min']*100:.0f}–{screener_thresholds['revenue']['max']*100:.0f}% · "
        f"EPS {screener_thresholds['eps']['min']*100:.0f}–{screener_thresholds['eps']['max']*100:.0f}% · "
        f"Equity {screener_thresholds['equity']['min']*100:.0f}–{screener_thresholds['equity']['max']*100:.0f}% · "
        f"OCF {screener_thresholds['ocf']['min']*100:.0f}–{screener_thresholds['ocf']['max']*100:.0f}% · "
        f"ROIC {screener_thresholds['roic']['min']*100:.0f}–{screener_thresholds['roic']['max']*100:.0f}% · "
        f"Cash Conversion ≥ {screener_thresholds['cash_conversion']['min']*100:.0f}% · "
        f"Discount FCF ≥ {screener_thresholds['discount_fcf']['min']*100:.0f}% · "
        f"Discount NP ≥ {screener_thresholds['discount_np']['min']*100:.0f}%"
    )

    if st.button("🔍 Start Screening", type="primary"):
        start_time = time.time()

        if force_refresh:
            # Nuke both caches
            st.cache_data.clear()                # Streamlit memory cache
            if SCREENER_CACHE.exists():
                SCREENER_CACHE.unlink()          # Disk cache
        
        # ---- Pre-scan filter: only analyze tickers that pass the settings filters ----
        universe_df = load_stock_universe()   # cached DataFrame with sector/exchange/mcap

        if not include_india:
            universe_df = universe_df[
                ~universe_df["ticker"].str.endswith((".NS", ".BO"), na=False)
            ]
        if include_india:
            try:
                india_tickers = get_indian_stock_tickers()
                india_df = pd.DataFrame({
                    "ticker": india_tickers,
                    "exchange": [
                        "NSE" if t.endswith(".NS") else "BSE"
                        for t in india_tickers
                    ],
                    "sector": [""] * len(india_tickers),
                    "market_cap": [0] * len(india_tickers),
                })
                universe_df = pd.concat([universe_df, india_df], ignore_index=True)
            except Exception:
                pass  # if India fetch fails, continue with US only

        if sector_filter:
            universe_df = universe_df[universe_df["sector"].isin(sector_filter)]

        if exchange and exchange != "All Exchanges":
            universe_df = universe_df[universe_df["exchange"] == exchange]

        if min_mcap > 0:
            universe_df = universe_df[universe_df["market_cap"] >= min_mcap]
        if max_mcap < float("inf"):
            universe_df = universe_df[universe_df["market_cap"] <= max_mcap]

        all_tickers = universe_df["ticker"].tolist()
        region = "US + India" if include_india else "US"

        if not all_tickers:
            st.warning("No tickers match the current Company Size / Exchange / Sector filters.")
            st.stop()

        st.info(
            f"📊 **Universe after pre-filters:** {len(all_tickers):,} {region}-listed stocks "
            f"— running scan on the filtered set"
        )

        results = deep_analyze_batch(
            all_tickers,
            max_workers,
            show_live_results,
            thresholds=screener_thresholds,
        )

        elapsed = time.time() - start_time
        st.session_state["screener_results"] = results
        st.session_state["screener_time"] = elapsed
        st.success(f"✅ Scanned **{len(all_tickers):,}** stocks in **{elapsed:.1f}s**")

    # Display final results
    if "screener_results" not in st.session_state:
        st.info("👆 Click **Start Screening** to find wonderful businesses!")
        return

    rows: list[ScreenerRow] = st.session_state["screener_results"]

    if not rows:
        st.warning("No results to display.")
        return

    # Stats
    display_screener_stats(rows)
    
    # All filtering is handled by the results table itself (column filters)
    filtered_rows = rows

    # Show count
    st.caption(f"📊 Showing **{len(filtered_rows)}** of **{len(rows)}** results")
    
    # Display filtered results
    if filtered_rows:
        df_filtered = rows_to_dataframe(filtered_rows)
        _render_aggrid(df_filtered)
    else:
        st.warning("No results match your filters. Try adjusting the criteria above.")
        df_filtered = pd.DataFrame()

    # ========== EXPORT AND CLEAR ==========
    col_export, col_clear = st.columns(2)
    
    with col_export:
        if st.button("📥 Export to CSV"):
            # Export the filtered results
            if filtered_rows:
                csv_data = rows_to_dataframe(filtered_rows)
            else:
                csv_data = rows_to_dataframe(rows)
            csv = csv_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"screener_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    with col_clear:
        if st.button("🗑️ Clear Results"):
            st.session_state.pop("screener_results", None)
            st.session_state.pop("screener_time", None)
            st.rerun()

    # ========== ERRORS ==========
    errors = [r for r in rows if r.error]
    if errors and st.checkbox(f"⚠️ Show {len(errors)} errors"):
        with st.expander(f"❌ {len(errors)} ticker(s) had errors"):
            for r in errors[:50]:
                st.markdown(f"**{r.ticker}** — {r.error}")
            if len(errors) > 50:
                st.markdown(f"... and {len(errors) - 50} more errors")