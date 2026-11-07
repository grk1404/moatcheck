"""SEC EDGAR companyfacts fetcher — 10+ years of annual 10-K XBRL data.

Free, no API key. SEC only requires a User-Agent header identifying the caller.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import requests


class EdgarError(Exception):
    """Ticker not covered by EDGAR (foreign filer, delisted, or invalid)."""


class EdgarTransientError(Exception):
    """SEC was unreachable or throttled us (429/5xx/timeout) after retries.

    Deliberately NOT an EdgarError: callers fall back to Yahoo-only data on EdgarError, and
    silently scoring a company on 4 years of Yahoo data because SEC hiccuped is worse than
    reporting the failure and retrying later.
    """


# SEC's edge requires an email address in the User-Agent (they reject UAs that
# look like URLs). The email is a contact address per SEC's fair-access policy;
# users deploying this can override via the moat_SEC_UA env var.
SEC_UA = os.environ.get("moat_SEC_UA", "moatcheck contact@moatcheck.com")
TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

_CACHE_DIR = Path.home() / ".moat_cache"
_TICKERS_TTL = 30 * 24 * 3600  # 30 days
_FACTS_TTL = 24 * 3600         # 1 day

# SEC publishes every company's XBRL facts as one nightly zip (~1.4 GB). Reading one company
# from it takes ~35 ms with no network call, which removes the per-ticker request (and its
# rate limit) from full-universe scans. Created by refresh_bulk_facts().
BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
_BULK_ZIP = _CACHE_DIR / "companyfacts.zip"
_BULK_MAX_AGE = 28 * 24 * 3600   # ignore a zip older than this and fall back to HTTP

IFRS_CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "Revenue",
        "RevenueFromContractsWithCustomers",
        "RevenueFromSaleOfGoods",
        "RevenueFromRenderingOfServices",
    ],
    "net_income": [
        "ProfitLoss",
        "ProfitLossAttributableToOwnersOfParent",
    ],
    "eps": [
        "DilutedEarningsLossPerShare",
        "BasicEarningsLossPerShare",
        "EarningsPerShareDiluted",
        "EarningsPerShareBasic",
    ],
    "shares": [
        "NumberOfSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfOrdinarySharesOutstanding",
    ],
    "equity": [
        "Equity",
        "EquityAttributableToOwnersOfParent",
    ],
    "long_term_debt": [
        "NoncurrentPortionOfLongtermBorrowings",
        "LongtermBorrowings",
    ],
    "ocf": [
        "CashFlowsFromUsedInOperatingActivities",
        "CashFlowsFromUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PurchaseOfPropertyPlantAndEquipment",
        "AcquisitionOfPropertyPlantAndEquipment",
        "PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwill",
    ],
    "ebit": [
        "ProfitLossFromOperatingActivities",
    ],
    "tax_provision": [
        "IncomeTaxExpenseContinuingOperations",
        "TaxExpenseIncome",
    ],
    "pretax_income": [
        "ProfitLossBeforeTax",
    ],
}

CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",  # CRWD and others
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "eps": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasic",
    ],
    "shares": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
        "SharesOutstanding",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "CommonStockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterestAndParent",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
    "ocf": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
        "PaymentsToAcquireProductiveAssetsAndIntangibleAssets",
        "PaymentsToAcquireOtherProductiveAssets",
        "PaymentsForCapitalImprovements",
        "PaymentsToAcquireMachineryAndEquipment",
    ],
    "ebit": [
        "OperatingIncomeLoss",
    ],
    "tax_provision": [
        "IncomeTaxExpenseBenefit",
    ],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
}

def extract_concept_series(us_gaap_facts: dict, concept_key: str) -> pd.Series:
    """
    Extracts an annual time series (by fiscal year) for a given concept key.
    Iterates through candidate tags in CONCEPTS[concept_key] until valid data is found.
    """
    tags = CONCEPTS.get(concept_key, [])
    for tag in tags:
        if tag not in us_gaap_facts:
            continue

        units = us_gaap_facts[tag].get("units", {})
        if not units:
            continue

        unit_key = next(iter(units.keys()))
        items = units[unit_key]

        rows = []
        for item in items:
            form = item.get("form", "")
            fp = item.get("fp", "")
            if form == "10-K" or fp == "FY":
                fy = item.get("fy")
                val = item.get("val")
                if fy is not None and val is not None:
                    rows.append({"fy": int(fy), "val": float(val)})

        if rows:
            df = pd.DataFrame(rows).drop_duplicates(subset=["fy"], keep="last")
            return df.set_index("fy")["val"].sort_index()

    return pd.Series(dtype=float)


def compute_bvps_series(us_gaap_facts: dict) -> pd.Series:
    """
    Calculates Book Value Per Share (BVPS) by taking Total Equity / Shares.
    Handles missing values and restricts output to the last 10 completed fiscal years.
    """
    equity_series = extract_concept_series(us_gaap_facts, "equity")
    shares_series = extract_concept_series(us_gaap_facts, "shares")

    if equity_series.empty or shares_series.empty:
        return pd.Series(dtype=float)

    common_years = equity_series.index.intersection(shares_series.index)
    if len(common_years) == 0:
        return pd.Series(dtype=float)

    bvps = equity_series.loc[common_years] / shares_series.loc[common_years]
    
    # Replace negative or zero BVPS with NaN to prevent invalid CAGR math
    bvps = bvps.apply(lambda v: v if v > 0 else np.nan)
    
    return bvps.tail(10)

@dataclass
class EdgarData:
    ticker: str
    cik: str
    entity_name: str
    revenue: pd.Series
    net_income: pd.Series
    eps: pd.Series
    shares: pd.Series
    equity: pd.Series
    bvps: pd.Series
    long_term_debt: pd.Series
    ocf: pd.Series
    capex: pd.Series
    fcf: pd.Series
    ebit: pd.Series
    tax_rate: pd.Series
    years_available: int = 0
    currency: str = "USD"          # reporting currency detected from XBRL units
    taxonomy: str = "us-gaap"      # "us-gaap" or "ifrs-full"
    raw: dict = field(default_factory=dict)
    split_note: str = ""           # non-empty when per-share history was split-adjusted


def _cache_read(path: Path, ttl: int) -> dict | None:
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _cache_write(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        tmp.replace(path)
    except OSError:
        # Cache is best-effort; if we can't write we still return the data.
        pass


class _RateLimiter:
    """Spaces calls at least 1/per_second apart across all threads (SEC allows ~10 req/s)."""

    def __init__(self, per_second: float):
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self._interval
        if start > now:
            time.sleep(start - now)


_SEC_LIMITER = _RateLimiter(8.0)


def set_sec_rate(per_second: float) -> None:
    """Set this process's share of the SEC request budget (SEC allows ~10 requests/second in total;
    when a scan runs N worker processes each gets 8/N)."""
    global _SEC_LIMITER
    _SEC_LIMITER = _RateLimiter(max(0.5, float(per_second)))


def _sec_get(url: str) -> dict:
    """GET a JSON URL from SEC: shared rate limit, retries with backoff on 429/5xx/timeouts."""
    last = "no response"
    for attempt in range(5):
        _SEC_LIMITER.wait()
        try:
            r = requests.get(url, headers={"User-Agent": SEC_UA, "Accept": "application/json"}, timeout=30)
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(min(30.0, 2 ** attempt + random.random()))
            continue
        if r.status_code == 404:
            raise EdgarError(f"SEC returned 404 for {url}")
        if r.status_code in (429, 500, 502, 503, 504):
            last = f"HTTP {r.status_code}"
            ra = r.headers.get("Retry-After", "")
            time.sleep(min(60.0, float(ra) if ra.isdigit() else 2 ** attempt + random.random()))
            continue
        r.raise_for_status()
        return r.json()
    raise EdgarTransientError(f"SEC request failed after retries ({last}): {url}")


_TICKER_MAP: dict[str, dict] | None = None


def _load_ticker_map() -> dict[str, dict]:
    """Ticker -> {cik, name} lookup. Cached in-memory and on disk."""
    global _TICKER_MAP
    if _TICKER_MAP is not None:
        return _TICKER_MAP

    cache = _CACHE_DIR / "company_tickers.json"
    data = _cache_read(cache, _TICKERS_TTL)
    if data is None:
        data = _sec_get(TICKER_URL)
        _cache_write(cache, data)

    result: dict[str, dict] = {}
    for entry in data.values():
        t = str(entry.get("ticker", "")).upper()
        if t:
            result[t] = {
                "cik": int(entry["cik_str"]),
                "name": entry.get("title", ""),
            }
    _TICKER_MAP = result
    return result


def _resolve_ticker(ticker: str) -> tuple[int, str]:
    tmap = _load_ticker_map()
    entry = tmap.get(ticker.upper())
    if not entry:
        raise EdgarError(
            f"Ticker {ticker!r} not found in SEC EDGAR. "
            "EDGAR only covers US-listed companies (foreign issuers file 20-F)."
        )
    return entry["cik"], entry["name"]


_ZIP_LOCK = threading.Lock()
_ZIP_STATE: dict = {"zf": None, "mtime": None}


def _bulk_zip():
    """Open handle on the bulk companyfacts zip, or None if absent, stale or unreadable."""
    try:
        st = _BULK_ZIP.stat()
    except OSError:
        return None
    if time.time() - st.st_mtime > _BULK_MAX_AGE:
        return None
    with _ZIP_LOCK:
        if _ZIP_STATE["zf"] is None or _ZIP_STATE["mtime"] != st.st_mtime:
            try:
                _ZIP_STATE["zf"] = zipfile.ZipFile(_BULK_ZIP)
                _ZIP_STATE["mtime"] = st.st_mtime
            except (zipfile.BadZipFile, OSError):
                _ZIP_STATE["zf"] = None
                return None
        return _ZIP_STATE["zf"]


def refresh_bulk_facts(progress=None) -> Path:
    """Download SEC's nightly companyfacts.zip (~1.4 GB, ~1 min). Atomic: the old file stays
    usable until the new one is complete. `progress(done_bytes, total_bytes)` is optional."""
    _BULK_ZIP.parent.mkdir(parents=True, exist_ok=True)
    tmp = _BULK_ZIP.with_suffix(".zip.tmp")
    done = 0
    with requests.get(BULK_URL, headers={"User-Agent": SEC_UA}, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    if total and done != total:
        tmp.unlink(missing_ok=True)
        raise EdgarTransientError(f"bulk download incomplete ({done} of {total} bytes)")
    with _ZIP_LOCK:
        if _ZIP_STATE["zf"] is not None:
            _ZIP_STATE["zf"].close()
            _ZIP_STATE["zf"] = None
        tmp.replace(_BULK_ZIP)
    return _BULK_ZIP


def _load_facts(cik: int, persist: bool = True) -> dict:
    """Company facts: fresh per-ticker cache, else the bulk zip, else a rate-limited HTTP call.

    persist=False skips writing the per-ticker cache (~3 MB each; a full-universe scan would
    otherwise write tens of GB)."""
    cache = _CACHE_DIR / "edgar" / f"CIK{cik:010d}.json"
    data = _cache_read(cache, _FACTS_TTL)
    if data is not None:
        return data
    zf = _bulk_zip()
    if zf is not None:
        try:
            with _ZIP_LOCK:
                raw = zf.read(f"CIK{cik:010d}.json")
            return json.loads(raw)
        except KeyError:
            pass                     # company not in the bulk file -> ask SEC directly
        except (zipfile.BadZipFile, OSError, ValueError):
            pass
    url = FACTS_URL.format(cik=f"{cik:010d}")
    data = _sec_get(url)
    if persist:
        _cache_write(cache, data)
    return data


def _series_and_filed(
    taxonomy: dict,
    concepts: list[str],
    form_prefixes: tuple[str, ...] = ("10-K",),
    unit_priority: tuple[str, ...] = ("USD", "USD/shares", "shares", "pure"),
) -> tuple[pd.Series, dict[int, str]]:
    """Merge multiple XBRL concepts into one year-indexed Series plus {year: filing date}.

    form_prefixes: which SEC form types count as "annual" for this taxonomy.
        US GAAP filers use 10-K. IFRS filers use 20-F.
    unit_priority: preferred unit keys, in order. Falls back to the first
        available unit if none of these match.

    Duration facts must cover a full fiscal year (340-380 days). Without this a 10-K that
    also tags a quarter (COST FY2017: 83 days, NFLX FY2016: 90 days) can win as the "annual"
    value. Point-in-time facts (equity, debt) have no start date and are unaffected.
    """
    year_value: dict[int, tuple[str, float]] = {}

    for concept in concepts:
        node = taxonomy.get(concept)
        if not node or "units" not in node:
            continue

        units = node["units"]
        unit_key = None
        for key in unit_priority:
            if key in units:
                unit_key = key
                break
        if unit_key is None:
            unit_key = next(iter(units.keys()), None)
        if unit_key is None:
            continue

        for entry in units[unit_key]:
            fp = entry.get("fp")
            if fp and fp != "FY":
                continue
            form = str(entry.get("form", ""))
            if not any(form.startswith(p) for p in form_prefixes):
                continue
            end = entry.get("end")
            if not end:
                continue
            start = entry.get("start")
            if start:
                try:
                    days = (date.fromisoformat(end) - date.fromisoformat(start)).days
                except ValueError:
                    continue
                if not 340 <= days <= 380:
                    continue
            try:
                year = int(end[:4])
            except (TypeError, ValueError):
                continue

            filed = entry.get("filed", "")
            val = entry.get("val")
            if val is None:
                continue

            if year not in year_value:
                year_value[year] = (filed, float(val))
            else:
                prev_filed, _ = year_value[year]
                if filed > prev_filed:
                    year_value[year] = (filed, float(val))

    if not year_value:
        return pd.Series(dtype=float), {}

    series = pd.Series({y: v for y, (_, v) in year_value.items()}, dtype=float).sort_index()
    return series, {y: f for y, (f, _) in year_value.items()}


def _series_for_concepts(*args, **kwargs) -> pd.Series:
    return _series_and_filed(*args, **kwargs)[0]


def fetch_edgar(ticker: str, persist: bool = True, adjust_splits: bool = True) -> EdgarData:
    """Pull annual facts (10-K US GAAP or 20-F IFRS) for a ticker from SEC EDGAR."""
    if not ticker or not ticker.strip():
        raise EdgarError("Please provide a ticker symbol.")

    cik, name = _resolve_ticker(ticker.strip().upper())
    facts = _load_facts(cik, persist=persist)
    all_facts = facts.get("facts", {})

    gaap = all_facts.get("us-gaap")
    ifrs = all_facts.get("ifrs-full")

    # Dual filers (Toyota, some Chinese ADRs) often have a token us-gaap
    # section that's just cover-page facts. Prefer the richer one.
    if gaap and ifrs and len(ifrs) > len(gaap) * 2:
        gaap = None

    if gaap:
        taxonomy, concepts = gaap, CONCEPTS
        form_prefixes = ("10-K",)
        unit_priority = ("USD", "USD/shares", "shares", "pure")
        taxonomy_name = "us-gaap"
    elif ifrs:
        taxonomy, concepts = ifrs, IFRS_CONCEPTS
        form_prefixes = ("20-F",)
        unit_priority = (
            "EUR", "USD", "GBP", "JPY", "CHF", "CAD",
            "EUR/shares", "USD/shares", "GBP/shares", "JPY/shares", "CHF/shares",
            "shares", "pure",
        )
        taxonomy_name = "ifrs-full"
    else:
        raise EdgarError(
            f"CIK {cik} has neither us-gaap nor ifrs-full facts."
        )

    # Detect the reporting currency from the first unit key we find in the
    # taxonomy. For US GAAP filers this is "USD". For IFRS filers it's the
    # home currency (EUR for Ferrari, JPY for Toyota, GBP for Shell, etc.).
    _currency = "USD" if taxonomy_name == "us-gaap" else None
    for _node in taxonomy.values():
        _units = (_node or {}).get("units") or {}
        for _k in _units:
            if len(_k) == 3 and _k.isalpha() and _k.isupper():
                _currency = _k
                break
        if _currency:
            break
    _currency = _currency or "USD"

    def _series(key: str) -> pd.Series:
        return _series_for_concepts(
            taxonomy, concepts[key],
            form_prefixes=form_prefixes,
            unit_priority=unit_priority,
        )

    def _series_f(key: str):
        return _series_and_filed(
            taxonomy, concepts[key],
            form_prefixes=form_prefixes,
            unit_priority=unit_priority,
        )

    revenue = _series("revenue")
    net_income = _series("net_income")
    eps, eps_filed = _series_f("eps")
    shares, shares_filed = _series_f("shares")

    # Put per-share history on today's share basis (EDGAR only restates ~3 years after a split).
    split_note = ""
    if adjust_splits and taxonomy_name == "us-gaap":
        try:
            from .splits import adjust_edgar_per_share
            eps, shares, split_note = adjust_edgar_per_share(ticker, eps, shares, eps_filed, shares_filed)
        except Exception:
            split_note = "split adjustment skipped (error)"
    equity = _series("equity")

    # Fallback: derive EPS from net income / shares when the filer doesn't
    # tag it directly. Shell files ProfitLoss but no EPS concept.
    if eps.empty and not net_income.empty and not shares.empty:
        aligned_eps = pd.concat([net_income, shares], axis=1, join="inner")
        aligned_eps.columns = ["ni", "sh"]
        aligned_eps = aligned_eps[aligned_eps["sh"] > 0]
        eps = aligned_eps["ni"] / aligned_eps["sh"]

    if not equity.empty and not shares.empty:
        aligned_bvps = pd.concat([equity, shares], axis=1, join="inner")
        aligned_bvps.columns = ["equity", "shares"]
        aligned_bvps = aligned_bvps[aligned_bvps["shares"] > 0]
        bvps = aligned_bvps["equity"] / aligned_bvps["shares"]
    else:
        bvps = pd.Series(dtype=float)

    long_term_debt = _series("long_term_debt")
    ocf = _series("ocf")
    capex = _series("capex")
    ebit = _series("ebit")
    tax_provision = _series("tax_provision")
    pretax_income = _series("pretax_income")

    if not ocf.empty and not capex.empty:
        aligned = pd.concat([ocf, capex], axis=1, join="inner")
        aligned.columns = ["ocf", "capex"]
        fcf = aligned["ocf"] - aligned["capex"].abs()
    else:
        fcf = pd.Series(dtype=float)

    if not tax_provision.empty and not pretax_income.empty:
        aligned = pd.concat([tax_provision, pretax_income], axis=1, join="inner")
        aligned.columns = ["tax", "pretax"]
        aligned = aligned[aligned["pretax"] != 0]
        tax_rate = (aligned["tax"] / aligned["pretax"]).clip(lower=0, upper=0.5)
    else:
        tax_rate = pd.Series(dtype=float)

    if revenue.empty and net_income.empty and equity.empty:
        raise EdgarError(
            f"No usable {taxonomy_name} facts extracted for {ticker} (CIK {cik})."
        )

    return EdgarData(
        ticker=ticker.upper(),
        cik=f"{cik:010d}",
        entity_name=name,
        revenue=revenue,
        net_income=net_income,
        eps=eps,
        shares=shares,
        equity=equity,
        bvps=bvps,
        long_term_debt=long_term_debt,
        ocf=ocf,
        capex=capex,
        fcf=fcf,
        ebit=ebit,
        tax_rate=tax_rate,
        years_available=(int(revenue.shape[0]) if not revenue.empty else int(net_income.shape[0]) if not net_income.empty else 0),
        currency=_currency,
        taxonomy=taxonomy_name,
        split_note=split_note,
    )
