"""SEC EDGAR companyfacts fetcher — 10+ years of annual 10-K XBRL data.

Free, no API key. SEC only requires a User-Agent header identifying the caller.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests


class EdgarError(Exception):
    """Ticker not covered by EDGAR (foreign filer, delisted, or invalid)."""


# SEC's edge requires an email address in the User-Agent (they reject UAs that
# look like URLs). The email is a contact address per SEC's fair-access policy;
# users deploying this can override via the moat_SEC_UA env var.
SEC_UA = os.environ.get("moat_SEC_UA", "moatcheck contact@moatcheck.com")
TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

_CACHE_DIR = Path.home() / ".moat_cache"
_TICKERS_TTL = 30 * 24 * 3600  # 30 days
_FACTS_TTL = 24 * 3600         # 1 day


CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
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
    raw: dict = field(default_factory=dict)


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


def _sec_get(url: str) -> dict:
    """GET a JSON URL from SEC with the required User-Agent."""
    r = requests.get(url, headers={"User-Agent": SEC_UA, "Accept": "application/json"}, timeout=15)
    if r.status_code == 404:
        raise EdgarError(f"SEC returned 404 for {url}")
    r.raise_for_status()
    return r.json()


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


def _load_facts(cik: int) -> dict:
    cache = _CACHE_DIR / "edgar" / f"CIK{cik:010d}.json"
    data = _cache_read(cache, _FACTS_TTL)
    if data is not None:
        return data
    url = FACTS_URL.format(cik=f"{cik:010d}")
    data = _sec_get(url)
    _cache_write(cache, data)
    return data


def _series_for_concepts(gaap: dict, concepts: list[str]) -> pd.Series:
    """Merge multiple XBRL concepts into one year-indexed pandas Series.

    Strategy:
      - For each concept, look at annual 10-K entries (fp='FY', form starts with '10-K').
      - Bucket entries by the *year of their 'end' date* (fiscal-year end), not by 'fy'
        — because a later 10-K restates prior years under a different 'fy' label.
      - Within a bucket, keep the entry with the latest 'filed' date (most recent restatement).
      - Merge concepts left-to-right in the list; later concepts do NOT overwrite
        earlier ones for a given year — the FIRST concept in the list wins if it has
        data, which lets callers prioritize (e.g. ASC-606 concept before old Revenues).
    """
    year_value: dict[int, tuple[str, float]] = {}  # year -> (filed_date, value)

    for concept in concepts:
        node = gaap.get(concept)
        if not node or "units" not in node:
            continue

        # Prefer USD, but fall back to any single-unit set (EPS uses USD/shares, shares uses "shares").
        units = node["units"]
        unit_key = None
        for key in ("USD", "USD/shares", "shares", "pure"):
            if key in units:
                unit_key = key
                break
        if unit_key is None:
            unit_key = next(iter(units.keys()), None)
        if unit_key is None:
            continue

        for entry in units[unit_key]:
            # Allow 'FY' or missing/empty fp tags on 10-K forms
            fp = entry.get("fp")
            if fp and fp != "FY":
                continue
            form = str(entry.get("form", ""))
            if not form.startswith("10-K"):
                continue
            end = entry.get("end")
            if not end:
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
                # Same year already recorded — keep the entry from the later filing.
                # Break ties in favor of the concept we saw first (earlier in list),
                # which is what the outer loop naturally gives us.
                prev_filed, _ = year_value[year]
                if filed > prev_filed:
                    year_value[year] = (filed, float(val))

    if not year_value:
        return pd.Series(dtype=float)

    s = pd.Series(
        {y: v for y, (_, v) in year_value.items()},
        dtype=float,
    ).sort_index()
    return s


def fetch_edgar(ticker: str) -> EdgarData:
    """Pull 10-K annual facts for a US ticker from SEC EDGAR."""
    if not ticker or not ticker.strip():
        raise EdgarError("Please provide a ticker symbol.")

    cik, name = _resolve_ticker(ticker.strip().upper())
    facts = _load_facts(cik)
    gaap = facts.get("facts", {}).get("us-gaap")
    if not gaap:
        raise EdgarError(f"CIK {cik} has no us-gaap facts (likely non-standard filer).")

    revenue = _series_for_concepts(gaap, CONCEPTS["revenue"])
    net_income = _series_for_concepts(gaap, CONCEPTS["net_income"])
    eps = _series_for_concepts(gaap, CONCEPTS["eps"])
    shares = _series_for_concepts(gaap, CONCEPTS["shares"])
    equity = _series_for_concepts(gaap, CONCEPTS["equity"])
    if not equity.empty and not shares.empty:
        aligned_bvps = pd.concat([equity, shares], axis=1, join="inner")
        aligned_bvps.columns = ["equity", "shares"]
        aligned_bvps = aligned_bvps[aligned_bvps["shares"] > 0]
        bvps = aligned_bvps["equity"] / aligned_bvps["shares"]
    else:
        bvps = pd.Series(dtype=float)
    long_term_debt = _series_for_concepts(gaap, CONCEPTS["long_term_debt"])
    ocf = _series_for_concepts(gaap, CONCEPTS["ocf"])
    capex = _series_for_concepts(gaap, CONCEPTS["capex"])
    ebit = _series_for_concepts(gaap, CONCEPTS["ebit"])
    tax_provision = _series_for_concepts(gaap, CONCEPTS["tax_provision"])
    pretax_income = _series_for_concepts(gaap, CONCEPTS["pretax_income"])

    # Derived: FCF = OCF - |CapEx|
    if not ocf.empty and not capex.empty:
        aligned = pd.concat([ocf, capex], axis=1, join="inner")
        aligned.columns = ["ocf", "capex"]
        fcf = aligned["ocf"] - aligned["capex"].abs()
    else:
        fcf = pd.Series(dtype=float)

    # Derived: effective tax rate, clipped to [0, 0.5]
    if not tax_provision.empty and not pretax_income.empty:
        aligned = pd.concat([tax_provision, pretax_income], axis=1, join="inner")
        aligned.columns = ["tax", "pretax"]
        aligned = aligned[aligned["pretax"] != 0]
        tax_rate = (aligned["tax"] / aligned["pretax"]).clip(lower=0, upper=0.5)
    else:
        tax_rate = pd.Series(dtype=float)

    if revenue.empty and net_income.empty and equity.empty:
        raise EdgarError(f"No usable us-gaap facts extracted for {ticker} (CIK {cik}).")

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
        years_available=int(revenue.shape[0]) if not revenue.empty else int(net_income.shape[0]),
    )
