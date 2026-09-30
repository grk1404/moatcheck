"""SEC EDGAR companyfacts fetcher — 10+ years of annual 10-K XBRL data.

Free, no API key. SEC only requires a User-Agent header identifying the caller.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
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


def _series_for_concepts(
    taxonomy: dict,
    concepts: list[str],
    form_prefixes: tuple[str, ...] = ("10-K",),
    unit_priority: tuple[str, ...] = ("USD", "USD/shares", "shares", "pure"),
) -> pd.Series:
    """Merge multiple XBRL concepts into one year-indexed pandas Series.

    form_prefixes: which SEC form types count as "annual" for this taxonomy.
        US GAAP filers use 10-K. IFRS filers use 20-F.
    unit_priority: preferred unit keys, in order. Falls back to the first
        available unit if none of these match.
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
        return pd.Series(dtype=float)

    return pd.Series({y: v for y, (_, v) in year_value.items()}, dtype=float,).sort_index()


def fetch_edgar(ticker: str) -> EdgarData:
    """Pull annual facts (10-K US GAAP or 20-F IFRS) for a ticker from SEC EDGAR."""
    if not ticker or not ticker.strip():
        raise EdgarError("Please provide a ticker symbol.")

    cik, name = _resolve_ticker(ticker.strip().upper())
    facts = _load_facts(cik)
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

    revenue = _series("revenue")
    net_income = _series("net_income")
    eps = _series("eps")
    shares = _series("shares")
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
    )
