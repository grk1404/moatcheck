"""Hybrid data source: SEC EDGAR for 10+ years of history + yfinance for snapshot.

The yfinance fetcher is fast but only returns ~4 years of annual statements.
EDGAR returns 15-20 years of official 10-K XBRL data. This module glues them:
history series come from EDGAR; current price / market cap / TTM P/E / dividend
yield / analyst growth come from yfinance.

Falls back to yfinance-only if EDGAR doesn't have the ticker (foreign filers,
recent IPOs before their first 10-K, etc.).
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

from .edgar import EdgarError, fetch_edgar
from .fetcher import Financials, FetchError, fetch as yf_fetch
import xml.etree.ElementTree as ET

_ECB_CACHE: dict[str, tuple[float, datetime]] = {}

_FX_CACHE: dict[str, tuple[float, datetime]] = {}
_FX_FALLBACK = {"EUR": 1.17, "GBP": 1.27, "JPY": 0.0067, "CHF": 1.12, "CAD": 0.73}

import xml.etree.ElementTree as ET

_ECB_CACHE: dict[str, tuple[float, datetime]] = {}


def _fx_rate_to_usd(from_currency: str | None) -> float:
    """Return USD per 1 unit of `from_currency`, using ECB daily reference rates.

    ECB publishes EUR-based rates (EUR -> XXX). To get XXX -> USD, we invert:
        XXX -> USD = (EUR -> USD) / (EUR -> XXX)
    Falls back to a hardcoded approximate rate if the ECB fetch fails.
    """
    if not from_currency or from_currency.upper() == "USD":
        return 1.0
    ccy = from_currency.upper()

    now = datetime.now()
    cached = _ECB_CACHE.get("_all")
    if cached and (now - cached[1]) < timedelta(hours=12):
        rates = cached[0]
    else:
        rates = {}
        try:
            import requests
            r = requests.get(
                "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml",
                timeout=10,
                headers={"User-Agent": "moatcheck contact@moatcheck.com"},
            )
            r.raise_for_status()
            root = ET.fromstring(r.content)
            # ECB XML uses namespaced Cube elements with currency + rate attributes
            for cube in root.iter():
                c = cube.attrib.get("currency")
                r_ = cube.attrib.get("rate")
                if c and r_:
                    try:
                        rates[c] = float(r_)
                    except ValueError:
                        pass
        except Exception:
            rates = {}
        _ECB_CACHE["_all"] = (rates, now)

    eur_to_usd = rates.get("USD")           # ECB base is EUR
    eur_to_target = rates.get(ccy)

    if eur_to_usd and eur_to_target and eur_to_target > 0:
        return eur_to_usd / eur_to_target

    # Fallback
    return _FX_FALLBACK.get(ccy, 1.0)

def _edgar_reporting_currency(edgar) -> str | None:
    """Detect the reporting currency from the raw EDGAR facts.

    Looks for the first IFRS or US-GAAP concept that has a currency-coded
    unit key (EUR, GBP, JPY, etc.) and returns it.
    """
    raw = edgar.raw or {}
    facts = raw.get("facts", {}) or {}
    for taxonomy in ("ifrs-full", "us-gaap"):
        tax = facts.get(taxonomy) or {}
        for node in tax.values():
            units = (node or {}).get("units") or {}
            for key in units:
                # Look for a pure 3-letter currency code
                if len(key) == 3 and key.isalpha() and key.isupper():
                    return key
    return None

def fetch(ticker: str) -> Financials:
    """Return a Financials object combining EDGAR history + yfinance snapshot.

    Behavior:
      - EDGAR success + yfinance success      → merged (history from EDGAR)
      - EDGAR success + yfinance fail         → EDGAR history + snapshot from .info
                                                (foreign filers: RACE, SHEL, TM, etc.)
      - EDGAR success + snapshot also fails   → EDGAR history only, no price/valuation
      - EDGAR fail    + yfinance success      → yfinance-only, ~4yr history. Warning banner.
      - EDGAR fail    + yfinance fail         → propagate FetchError.
    """
    import numpy as np
    import pandas as pd

    # ---- Snapshot: yfinance first, then a minimal .info fallback ----
    snapshot: Financials | None = None
    snapshot_err: Exception | None = None
    try:
        snapshot = yf_fetch(ticker)
    except FetchError as e:
        snapshot_err = e
        # Foreign filers (20-F) don't expose income_stmt via yfinance, so
        # yf_fetch raises — but quote data (price, cap, beta, analyst) is
        # still available on Yahoo. Build a minimal snapshot from .info.
        try:
            from data_provider import get_ticker

            yft = get_ticker(ticker)
            info = yft.info or {}
            fi = yft.fast_info

            _price = fi.get("last_price") or fi.get("lastPrice")
            if _price is not None and isinstance(_price, float) and np.isnan(_price):
                _price = None
            _cap = fi.get("market_cap") or fi.get("marketCap")
            _cap = float(_cap) if _cap is not None else None

            _dy = info.get("dividendYield")
            _dy = float(_dy) / 100.0 if _dy is not None else None

            empty = pd.Series(dtype=float)
            snapshot = Financials(
                ticker=ticker.upper(),
                revenue=empty, net_income=empty, eps=empty, equity=empty,
                shares=empty, operating_cash_flow=empty, capex=empty,
                free_cash_flow=empty, long_term_debt=empty, tax_rate=empty,
                ebit=empty,
                current_price=float(_price) if _price is not None else None,
                market_cap=_cap,
                pe_ratio_ttm=info.get("trailingPE"),
                analyst_5yr_growth=(
                    info.get("earningsGrowth") or info.get("revenueGrowth")
                ),
                shares_outstanding=info.get("sharesOutstanding"),
                dividend_yield=_dy,
                beta=info.get("beta"),
                book_value_per_share=info.get("bookValue"),
                company_name=(
                    info.get("longName") or info.get("shortName") or ticker
                ),
                exchange=(
                    info.get("exchange", "") or info.get("fullExchangeName", "")
                ),
                data_source="yfinance-snapshot",
            )
        except Exception:
            snapshot = None  # truly no snapshot; EDGAR-only branch below

    # ---- EDGAR history ----
    edgar = None
    edgar_err: Exception | None = None
    try:
        edgar = fetch_edgar(ticker)
    except EdgarError as e:
        edgar_err = e

    # ---- Both sources failed ----
    if snapshot is None and edgar is None:
        raise FetchError(str(snapshot_err) if snapshot_err else str(edgar_err))

    # ---- Merge: EDGAR history + yfinance snapshot ----
    if edgar is not None and snapshot is not None:
        # ---- FX: convert EDGAR series into USD ----
        # For US-listed names this keeps EPS, BVPS, and P/E consistent with
        # the USD price from yfinance. Non-USD filers (RACE in EUR, TM in JPY,
        # SHEL in GBP, etc.) are normalized at this step.
        rate = _fx_rate_to_usd(getattr(edgar, "currency", "USD"))

        def _conv(s: pd.Series) -> pd.Series:
            return s * rate if not s.empty else s

        rev    = _conv(edgar.revenue)
        ni     = _conv(edgar.net_income)
        eps_s  = _conv(edgar.eps)
        eq     = _conv(edgar.equity)
        ocf_s  = _conv(edgar.ocf)
        cpx    = _conv(edgar.capex)
        fcf_s  = _conv(edgar.fcf)
        debt   = _conv(edgar.long_term_debt)
        ebit_s = _conv(edgar.ebit)
        # tax_rate is a unit-less ratio — no conversion.

        # ---- Shares: override for IFRS filers ----
        # Foreign private issuers (IFRS taxonomy) often report total shares
        # across all classes — e.g. Ferrari reports 243M covering both the
        # NYSE-listed RACE shares and Exor's special voting shares. That's
        # not the count that matches the traded price or market cap. For
        # IFRS filers we substitute yfinance's sharesOutstanding so per-share
        # math and valuations stay consistent with the rest of the page.
        shares_series = edgar.shares
        if getattr(edgar, "taxonomy", "us-gaap") == "ifrs-full":
            shares_out = getattr(snapshot, "shares_outstanding", None)
            if shares_out and shares_out > 0:
                # Use whichever index is longest among the series we care about
                idx = set()
                for s in (edgar.revenue, edgar.net_income, edgar.equity, edgar.eps):
                    if not s.empty:
                        idx.update(s.index.tolist())
                if idx:
                    shares_series = pd.Series(
                        float(shares_out),
                        index=sorted(idx),
                        dtype=float,
                    )

        # ---- Data-source note ----
        note_parts = [
            f"Data: SEC EDGAR {getattr(edgar, 'taxonomy', 'us-gaap').upper()} "
            f"({edgar.years_available} yrs)"
        ]
        if rate != 1.0 and getattr(edgar, "currency", None):
            note_parts.append(
                f"converted from {edgar.currency} to USD at {rate:.4f}"
            )
        if getattr(edgar, "taxonomy", "us-gaap") == "ifrs-full" and \
           shares_series is not edgar.shares:
            note_parts.append("shares count from Yahoo Finance")
        note_parts.append(
            "Yahoo Finance (price, market cap, P/E, dividend yield, analyst est.)"
        )
        note = " + ".join(note_parts)
        if snapshot.data_source_note:
            note = f"{snapshot.data_source_note} {note}"

        return replace(
            snapshot,
            revenue=rev,
            net_income=ni,
            eps=eps_s,
            equity=eq,
            shares=shares_series,
            operating_cash_flow=ocf_s,
            capex=cpx,
            free_cash_flow=fcf_s,
            long_term_debt=debt,
            tax_rate=edgar.tax_rate,
            ebit=ebit_s,
            data_source="edgar+yfinance",
            data_source_note=note,
        )

    # ---- EDGAR-only (yfinance snapshot genuinely unavailable) ----
    if edgar is not None and snapshot is None:
        return Financials(
            ticker=edgar.ticker,
            revenue=edgar.revenue,
            net_income=edgar.net_income,
            eps=edgar.eps,
            equity=edgar.equity,
            shares=edgar.shares,
            operating_cash_flow=edgar.ocf,
            capex=edgar.capex,
            free_cash_flow=edgar.fcf,
            long_term_debt=edgar.long_term_debt,
            tax_rate=edgar.tax_rate,
            ebit=edgar.ebit,
            current_price=None,
            market_cap=None,
            pe_ratio_ttm=None,
            analyst_5yr_growth=None,
            company_name=edgar.entity_name,
            data_source="edgar",
            data_source_note=(
                f"Data: SEC EDGAR only ({edgar.years_available} yrs). "
                f"Yahoo Finance snapshot unavailable — valuation metrics needing "
                f"current price/EPS may be missing."
            ),
        )

    # ---- yfinance-only (EDGAR unavailable: recent IPO, non-SEC filer, etc.) ----
    reason = str(edgar_err) if edgar_err else "unknown"
    note = (
        f"Data: Yahoo Finance only ({snapshot.years_available} yrs of history). "
        f"SEC EDGAR unavailable — {reason}. "
        f"10-year Big 5 windows may show 'n/a' where history is short."
    )
    if snapshot.data_source_note:
        note = f"{snapshot.data_source_note} {note}"
    return replace(snapshot, data_source="yfinance", data_source_note=note)
