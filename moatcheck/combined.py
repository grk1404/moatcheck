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

from .edgar import EdgarError, fetch_edgar
from .fetcher import Financials, FetchError, fetch as yf_fetch


def fetch(ticker: str) -> Financials:
    """Return a Financials object combining EDGAR history + yfinance snapshot.

    Behavior:
      - EDGAR success + yfinance success → merged (history from EDGAR).
      - EDGAR success + yfinance fail    → EDGAR history but no price/valuation
                                            (snapshot fields are None). UI still
                                            shows Big 5 but can't do value price.
      - EDGAR fail    + yfinance success → yfinance-only, ~4yr history. Warning banner.
      - EDGAR fail    + yfinance fail    → propagate FetchError.
    """
    snapshot: Financials | None = None
    snapshot_err: Exception | None = None
    try:
        snapshot = yf_fetch(ticker)
    except FetchError as e:
        snapshot_err = e

    edgar = None
    edgar_err: Exception | None = None
    try:
        edgar = fetch_edgar(ticker)
    except EdgarError as e:
        edgar_err = e

    if snapshot is None and edgar is None:
        # Both failed — surface the yfinance error since it's more likely to be
        # a "ticker doesn't exist" style message the user can act on.
        raise FetchError(str(snapshot_err) if snapshot_err else str(edgar_err))

    if edgar is not None and snapshot is not None:
        note = (
            f"Data: SEC EDGAR 10-K XBRL ({edgar.years_available} yrs of history) "
            f"+ Yahoo Finance (price, market cap, P/E, dividend yield, analyst est.)"
        )
        return replace(
            snapshot,
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
            data_source="edgar+yfinance",
            data_source_note=note,
        )

    if edgar is not None and snapshot is None:
        # No snapshot — construct a minimal Financials with history only.
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

    # EDGAR failed, yfinance succeeded.
    reason = str(edgar_err) if edgar_err else "unknown"
    note = (
        f"Data: Yahoo Finance only ({snapshot.years_available} yrs of history). "
        f"SEC EDGAR unavailable — {reason}. "
        f"10-year Big 5 windows may show 'n/a' where history is short."
    )
    # Preserve any FX-conversion note the yfinance fetcher already set.
    if snapshot.data_source_note:
        note = f"{snapshot.data_source_note} {note}"
    return replace(snapshot, data_source="yfinance", data_source_note=note)
