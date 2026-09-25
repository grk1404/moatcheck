"""CSV export for moatcheck.

Dumps the Financials / EdgarData dataclasses you already build into local CSVs.
Does not re-fetch: pass in objects you've already constructed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import os
import tempfile
from pathlib import Path


from .fetcher import Financials
from .edgar import EdgarData

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

DEFAULT_OUT = Path("data")

# Metric name -> which Series on Financials / EdgarData it lives on.
# Keeps the two sources aligned in the long-format output.
_FIN_METRICS = [
    ("revenue", "revenue"),
    ("net_income", "net_income"),
    ("eps", "eps"),
    ("shares", "shares"),
    ("equity", "equity"),
    ("long_term_debt", "long_term_debt"),
    ("operating_cash_flow", "operating_cash_flow"),
    ("capex", "capex"),
    ("free_cash_flow", "free_cash_flow"),
    ("ebit", "ebit"),
    ("tax_rate", "tax_rate"),
]

_EDGAR_METRICS = [
    ("revenue", "revenue"),
    ("net_income", "net_income"),
    ("eps", "eps"),
    ("shares", "shares"),
    ("equity", "equity"),
    ("bvps", "bvps"),
    ("long_term_debt", "long_term_debt"),
    ("ocf", "ocf"),
    ("capex", "capex"),
    ("fcf", "fcf"),
    ("ebit", "ebit"),
    ("tax_rate", "tax_rate"),
]

# Scalar fields: single value per ticker, not a time series.
_FIN_SCALARS = [
    "current_price", "market_cap", "pe_ratio_ttm", "analyst_5yr_growth",
    "shares_outstanding", "dividend_yield", "beta", "book_value_per_share",
    "analyst_rec_key", "analyst_rec_mean", "analyst_count",
    "analyst_target_mean", "analyst_target_high", "analyst_target_low",
    "company_name", "data_source", "data_source_note", "exchange",
]


# --------------------------------------------------------------------------- #
# Long-format series dumps
# --------------------------------------------------------------------------- #

def _series_to_rows(ticker: str, metric: str, series: pd.Series, source: str) -> list[dict]:
    if series is None or series.empty:
        return []
    rows = []
    for year, val in series.items():
        if pd.isna(val):
            continue
        rows.append({
            "ticker": ticker,
            "source": source,
            "metric": metric,
            "fiscal_year": int(year),
            "value": float(val),
        })
    return rows


def financials_to_rows(fin: Financials) -> list[dict]:
    rows: list[dict] = []
    for metric, attr in _FIN_METRICS:
        series = getattr(fin, attr, None)
        if isinstance(series, pd.Series):
            rows.extend(_series_to_rows(fin.ticker, metric, series, "yfinance"))
    return rows


def edgar_to_rows(ed: EdgarData) -> list[dict]:
    rows: list[dict] = []
    for metric, attr in _EDGAR_METRICS:
        series = getattr(ed, attr, None)
        if isinstance(series, pd.Series):
            rows.extend(_series_to_rows(ed.ticker, metric, series, "edgar"))
    return rows


def financials_to_scalars(fin: Financials) -> dict:
    row = {"ticker": fin.ticker}
    for name in _FIN_SCALARS:
        row[name] = getattr(fin, name, None)
    return row


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #

def _upsert(df: pd.DataFrame, path: Path, keys: list[str]) -> None:
    """Merge new rows into an existing CSV, de-duplicating on `keys`.

    Writes atomically: temp file in the same dir, then os.replace.
    """
    if df.empty:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        old = pd.read_csv(path)
        # Align columns: keep the union, old order first, then any new cols.
        cols = list(old.columns) + [c for c in df.columns if c not in old.columns]
        old = old.reindex(columns=cols)
        df = df.reindex(columns=cols)
        df = pd.concat([old, df], ignore_index=True)

    df = (
        df.drop_duplicates(subset=keys, keep="last")
        .sort_values(keys)
        .reset_index(drop=True)
    )

    # Keep fiscal_year integer in the file, not float.
    if "fiscal_year" in df.columns:
        df["fiscal_year"] = df["fiscal_year"].astype("Int64")

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def dump_financials(
    fin: Financials,
    outdir: Path = DEFAULT_OUT,
    series_file: str = "financials.csv",
    scalars_file: str = "financials_snapshot.csv",
) -> None:
    """Write one Financials object to long-format series + snapshot CSVs."""
    outdir = Path(outdir)

    series_df = pd.DataFrame(financials_to_rows(fin))
    _upsert(
        series_df,
        outdir / series_file,
        keys=["ticker", "source", "metric", "fiscal_year"],
    )

    scalars_df = pd.DataFrame([financials_to_scalars(fin)])
    _upsert(scalars_df, outdir / scalars_file, keys=["ticker"])


def dump_edgar(
    ed: EdgarData,
    outdir: Path = DEFAULT_OUT,
    series_file: str = "edgar.csv",
    meta_file: str = "edgar_meta.csv",
) -> None:
    """Write one EdgarData object to long-format series + entity metadata CSV."""
    outdir = Path(outdir)

    series_df = pd.DataFrame(edgar_to_rows(ed))
    _upsert(
        series_df,
        outdir / series_file,
        keys=["ticker", "source", "metric", "fiscal_year"],
    )

    meta_df = pd.DataFrame([{
        "ticker": ed.ticker,
        "cik": ed.cik,
        "entity_name": ed.entity_name,
        "years_available": ed.years_available,
    }])
    _upsert(meta_df, outdir / meta_file, keys=["ticker"])


def dump_all(
    fin: Financials,
    ed: EdgarData | None = None,
    outdir: Path = DEFAULT_OUT,
) -> None:
    """Convenience: dump yfinance Financials and (optionally) EDGAR side by side."""
    dump_financials(fin, outdir=outdir)
    if ed is not None:
        dump_edgar(ed, outdir=outdir)