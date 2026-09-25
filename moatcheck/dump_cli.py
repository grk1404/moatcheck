# moatcheck/dump_cli.py
"""python -m moatcheck.dump_cli AAPL MSFT NVDA ..."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .fetcher import fetch, FetchError
from .edgar import fetch_edgar, EdgarError       # <-- change if named differently
from .dump import (
    financials_to_rows, financials_to_scalars,
    edgar_to_rows, _upsert,
)


def main(tickers: list[str], outdir: Path = Path("data")) -> None:
    outdir = Path(outdir)

    fin_rows: list[dict] = []
    fin_scalars: list[dict] = []
    edgar_rows: list[dict] = []
    edgar_meta: list[dict] = []

    for t in tickers:
        try:
            fin = fetch(t)
            fin_rows.extend(financials_to_rows(fin))
            fin_scalars.append(financials_to_scalars(fin))
            print(f"[yfinance] {t}: ok ({fin.years_available} yrs)")
        except FetchError as e:
            print(f"[yfinance] {t}: {e}")

        try:
            ed = fetch_edgar(t)
            edgar_rows.extend(edgar_to_rows(ed))
            edgar_meta.append({
                "ticker": ed.ticker,
                "cik": ed.cik,
                "entity_name": ed.entity_name,
                "years_available": ed.years_available,
            })
            print(f"[edgar]    {t}: ok (CIK {ed.cik})")
        except EdgarError as e:
            print(f"[edgar]    {t}: {e}")

    # One write per file for the whole batch.
    _upsert(
        pd.DataFrame(fin_rows),
        outdir / "financials.csv",
        keys=["ticker", "source", "metric", "fiscal_year"],
    )
    _upsert(
        pd.DataFrame(fin_scalars),
        outdir / "financials_snapshot.csv",
        keys=["ticker"],
    )
    _upsert(
        pd.DataFrame(edgar_rows),
        outdir / "edgar.csv",
        keys=["ticker", "source", "metric", "fiscal_year"],
    )
    _upsert(
        pd.DataFrame(edgar_meta),
        outdir / "edgar_meta.csv",
        keys=["ticker"],
    )

    print(f"\nWrote to {outdir}/")


if __name__ == "__main__":
    main(sys.argv[1:] or ["AAPL"])