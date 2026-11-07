"""Stock-split handling for per-share series.

EDGAR's companyfacts only restates the last ~3 fiscal years after a split, so older
EPS / share counts stay on the pre-split basis (GOOGL 2017 EPS 18.00 next to 2025 EPS
10.81). This module fetches split history from Yahoo (disk-cached) and puts every value
on today's share basis, using each value's own *filing date*: a value filed after a
split is already adjusted, a value filed before it is not.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

_CACHE_DIR = Path.home() / ".moat_cache" / "splits"
_TTL = 30 * 24 * 3600


def _safe(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", ticker.upper())


def _read(path: Path) -> list | None:
    try:
        if time.time() - path.stat().st_mtime > _TTL:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else None
    except (OSError, ValueError):
        return None


def _write(path: Path, events: list) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(events), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def get_split_events(ticker: str) -> tuple[list[tuple[date, float]], bool]:
    """Return ([(split_date, ratio), ...] sorted by date, ok).

    ratio > 1 is a forward split (2.0 = 2-for-1), ratio < 1 a reverse split. `ok` is
    False when Yahoo could not be reached, so callers can flag unadjusted data.
    """
    path = _CACHE_DIR / f"{_safe(ticker)}.json"
    cached = _read(path)
    if cached is not None:
        return _parse(cached), True
    try:
        import yfinance as yf
        from data_provider import _shared_session

        s = yf.Ticker(ticker, session=_shared_session).splits
        rows = []
        if s is not None and len(s):
            for idx, val in s.items():
                d = pd.Timestamp(idx).date().isoformat()
                v = float(val)
                if v > 0:
                    rows.append([d, v])
        _write(path, rows)
        return _parse(rows), True
    except Exception:
        return [], False


def _parse(rows: list) -> list[tuple[date, float]]:
    out = []
    for r in rows:
        try:
            out.append((datetime.strptime(str(r[0])[:10], "%Y-%m-%d").date(), float(r[1])))
        except (ValueError, TypeError, IndexError):
            continue
    return sorted(out)


def factor_after(filed: str | None, events: list[tuple[date, float]]) -> float | None:
    """Product of split ratios dated after `filed` (None if the filing date is unknown)."""
    if not filed:
        return None
    try:
        fd = datetime.strptime(str(filed)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    f = 1.0
    for d, ratio in events:
        if d > fd:
            f *= ratio
    return f


def adjust_per_share(
    eps: pd.Series,
    shares: pd.Series,
    eps_filed: dict,
    shares_filed: dict,
    events: list[tuple[date, float]],
) -> tuple[pd.Series, pd.Series, int]:
    """Put EPS and share counts on today's share basis.

    Returns (eps_adj, shares_adj, n_unadjustable) where n_unadjustable counts values that
    had no filing date and were left as-is.
    """
    unknown = 0

    def _adj(series: pd.Series, filed_map: dict, per_share: bool) -> pd.Series:
        nonlocal unknown
        if series is None or series.empty or not events:
            return series
        out = series.astype(float).copy()
        for year in out.index:
            f = factor_after(filed_map.get(int(year)) if filed_map else None, events)
            if f is None:
                unknown += 1
                continue
            if f != 1.0:
                out.loc[year] = out.loc[year] / f if per_share else out.loc[year] * f
        return out

    return _adj(eps, eps_filed, True), _adj(shares, shares_filed, False), unknown


def _last_split_date(ticker: str):
    """(date | None, known). known=False when Yahoo's info could not be read, in which case
    callers must not assume the company never split."""
    try:
        from data_provider import get_ticker

        info = get_ticker(ticker).info or {}
        if not info:
            return None, False
        ts = info.get("lastSplitDate")
        if ts in (None, "", 0):
            return None, True
        return datetime.utcfromtimestamp(int(ts)).date(), True
    except Exception:
        return None, False


def adjust_edgar_per_share(
    ticker: str,
    eps: pd.Series,
    shares: pd.Series,
    eps_filed: dict,
    shares_filed: dict,
) -> tuple[pd.Series, pd.Series, str]:
    """Split-adjust EDGAR EPS and share counts. Returns (eps, shares, note).

    Skips the split-history request for companies that have not split since the oldest
    filing in the series (Yahoo's lastSplitDate, which is already fetched for the snapshot),
    so a full-universe scan only pays for the minority that did split.
    """
    if (eps is None or eps.empty) and (shares is None or shares.empty):
        return eps, shares, ""

    filed_dates = [d for d in list((eps_filed or {}).values()) + list((shares_filed or {}).values()) if d]
    try:
        oldest = min(datetime.strptime(d[:10], "%Y-%m-%d").date() for d in filed_dates)
    except ValueError:
        oldest = None

    last, known = _last_split_date(ticker)
    if known and oldest is not None and (last is None or last <= oldest):
        return eps, shares, ""

    events, ok = get_split_events(ticker)
    if not ok:
        return eps, shares, "per-share history NOT split-adjusted (Yahoo split data unavailable)"
    if oldest is not None:
        events = [e for e in events if e[0] > oldest]
    if not events:
        return eps, shares, ""

    eps_adj, shares_adj, unknown = adjust_per_share(eps, shares, eps_filed, shares_filed, events)
    note = f"EPS and share counts adjusted for {len(events)} stock split(s)"
    if unknown:
        note += f" ({unknown} value(s) without a filing date left as reported)"
    return eps_adj, shares_adj, note
