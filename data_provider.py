"""
Shared ticker provider — routes all yfinance calls through:
  - curl_cffi Chrome impersonation (defeats Yahoo's TLS fingerprinting)
  - yfinance_cache (persistent SQLite cache, drastically reduces request volume)

Import get_ticker() anywhere you'd otherwise call yf.Ticker().
"""
from curl_cffi import requests as curl_requests
import yfinance_cache as yfc


# One shared session for the whole process. Reusing it preserves cookies and
# the TLS session so Yahoo sees consistent traffic from a single browser.
_shared_session = curl_requests.Session(impersonate="chrome", timeout=15)


def get_ticker(symbol: str):
    """
    Return a Ticker with browser impersonation + persistent cache.

    Usage:
        t = get_ticker("AAPL")
        price = t.history(period="1d")["Close"].iloc[-1]
        info  = t.info
    """
    return yfc.Ticker(symbol, session=_shared_session)