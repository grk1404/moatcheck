# Value Investor App

A Buffett "Value" screening tool with additional intrinsic-value methods. Enter a ticker to compute:

- **The Big 5** — ROIC, Sales, EPS, Equity (BVPS), and Free Cash Flow growth over 10/5/3/1-year windows.
- **Sticker Price & MOS** — Buffett's discounted intrinsic value with 50% margin of safety.
- **Two-Stage DCF** — Free cash flow projection with fade + Gordon terminal value.
- **Peter Lynch Fair Value** — Fair PE = growth% + dividend yield%.
- **Graham Number** — `sqrt(22.5 × EPS × BVPS)`.
- **Graham Formula (1974)** — `EPS × (8.5 + 2g) × 4.4 / bond_yield`.
- **PEG Fair Value** — implied price at PEG = 1.
- **Health checks** — Payback Time (≤8yr) and Debt/FCF (<3).

## Run locally

```bash
cd moatcheck
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 (or whichever port Streamlit prints).

## Deploy to the web (free)

1. Push this folder to a new GitHub repo.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app** → pick the repo → main file is `app.py` → **Deploy**.
4. Wait ~2 minutes. You get a URL like `https://moatcheck-<yourname>.streamlit.app`.

## Install on your phone

- **iOS (Safari):** open the deployed URL → tap the **Share** button → **Add to Home Screen**.
- **Android (Chrome):** open the deployed URL → tap the **⋮** menu → **Add to Home screen** (or **Install app**).

It appears as an icon on your home screen and launches full-screen, no browser chrome.

## Data caveats

- `yfinance` typically returns 4 years of annual statements. The app degrades gracefully when history is shorter than 10 years.
- For a proper Value analysis, cross-check with 10-K filings on SEC EDGAR.
- Not investment advice.
