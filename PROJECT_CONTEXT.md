# MoatCheck — Project Context

## What This App Is
Streamlit app for value investing + technical analysis. Tabs:
1. Stock Analyzer (Big 5, DCF, intrinsic value methods)
2. Stock Screener
3. Technical Analysis (indicators, swing signals, TQQQ/SQQQ strategy)

## Data Providers
- Default: yfinance (no API key)
- Polygon.io: tested, works, free tier capped at 2 years
- Nasdaq Data Link: WIKI dataset deprecated, abandoned
- Massive flat files: not implemented

## Files
- app.py — Streamlit UI, tabs, render functions
- technical_indicators.py — TechnicalIndicatorAnalyzer, indicators, verdict, swing signals
- sleeves_strategy.py — the LIVE traded strategy (see below)
- tqqq_sqqq_strategies.py — 7-vote dial + Mallik Sim (reference only)
- trade_manager.py — target share calculator + trade CSV logging
- investment_alerts.py — WhatsApp alert service (CallMeBot), runs as a Windows service on a separate machine
- moatcheck/ — Big 5 / EDGAR / Yahoo fetch code (edgar.py, combined.py, fetcher.py, big5.py, growth.py)
- data/trade_log.csv — trade history

## TQQQ/SQQQ Strategy: the Sleeve is the traded method
Since 2026-11-02 the user trades ONLY the Sleeve strategy, manually at ~2:50 PM CST (3:50 PM ET), on a ~$5k sleeve (starting capital $5,000, started 2026-10-30). The 7-vote dial and Mallik Sim stay on the page as reference only.

Rule (sleeves_strategy.py): ^NDX regime weight +1 above both the 50d and 250d MA, -1 below both, +0.5 between; x0.80 sleeve share; volatility overlay (TARGET_VOL 0.45 / (3 x 20d realized vol), capped at 1); max 10% per day position change; long-or-cash (`short_scale` = 0).

**The Sleeve never buys SQQQ.** Negative weights are multiplied by `short_scale` = 0, so a bear regime means 100% cash. This is deliberate (see findings).

### Backtest results (real TQQQ/SQQQ returns, 5 bps costs, signals from ^NDX)
- 2011 to now: Sleeve 19.8% CAGR / -37.4% max DD / Sharpe 0.79. TQQQ buy-and-hold is about 38-41% CAGR but -82% DD.
- Since 2022-07-23: Sleeve 23.5% / -28.9%.
- 2022 alone: about -20%. The Sleeve goes to cash or small longs in a bear and does not profit from the downtrend.
- Results are in-sample (rules designed after seeing the data). Expect worse live; 30-45% drawdowns are possible.

### Does adding SQQQ help? (tested 2026-11-05)
- Trend-break SQQQ triggers (price < 50d with falling 50d; price < both MAs; price < 20d < 50d) at short_scale 0.5 / 1.0: all WORSE than long-or-cash in 2011-2018, in 2019-2026, and in 2022.
- Plain `short_scale` 0.5 / 1.0 on the existing rule: CAGR down 1-3 points, max DD deeper. No benefit.
- Mean-reversion SQQQ add-on (SQQQ only after a 2-day rally while ^NDX < falling 50d, laddered over 3 days, capped at 20-40% of sleeve): roughly neutral. Max DD improved by 1.7 (2011-18) and 2.9 (2019+) points, 2022 improved from -19.9% to -11.2% (40% cap), but CAGR fell about 1.1 points in 2019+ and the SQQQ trades themselves lost money in 2019+. RSI(>55) filter almost never fired.
- Conclusion: keep the Sleeve long-or-cash. If a crash hedge is wanted, trial the 20%-cap mean-reversion add-on on paper first (not built).

### Mallik source review (video pBS5vrqrUjk, tqqqtrading.wordpress.com, Collective2 "Whitelight", screenshots)
- His system: 7 sub-strategies = 1 momentum long (TQQQ), 3 mean-reversion longs (TQQQ), 1 mean-reversion short (SQQQ), 2 momentum shorts (SQQQ). Inputs: 50/250 MAs, distance from MAs, Bollinger Bands, RSI. No thresholds published. About 2 trades per week, decided ~10 minutes before the close, data from ^NDX.
- The Sleeve is only his momentum-long system. His SQQQ side is short-hold (days, not weeks) and mixed in his own backtest trade list.
- Shorts were added only about a year before the video; the 3-year live curve is flat in the 2022 bear market.
- Claims: 40% CAGR / 33% DD live (his own account), 80% CAGR backtest (synthetic, not reproducible). Collective2 now shows +16.3% CAGR / -37.58% DD, 102 trades, 38.2% win rate, TQQQ-only description. Kinfo and X pages could not be read.

### Live position (STALE: verify before relying on it)
`data/current_position.json` was last updated 2026-10-30 14:54: 44.533 TQQQ @ $61.28, 0 SQQQ, cash $2,271. The trade log has no entries after 2026-10-30. The old "62 SQQQ" position in earlier versions of this file is obsolete.

### Daily workflow
1. At ~2:50 PM CST open the app, click Refresh Signals.
2. Read the Sleeves column (data-as-of date and stale flag) and the Sleeve block in the Position Calculator.
3. Check that the latest ^NDX bar is not intraday before trading.
4. Place trades in Fidelity, then click the Log buttons (expect signal `SLEEVE ...`, note `sleeve as_of=...`; untested until the first real trade).

## Alerts (investment_alerts.py)
- Daily TQQQ message now reads the SLEEVE target (changed 2026-11-05; was the dial). If the sleeve target is unavailable it sends nothing and never falls back to the dial. Not committed or pushed as of this note.
- Dry run (WhatsApp stubbed) produced 52-week high/low, BNF, daily TQQQ and swing alerts correctly. Cycle takes about 3 minutes for 56 positions.
- Likely reason for missing daily alerts: the TQQQ message only fires in a 14:30-15:00 CT window and the service checks about every 63 minutes, so many days no check lands in the window. Service may also be stopped or asleep. Not yet confirmed on the service machine.
- Proposed (not done): send once per trading day after 14:45 CT tracked by date; add a heartbeat log line. TQQQ swing BUY alerts (15%/20% below 20-day high) can conflict with the Sleeve.
- `data/.alert_state.json` is now untracked in git (commit e1e19b5), so pulling deletes the local copy and causes one batch of repeat alerts.
- Security note: the CallMeBot phone number and API key are hard-coded in investment_alerts.py and are in git history.

## Stock Analyzer data fixes (committed f913c6f, pushed 2026-11-05)
- CRWD Sales growth was blank because EDGAR concepts lacked `RevenueFromContractWithCustomerIncludingAssessedTax`. Added it (edgar.py) plus `SalesRevenueServicesNet`.
- Visa and Berkshire (per-class EPS filers) had no usable EDGAR EPS. combined.py now falls back to Yahoo EPS when EDGAR EPS is missing or more than a year stale, and adds a data-source note. Yahoo history is only about 4 years, so 5yr/10yr stay blank.
- Environment: `pulp` 4.x removed `LpVariable.dicts`, which breaks yfinance-cache (fetch raises "No financial statements available"). requirements.txt now pins `pulp<4`; run `python -m pip install "pulp<4"` on every machine.
- Remaining blanks are legitimate: CRWD EPS growth (negative EPS), 10yr windows for short histories. Unverified: GOOGL 10yr EPS growth shows -6.2% (suspected split restatement).

## Constraints / Rules
- Do NOT reference "Rule #1" or "Buffett" in code (licensing)
- Keep changes modular — don't modify existing working code
- Provider pattern: swap DATA_PROVIDER constant to switch data sources
- MACD uses 8, 17, 9 (not standard 12, 26, 9)

## Open Items / Next Steps
- Commit and push the investment_alerts.py sleeve change, then pull and restart the service on the alert machine (`nssm restart <service-name>`).
- Fix the alert timing (once per day after 14:45 CT) and add a heartbeat; confirm service status and logs on the service machine.
- Back Test page: read ^NDX (not QQQ) for the Sleeves signal; add a cost-per-trade input; `test_backtest.py` is stale.
- Fix the sleeves_strategy.py docstring (says "TQQQ / SQQQ / cash" and "-56% max drawdown").
- Replace flat 2% financing in synthetic.py with the actual Fed funds rate.
- Verify the Log buttons write `SLEEVE` rows on the first real trade.
- Optional: build the 20%-cap mean-reversion SQQQ add-on behind a default-off switch for paper trading; test an RSI-scaled TQQQ size against the volatility overlay.
- Eventually: automate execution (recommend IBKR over Fidelity for ToS/API reasons)
