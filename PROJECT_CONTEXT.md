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
- investment_alerts.py — alert service (Telegram primary, CallMeBot fallback), market-hours sweeps, runs as a Windows service on a separate machine
- telegram_setup.py — one-time Telegram setup (finds chat id, writes data/alert_secrets.json, sends a test)
- screener.py — Stock Screener UI + parallel analysis; moatcheck/splits.py — split adjustment for per-share history
- technical_indicators.py — indicators, weighted verdict, find_support_resistance(), find_entry_price()
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

## Alerts (investment_alerts.py, runs as a Windows service via NSSM on a separate machine)
**Channels:** Telegram is primary (`TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` from env or `data/alert_secrets.json`); CallMeBot WhatsApp is the fallback. As of 2026-11-06 CallMeBot's WhatsApp quota is exhausted ("You have 0 messages left... Message not sent", returned with HTTP 200), so no alert reached the phone. The CallMeBot phone-call test page worked, but that is a different API. `send_whatsapp` now reads the reply text, so a refusal counts as a failure.

**Schedule (America/Chicago, NYSE calendar via `exchange_calendars`, so holidays and early closes are handled):**
- Threshold sweep starts 1 minute after the open (8:31 AM CT) and repeats every 15 minutes until 5 minutes past the close (about 27 sweeps per trading day). Each sweep checks 52-week highs/lows, MOS buy price, BNF reversal setup and TQQQ swing levels (swing is reference only; the Sleeve is the traded signal).
- Daily Sleeve signal: once per trading day, 20 minutes before the close (2:40 PM CT; 11:40 AM CT on early-close days). The "sent today" date is saved only after delivery succeeds; failures retry every 5 minutes. It refuses to send if the sleeve data is not dated today.
- Cooldowns: 24 h per ticker per alert type (52-week, MOS, swing), 7 days for BNF. Hits from one sweep are batched into one message per category (at most 5 messages per sweep). Expect about 1 daily signal plus 0-3 threshold messages on a normal day; a first sweep after deploy sends about 3-4.

**Fixes made:** the old hourly loop caught the 30-minute daily window on only about half of days (simulated ~51% missed, streaks of 4-5 days); a failed attempt blocked retries; CallMeBot refusals counted as success; 52-week/MOS checks used Yahoo `info` prices that lag by days (SPY showed 766.81 vs a latest close of 735.29) and now use price history; hung cycles exit after 25 minutes so NSSM restarts the service; warnings are suppressed (one cycle wrote 738 stderr lines, now 0); stdout is line-buffered (the old wrapper held log lines in memory); startup banner `Starting Alert Service ---`; heartbeat line every 30 minutes; the daily message uses the Sleeve, never the dial.

**Deploy on the service machine:** `git pull origin master`; `python -m pip install tzdata exchange_calendars "pulp<4"`; `python telegram_setup.py <token>` (press Start on the bot first; it writes `data/alert_secrets.json` with the numeric chat id); `nssm restart <service-name>`. Check the log for the banner, "Telegram configured" and the heartbeat. `data/.alert_state.json` is untracked, so a pull deletes it and causes one batch of repeat alerts.

**SECURITY, open:** commit 6b567f8 added `data/alert_secrets.json` to the PUBLIC repo with a real Telegram bot token and `TELEGRAM_CHAT_ID` set to a username ("grk1404") instead of the numeric chat id. Revoke the token with @BotFather (`/revoke`), then `git rm --cached data/alert_secrets.json` and keep it out of git. The CallMeBot phone number and key are also hard-coded in `investment_alerts.py` and are in history.

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
- SECURITY: revoke the Telegram token exposed in commit 6b567f8, untrack `data/alert_secrets.json`, set the numeric chat id (run `telegram_setup.py` with the new token).
- Deploy the alert service on its machine (see Alerts) and confirm the banner, heartbeat and one Telegram test message. Restore the CallMeBot quota or drop it as fallback.
- Run a full screener scan (download the SEC bulk file first) and check the Retry/Error counts; rebuild the universe cache once.
- Technical Analysis: wire the Analysis Period selector to the analyzer; consider a per-stock volatility-based search floor instead of a fixed -60%.
- Back Test page: read ^NDX (not QQQ) for the Sleeves signal; add a cost-per-trade input; `test_backtest.py` is stale.
- Fix the sleeves_strategy.py docstring (says "TQQQ / SQQQ / cash" and "-56% max drawdown").
- Replace flat 2% financing in synthetic.py with the actual Fed funds rate.
- Verify the Log buttons write `SLEEVE` rows on the first real trade.
- Optional: build the 20%-cap mean-reversion SQQQ add-on behind a default-off switch for paper trading; test an RSI-scaled TQQQ size against the volatility overlay.
- Eventually: automate execution (recommend IBKR over Fidelity for ToS/API reasons)

## Stock Screener rework (committed 3b3b367, pushed 2026-11-06)
**Bugs fixed**
- User Big 5 thresholds were ignored: the screener passes `{"revenue": {"min","max"}}` dicts but `resolve_thresholds` only accepted plain numbers (NVDA scored 5.9 under defaults and under absurd 60-70% thresholds). It now accepts dicts or numbers and never mutates the defaults.
- The result cache was keyed by ticker only for 7 days, so changed thresholds re-served old scores. Rows now cache the threshold-independent growth numbers (`big5_values`) and are re-scored instantly under the current thresholds.
- NASDAQ stocks were never screened: the SEC list says "Nasdaq", the code upper-cased it then compared to mixed case, dropping 4,358 rows (NVDA, MSFT, AAPL, GOOGL, AMZN, META, COST, CRWD absent). Fixed, universe version 4, now about 7,600 tickers.
- Failed fetches (737 of 2,993 cached rows, 117 of them rate-limit errors) were cached 7 days. Transient SEC/Yahoo errors are now marked "Retry" and not cached; hard failures expire after 3 days.
- Metrics were matched to thresholds by list position, so a missing cash-conversion metric made the fair-value discounts use its 80% floor. Now matched by name. Cash conversion and the two discounts count once in the pass rate (they were copied into 4 windows) and are excluded from consistency/trend.
- CAGR used positional lookback, so a missing year stretched a "5-year" window (NVDA FCF 2012 to 2022). `cagr` now looks up the start year by label and returns None if it is missing; `window_average` requires every year.
- EDGAR EPS and share counts were not split-adjusted (EDGAR restates only ~3 years: GOOGL EPS 18.00 in 2017 next to 2.93 in 2020; GOOGL 10yr EPS CAGR -6.2%). New `moatcheck/splits.py` adjusts using Yahoo split history and each value's filing date.
- EDGAR duration facts now must be 340-380 days (COST FY2017 had an 83-day value, NFLX FY2016 a 90-day value).
- Growth feeding the value price is capped at 30% in the screener; "analyst 5-year growth" is really Yahoo's earningsGrowth/revenueGrowth (latest quarter YoY), not a 5-year estimate.

**Speed (measured on samples; a full 7,600-stock run was not completed)**
- Per stock before: about 3,100 ms (Yahoo statements ~2.3 s + EDGAR ~0.8 s). Now about 70-130 ms.
- Light fetch: EDGAR first plus one Yahoo `info` call; full Yahoo statements only when EDGAR has no filings or no EPS series. SEC bulk `companyfacts.zip` (1.4 GB, ~1 min download, stored at `~/.moat_cache/companyfacts.zip`) is read locally (~35 ms/company) with no per-stock SEC request. Shared SEC rate limiter with retries; `EdgarTransientError` separates "SEC busy" from "no filings".
- Process pool (default 80 workers, ~10 processes x 8 threads) for batches of 150+; threads below that. Saves every 100 stocks (atomic write). Parallel universe metadata build that reuses old cache rows. Prices refreshed only for rows with a valuation.
- Measured: 400 stocks cold 45 s, warm 3.4 s; 600 stocks 42 s (10 procs x 8 threads) = about 9 min for 7,600 cold.
- Yahoo returned HTTP 429 after heavy testing, so the batch price refresh could not be verified; throttled rows show "Retry".

**Score/value changes to expect:** AAPL score 7.4 to 6.8, value 112 to 215; CRWD 7.1 to 6.8; ODFL value 449 to 78. Splits and threshold handling both move numbers.

**First run on a machine:** use "Download / refresh SEC bulk data" on the screener page once; the universe rebuilds on first use (about 4,400 new tickers from Yahoo).

**Not changed / open:** Yahoo "analyst growth" is still earningsGrowth/revenueGrowth; multi-class filers rely on the Yahoo EPS fallback; 66 of 300 sampled stocks (foreign filers, trusts) have no usable statements and show Error.

## Technical Analysis page fixes (committed 9478d5b, pushed 2026-11-06)
- Why "Technical BUY Price" was often missing even below support: (1) `_verdict_at_price` changed the last bar but the scores read a stale `current_price`, so the tested price never reached the SMA/MACD/Bollinger scores; (2) even with that fixed, a lower price pushes the SMA and MACD scores to -1, so a stock under its moving averages cannot reach the +0.20 BUY threshold at any price (composite plateaus near +0.04); (3) the page's "Support Level" was the lower Bollinger Band, a moving envelope that sits ABOVE price when the stock is falling, so "below support" could never produce a level.
- Now: `find_support_resistance()` returns real levels (confirmed swing lows/highs clustered within 1.5%, SMA 50/100/200, 52-week low/high; broken levels flip role). `find_entry_price()` returns the current price if the verdict is already BUY (LEAN BUY counts, matching the page), else the highest price where the model turns BUY (1% steps down to -60%), else the nearest support, with a stop one ATR under the level (clamped 1-2 ATR from entry) and the risk %. It works on a copy and restores all state (verified unchanged). No entry now only when the stock is at new lows with no support below.
- Math fixes: RSI uses Wilder smoothing (the old 14-bar mean differed up to 6 points, KO 57.8 vs 64.4); Bollinger uses population std (ddof=0); Stochastic %K guards a flat 14-bar range (division by zero). These changed the verdict for 5 of 50 stocks tested (INTC WAIT to LEAN BUY, FTNT LEAN BUY to WAIT, HLI to BUY, CVNA to LEAN BUY, INGM to WAIT).
- Result on 50 stocks: no entry price for 23 before, 0 now. Support-based entries are price-action levels, not model BUY signals; the page says so and shows the model verdict beside them. The page now analyzes full history ('max'); the Analysis Period selector in the UI is still not wired to it.
- Page behaviour: Key Levels shows real support/resistance with the level's source (swing low, SMA 50, 52-week low, "former resistance"...), and the Bollinger band range as a reference line. The Technical BUY Price card says whether the entry is "current" (verdict already BUY), "model" (verdict flips to BUY at that price) or "support" (price-action level; model verdict shown beside it). A SELL-rated stock at support shows "AT SUPPORT, MODEL BEARISH" in amber. Stop = one ATR under the level, 1-2 ATR from the entry.
- Examples (2026-11-06): MU $690 entry $652.01 (support), stop $593; NVDA $186.64 entry $184.05; AVGO $293.69 entry $293.68, model SELL; AMD entry $435.57 (model); KO/TGT/DIS already BUY. Timing: 0 ms when already BUY, about 30-70 ms by model, about 250 ms on the support fallback.
- Not changed: the Analysis Period selector does nothing (page loads 'max'). `check_bnf_reversal_setup` and `get_mr_signal` read the same RSI and Bollinger columns, so the BNF alert counts can shift slightly after the math corrections.
- Backups of pre-fix files are in the .temp folder under the user profile (technical_indicators.before_ta_fix.py, app.before_ta_fix.py, plus .before_ files from the screener rework).

## Stock Analyzer valuation page changes (2026-11-06)
- Value Price method (moatcheck/valuation.py): growth rate, future P/E = min(trailing P/E, 2 x growth%), future EPS = EPS x (1+g)^10, future price = EPS x P/E, discount 15% for 10 years, MOS = 50%. Arithmetic verified against a hand calculation. The growth cap in value_price() (`#growth = min(growth, 0.15) #RG`) is deliberately left commented out by the user; the Analyzer already limits Big 5 growth to 30% (floor 2%) and analyst growth to 25%.
- Default growth source is now "Model: lower of Big 5 & analyst" (what the valuation function does on its own; MSFT 21.5% to $870.21). "Value conservative: 15%" stays as the second radio option; Big 5, Analyst and Custom are unchanged. The page resets to the model default on each new ticker. Falls back to Value conservative when no growth rate exists.
- Analyst 5Y growth is Yahoo earningsGrowth/revenueGrowth (latest YoY), not a 5-year forecast. AAPL under that option: 25% growth gives Value $817 and MOS $409 (verdict BUY at $345) versus $215/$108 (AVOID) under the model. Open decision: relabel the option "Latest YoY growth (Yahoo)" with a warning when it is far above the 5-year record, cap it at 15%, or leave as is.
- UI: "Growth used: X% (source)" line under the Value Price (single line, no wrap); all DEBUG lines removed; step 1 text of the calculation expander corrected (it wrongly said "capped at 15%").
- Analyst Rating was always n/a because the quick Yahoo snapshot (used when full Yahoo statements fail, e.g. the pulp 4.x issue) never copied the analyst fields. Fixed with a shared `_analyst_fields()` helper; the same fix also fills sector, industry, country, business summary and 52-week range in that snapshot, which is why the Company Summary now shows. The Analyst Rating cell also shows "Analyst growth: X%".
- Loss-making companies (e.g. IONQ, EPS negative every year): all earnings-based options give n/a, Custom cannot help. The warning now says so and points to the P/S valuation. Still open: "Value conservative" substitutes a fake $0.01 EPS (IONQ shows $0.30); an optional "expected EPS in 10 years" input for Custom was proposed and not built.

## Stock Analyzer ticker picker (2026-11-06)
- The Analyzer form has a second box, "Or pick from My Investments": the 44 stocks and funds in `data/fidelity_positions.csv`, largest position first, from `investments.get_holding_tickers()` (cash sweeps like SPAXX/FCASH and CUSIP-style rows are dropped; a stock held in several accounts appears once). Choosing one and clicking Analyze copies it into the Ticker symbol box (`analyzer_ticker_box` via an `on_click` callback) and analyzes it, then resets the list to "(none)" so a stale pick never overrides a typed ticker. The help tooltip was removed because its "?" misaligned the row. The list is read on page load, so refresh after dropping in a new export.
- Payback Time check (not changed): the arithmetic is correct (MSFT: market cap $3,980B, FCF $67B, 1.7% yield, grown 20% a year = 14 years; the method grows free cash flow at the EPS growth rate, capped at 20%). MSFT FCF actually grew only 4% a year (2023 $59B to 2026 $67B), which would give 30 years. Open: use the company's own 3-year FCF growth (2-20%) instead of EPS growth.
