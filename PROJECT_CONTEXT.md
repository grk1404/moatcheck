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
- tqqq_sqqq_strategies.py — 7 sub-strategies voting on TQQQ/SQQQ allocation
- trade_manager.py — target share calculator + trade CSV logging
- trade_log.csv — trade history

## TQQQ/SQQQ Strategy (Malik-inspired)
- 7 independent sub-strategies vote daily on target allocation
- Regime detection via QQQ vs SMA 50 / SMA 200
- Mean reversion via distance from SMA 50
- Bollinger Band position for oversold/overbought
- BB width for position-size multiplier (0.8x narrow, 1.2x wide)
- Signals: TQQQ HEAVY, MILD TQQQ, BALANCED, MILD SQQQ, SQQQ HEAVY
- Trade once daily, 3:50 PM ET, before close
- Win rate < 40%, max drawdown ~30%, ~2 trades/week
- Never liquidate a leg unless signal is BALANCED

## Current Live Position
- Account value: $2,470
- Holdings: 62 SQQQ @ $39.85 (bought 2026-09-10, SQQQ HEAVY, target 62.3%)
- TQQQ: 0 shares
- Next action: rebalance to target at next 3:50 PM ET check

## Daily Workflow
1. Open app at 3:50 PM ET
2. Click Refresh Signals
3. Enter account value + current holdings in Position Calculator
4. Read target shares + buy/sell deltas
5. Place trades in Fidelity
6. Click Log buttons to append to trade_log.csv

## Constraints / Rules
- Do NOT reference "Rule #1" or "Buffett" in code (licensing)
- Keep changes modular — don't modify existing working code
- Provider pattern: swap DATA_PROVIDER constant to switch data sources
- MACD uses 8, 17, 9 (not standard 12, 26, 9)

## Open Items / Next Steps
- Backtest loop for the 7-strategy system using longer history
- Trade accuracy analysis from trade_log.csv
- Eventually: automate execution (recommend IBKR over Fidelity for ToS/API reasons)