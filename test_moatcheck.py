import sys
import os
import warnings

# --- 1. SILENCE THE WARNINGS & PRINTS ---
# yfinance_cache is notorious for spamming stderr during import. 
# We redirect stderr to devnull before importing anything.
_real_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
warnings.filterwarnings("ignore")

import time
import pandas as pd
from datetime import datetime, timedelta

# --- 2. IMPORT YOUR APP's MODULES ---
try:
    from moatcheck.fetcher import fetch, FetchError
    from moatcheck.big5 import compute_big5
    from moatcheck import value_price, dcf_two_stage
    from technical_indicators import TechnicalIndicatorAnalyzer
    from backtest import load_backtest_data, run_backtest
    import backtest_strategies as bs
except ImportError as e:
    sys.stderr = _real_stderr
    print(f"CRITICAL: Could not import MoatCheck modules. Are you in the right folder? Error: {e}")
    sys.exit(1)

# --- 3. TEST CONFIGURATION ---
TICKERS = [
    "AAPL",   # Mega-cap Tech (Dividends & Buybacks)
    "KO",     # Dividend King (Stable)
    "NVDA",   # High Growth (Recent Split)
    "TSLA",   # High Volatility
    "PLTR",   # Recently turned profitable
    "TQQQ",   # 3x Leveraged ETF
    "SPXL",   # 3x Leveraged ETF
    "SPY",    # Broad Market ETF (No Financials)
    "F",      # Distressed / High Debt
    "O",      # REIT
    "MSFT",   # <--- Just add it here
    "AMZN",   # <--- And here
    "TCS",    # Indian Ticker
]

# --- 4. TEST HELPER FUNCTIONS ---

def _print_header(title):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")

def test_fetch_and_valuation(sym):
    """Tests the Stock Analyzer data pipeline."""
    start_t = time.time()
    try:
        fin = fetch(sym)
        
        # 1. Basic Data Check
        price = fin.current_price
        w52_h = fin.week52_high
        w52_l = fin.week52_low
        eps = float(fin.eps.iloc[-1]) if not fin.eps.empty else None
        
        # 2. Math Check
        big5 = compute_big5(fin)
        eps_growth = big5.eps.values.get(5) or 0.05
        
        # Use fallback EPS if negative
        eps_for_val = eps if eps and eps > 0 else (float(fin.eps[fin.eps > 0].mean()) if not fin.eps[fin.eps > 0].empty else None)
        
        val = value_price(
            current_eps=eps_for_val,
            big5_eps_growth=eps_growth,
            analyst_growth=fin.analyst_5yr_growth,
            historical_pe=fin.pe_ratio_ttm,
            custom_growth=0.15,
        )
        
        # 3. Format Output
        p_str = f"${price:.2f}" if price else "n/a"
        w52_str = f"${w52_l:.0f}-${w52_h:.0f}" if w52_h and w52_l else "n/a"
        e_str = f"${eps:.2f}" if eps else "n/a"
        v_str = f"${val.value_price:.2f}" if val else "n/a"
        
        # Determine Status
        if w52_str == "n/a": status = "⚠️ MISSING 52W"
        elif p_str == "n/a": status = "❌ NO PRICE"
        else: status = "✅ OK"
        
        print(f"{sym:<8} | {p_str:<9} | {w52_str:<12} | {e_str:<8} | {v_str:<10} | {status}")
        
    except FetchError as e:
        print(f"{sym:<8} | SKIPPED  | SKIPPED      | SKIPPED  | SKIPPED    | ⚠️ {str(e)[:30]}")
    except Exception as e:
        print(f"{sym:<8} | ERROR: {str(e)[:60]}")

def test_technical(sym):
    """Tests the Technical Indicator Analyzer."""
    try:
        analyzer = TechnicalIndicatorAnalyzer(sym, '1y')
        if not analyzer.fetch_data():
            return f"❌ Data Fail"
        analyzer.calculate_all_indicators()
        verdict = analyzer.get_verdict()
        return f"✅ {verdict['verdict']} ({verdict['confidence']:.0f}%)"
    except Exception as e:
        return f"❌ {str(e)[:40]}"

def test_backtest(sym):
    """Tests all Backtest Engine strategies for a 3-year window."""
    end_dt = datetime.now().date()
    start_dt = end_dt - timedelta(days=365*3)
    
    try:
        prices = load_backtest_data(sym, start_dt.isoformat(), end_dt.isoformat())
        if prices.empty or len(prices) < 60:
            return "❌ Data Fail"
            
        qqq_close = prices["QQQ_close"]
        tqqq_close = prices["TQQQ_close"]
        
        results = []
        
        # 1. Buy and Hold
        signal_bh = bs.signal_buy_hold(qqq_close, tqqq_close)
        res_bh = run_backtest(prices, signal_bh, starting_cash=10000, warmup_days=0)
        results.append(f"B&H:{res_bh.total_return*100:.0f}%")
        
        # 2. 7-Step TQQQ
        signal_7step = bs.signal_7step(qqq_close, tqqq_close)
        res_7 = run_backtest(prices, signal_7step, starting_cash=10000, warmup_days=0)
        results.append(f"7Step:{res_7.total_return*100:.0f}%")
        
        # 3. One Percent A Week
        signal_1pct = bs.signal_1pct_week(qqq_close, tqqq_close)
        res_1pct = run_backtest(prices, signal_1pct, starting_cash=10000, warmup_days=0)
        results.append(f"1%:{res_1pct.total_return*100:.0f}%")
        
        # 4. Swing (15% since last exit)
        signal_swing = bs.signal_15pct_swing(
            qqq_close, tqqq_close, 
            tqqq_open=prices["TQQQ_open"], 
            threshold=0.15
        )
        res_swing = run_backtest(prices, signal_swing, starting_cash=10000, warmup_days=0)
        results.append(f"Swg:{res_swing.total_return*100:.0f}%")
        
        # 5. Regime-Aware Swing
        signal_regime = bs.signal_swing_regime(
            qqq_close, tqqq_close, 
            tqqq_open=prices["TQQQ_open"], 
            threshold=0.15
        )
        res_regime = run_backtest(prices, signal_regime, starting_cash=10000, warmup_days=0)
        results.append(f"Reg:{res_regime.total_return*100:.0f}%")
        
        return " | ".join(results)
        
    except Exception as e:
        return f"❌ {str(e)[:40]}"


# --- 5. RUN THE TESTS ---

if __name__ == "__main__":
    # Restore stderr just in case we need to see a fatal crash
    sys.stderr = _real_stderr 
    
    print(f"Starting MoatCheck QA Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # --- TEST 1: DATA & VALUATION ---
    _print_header("TEST 1: STOCK ANALYZER (Data + Value Price)")
    print(f"{'Ticker':<8} | {'Price':<9} | {'52w Range':<12} | {'EPS':<8} | {'Value Price':<10} | {'Status'}")
    print("-" * 75)
    
    # Mute stderr again for the yfinance calls
    sys.stderr = open(os.devnull, 'w')
    for sym in TICKERS:
        test_fetch_and_valuation(sym)
    sys.stderr = _real_stderr

    # --- TEST 2: TECHNICAL ANALYSIS ---
    _print_header("TEST 2: TECHNICAL ANALYSIS (Verdict)")
    print(f"{'Ticker':<8} | {'Verdict Result'}")
    print("-" * 45)
    for sym in TICKERS[:5] + ["TQQQ"]: # Test top 5 + TQQQ
        res = test_technical(sym)
        print(f"{sym:<8} | {res}")

    # --- TEST 3: BACKTEST ENGINE ---
    _print_header("TEST 3: BACKTEST ENGINE (Buy & Hold 3Y)")
    print(f"{'Ticker':<8} | {'Backtest Result'}")
    print("-" * 45)
    for sym in ["AAPL", "KO", "TQQQ", "SPXL"]: # Keep it short, backtests are slow
        res = test_backtest(sym)
        print(f"{sym:<8} | {res}")

    print("\nQA Run Complete.")