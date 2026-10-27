import sys
import os

# Redirect standard error to a black hole to kill the warnings
# (But keep a backup of the real stderr in case of a fatal crash)
_real_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import warnings
warnings.filterwarnings("ignore")


from moatcheck.fetcher import fetch
from moatcheck.big5 import compute_big5
from moatcheck import value_price, dcf_two_stage

tickers = ["KO", "AAPL", "NVDA", "TSLA", "F", "PLTR", "SPY", "O"]
print(f"{'Ticker':<8} | {'Price':<10} | {'52w Range':<18} | {'EPS (TTM)':<10} | {'Value Price':<11} | {'DCF':<10} | {'Status'}")
print("-" * 95)

for sym in tickers:
    try:
        fin = fetch(sym)
        
        # 1. Basic Data
        price = f"${fin.current_price:.2f}" if fin.current_price else "n/a"
        w52 = f"${fin.week52_low:.0f}-${fin.week52_high:.0f}" if fin.week52_high and fin.week52_low else "n/a"
        
        eps_ttm = float(fin.eps.iloc[-1]) if not fin.eps.empty else None
        eps_str = f"${eps_ttm:.2f}" if eps_ttm else "n/a"
        
        # 2. Big 5 Calculation
        big5 = compute_big5(fin)
        eps_growth = big5.eps.values.get(5) # 5yr growth
        
        # 3. Value Price Calculation (FIXED: added analyst_growth)
        val = value_price(
            current_eps=eps_ttm if eps_ttm and eps_ttm > 0 else (float(fin.eps[fin.eps > 0].mean()) if not fin.eps[fin.eps > 0].empty else None),
            big5_eps_growth=eps_growth,
            analyst_growth=fin.analyst_5yr_growth, # <--- THE FIX
            historical_pe=fin.pe_ratio_ttm,
            custom_growth=0.15,
        )
        val_str = f"${val.value_price:.2f}" if val else "n/a"
        
        # 4. DCF Calculation
        fcf = float(fin.free_cash_flow.iloc[-1]) if not fin.free_cash_flow.empty else None
        dcf = dcf_two_stage(
            fcf_ttm=fcf,
            shares_out=fin.shares_outstanding,
            current_price=fin.current_price,
            growth_rate=eps_growth or 0.05,
            discount_rate=0.10,
            terminal_growth=0.025,
        )
        dcf_str = f"${dcf.fair_value:.2f}" if dcf and dcf.fair_value else "n/a"
        
        status = "OK" if w52 != "n/a" and price != "n/a" else "MISSING DATA"
        
        print(f"{sym:<8} | {price:<10} | {w52:<18} | {eps_str:<10} | {val_str:<11} | {dcf_str:<10} | {status}")
        
    except Exception as e:
        print(f"{sym:<8} | SKIPPED (ETF/Financials missing): {str(e)[:50]}")