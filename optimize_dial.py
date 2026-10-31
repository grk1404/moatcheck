"""
Optimize Dial Parameters across multiple timeframes.
Tests different Dead Zones and Multipliers to find the best CAGR/Risk profile.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_provider import get_ticker
from synthetic import synthetic_levered_series
from execute_backtest import execute



# ============================================================
# 1. PARAMETER GRID (Tweak these to test different ideas)
# ============================================================
DEAD_ZONES = [0.00, 0.02, 0.05, 0.08]
HIGH_VOL_MULTS = [1.0, 1.2, 1.5]
LOW_VOL_MULTS = [0.5, 0.8, 1.0]
TIMEFRAMES = [1, 2, 3, 5, 10, 20]

# ============================================================
# 2. DYNAMIC STRATEGY LOGIC
# ============================================================
def build_targets(qqq_df: pd.DataFrame, dead_zone: float, high_mult: float, low_mult: float) -> pd.DataFrame:
    close = qqq_df["Close"]
    sma_50 = close.rolling(50).mean()
    sma_250 = close.rolling(250).mean()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    rows = []
    for d in close.index:
        if pd.isna(sma_250.loc[d]): continue
            
        price = close.loc[d]
        s50, s250 = sma_50.loc[d], sma_250.loc[d]
        b_upper, b_lower, b_mid = bb_upper.loc[d], bb_lower.loc[d], bb_mid.loc[d]
        
        bb_width = (b_upper - b_lower) / b_mid if b_mid else 0.0
        bb_pos = ((price - b_lower) / (b_upper - b_lower) * 100) if (b_upper - b_lower) else 50.0
        dist_50 = (price - s50) / s50 if s50 else 0.0
        
        if price > s50 > s250: regime = "BULL"
        elif price < s50 < s250: regime = "BEAR"
        else: regime = "NEUTRAL"

        votes = []
        if regime == "BULL": votes.append(1.0)
        elif regime == "BEAR": votes.append(0.0)
        else: votes.append(0.5)
        if dist_50 < -0.08 and regime != "BEAR": votes.append(0.8)
        elif dist_50 > 0.10: votes.append(0.2)
        else: votes.append(0.5)
        if bb_pos < 15: votes.append(0.9)
        elif bb_pos > 85: votes.append(0.1)
        else: votes.append(0.5)
        if regime == "BULL" and bb_width > 0.08: votes.append(0.9)
        else: votes.append(0.5)
        if regime == "BEAR": votes.append(0.0)
        else: votes.append(0.5)
        if dist_50 > 0.12 and regime != "BULL": votes.append(0.1)
        else: votes.append(0.5)
        if regime == "BEAR" and dist_50 > -0.02: votes.append(0.0)
        elif regime == "BEAR" and dist_50 < -0.10: votes.append(0.3)
        else: votes.append(0.5)

        base_dial = sum(votes) / len(votes)
        deviation = base_dial - 0.50
        
        # DYNAMIC MULTIPLIER
        if regime in ("BULL", "BEAR") and bb_width > 0.10:
            multiplier = high_mult
        else:
            multiplier = low_mult
            
        dial = max(0.0, min(1.0, 0.50 + (deviation * multiplier)))
        
        # DYNAMIC DEAD ZONE ALLOCATION
        if dial >= 0.50 + dead_zone:
            tq, sq = dial, 0.0
        elif dial <= 0.50 - dead_zone:
            tq, sq = 0.0, 1.0 - dial
        else:
            tq, sq = 0.0, 0.0
            
        cash = max(0.0, 1.0 - tq - sq)
        rows.append({"date": d, "tqqq": tq, "sqqq": sq, "cash": cash})
        
    return pd.DataFrame(rows).set_index("date")

# ============================================================
# 3. RUN THE OPTIMIZATION
# ============================================================
if __name__ == "__main__":
    print("Fetching 20 years of ^NDX data...")
    # We fetch max to have warmup for the 20-year test
    qqq = get_ticker("^NDX").history(start="2000-01-01")
    qqq.index = pd.to_datetime(qqq.index).tz_localize(None)
    
    print("Generating synthetic 3x TQQQ/SQQQ...")
    qqq_ret = qqq["Close"].pct_change().fillna(0.0)
    tq_nav = synthetic_levered_series(qqq_ret, leverage=3.0, expense_ratio=0.0095, financing_spread=0.02)
    sq_nav = synthetic_levered_series(qqq_ret, leverage=-3.0, expense_ratio=0.0095, financing_spread=0.02)
    
    tqqq = pd.DataFrame({"Open": tq_nav.shift(1).fillna(tq_nav.iloc[0]), "Close": tq_nav})
    sqqq = pd.DataFrame({"Open": sq_nav.shift(1).fillna(sq_nav.iloc[0]), "Close": sq_nav})

    results = []

    for dz in DEAD_ZONES:
        for hi in HIGH_VOL_MULTS:
            for lo in LOW_VOL_MULTS:
                print(f"Testing: DeadZone={dz:.2f}, HighMult={hi:.1f}, LowMult={lo:.1f}")
                
                targets = build_targets(qqq, dead_zone=dz, high_mult=hi, low_mult=lo)
                
                for years in TIMEFRAMES:
                    end_date = qqq.index[-1]
                    # Subtract 2 years for warmup, then add 2 years to the slice start
                    warmup_start = end_date - pd.Timedelta((years + 2)*365, unit='D')
                    actual_start = end_date - pd.Timedelta(years*365, unit='D')

                    # Slice the targets using the warmup_start, but only EXECUTE from actual_start
                    t_slice = targets[(targets.index >= actual_start)]
                    tq_slice = tqqq[tqqq.index >= actual_start]
                    sq_slice = sqqq[sqqq.index >= actual_start]
                    
                    if len(t_slice) < 50: continue
                    
                    res = execute(t_slice, tq_slice, sq_slice, starting_cash=10000)
                    
                    results.append({
                        "DeadZone": dz,
                        "HighMult": hi,
                        "LowMult": lo,
                        "Years": years,
                        "CAGR": res.diagnostics["cagr"] * 100,
                        "MaxDD": res.diagnostics["max_dd"] * 100,
                        "Sharpe": res.diagnostics["sharpe"],
                        "AvgTQQQ": res.diagnostics["avg_tqqq_weight"] * 100,
                    })

    # Output to CSV for analysis
    df = pd.DataFrame(results)
    # Ensure the data directory exists
    from pathlib import Path
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = data_dir / "dial_optimization_results.csv"
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("Results saved to: dial_optimization_results.csv")
    print(f"Results saved to: {output_path}")
    
    # Print the best 5-year and 10-year configs
    for y in [5, 10]:
        print(f"\n--- TOP 3 CONFIGS FOR {y} YEARS (Sorted by Sharpe) ---")
        top = df[df["Years"] == y].sort_values("Sharpe", ascending=False).head(3)
        for _, row in top.iterrows():
            print(f"DZ={row['DeadZone']:.2f} | Hi={row['HighMult']:.1f} | Lo={row['LowMult']:.1f} | "
                  f"CAGR={row['CAGR']:.2f}% | DD={row['MaxDD']:.2f}% | Sharpe={row['Sharpe']:.2f}")