import pandas as pd
import numpy as np
from tqqq_sqqq_strategies import dial_to_allocation

def generate_mallik_signals(qqq_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate Mallik's continuous sizing formula based on his interviews."""
    df = qqq_df.copy()
    
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['sma_250'] = df['Close'].rolling(250).mean()
    
    df['bb_mid'] = df['Close'].rolling(20).mean()
    df['bb_std'] = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    df = df.dropna()
    
    signals = []
    for idx, row in df.iterrows():
        price = row['Close']
        sma50 = row['sma_50']
        sma250 = row['sma_250']
        bb_width = row['bb_width']
        
        dist_250 = (price - sma250) / sma250
        
        # Continuous sizing
        base_target = max(0.0, min(1.0, 0.5 + dist_250 * 5.0))
        
        # Regime Filters
        if price > sma50 and sma50 > sma250:
            target = base_target
        elif price < sma50 and sma50 < sma250:
            target = base_target * 0.5
        else:
            target = 0.5
            
        # Volatility Multiplier
        if bb_width > 0.10:
            target = 0.5 + (target - 0.5) * 1.2
            
        target = max(0.0, min(1.0, target))
        
        # Use the NEW dial_to_allocation function from our live app!
        alloc = dial_to_allocation(target)
            
        signals.append({
            'date': idx,
            'tqqq': alloc["tqqq"],
            'sqqq': alloc["sqqq"],
            'cash': alloc["cash"]
        })
        
    return pd.DataFrame(signals).set_index('date')