import pandas as pd
import numpy as np

def generate_mallik_signals(qqq_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate Mallik's proprietary sizing formula based on his public interviews.
    
    Rules:
    1. Trend: Price vs 50 & 250 SMA.
    2. Extension: Distance from 250 SMA drives the base size.
    3. Momentum: BB Width > 10% amplifies size by 1.2x.
    """
    df = qqq_df.copy()
    
    # 1. Calculate Indicators
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['sma_250'] = df['Close'].rolling(250).mean()
    
    df['bb_mid'] = df['Close'].rolling(20).mean()
    df['bb_std'] = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    # Drop warmup rows
    df = df.dropna()
    
    signals = []
    
    for idx, row in df.iterrows():
        price = row['Close']
        sma50 = row['sma_50']
        sma250 = row['sma_250']
        bb_width = row['bb_width']
        
        # 2. Base sizing based on distance from 250 SMA (Extension)
        dist_250 = (price - sma250) / sma250
        
        # Continuous sizing: 0% at -10% distance, 100% at +10% distance
        base_target = 0.5 + (dist_250 * 5.0) 
        base_target = max(0.0, min(1.0, base_target))
        
        # 3. Trend filter (Momentum)
        if price > sma50 and sma50 > sma250:
            # Bull regime
            target = base_target
        elif price < sma50 and sma50 < sma250:
            # Bear regime
            target = base_target * 0.5 # heavily reduce exposure in bear
        else:
            # Neutral
            target = 0.5
            
        # 4. BB Width Multiplier
        if bb_width > 0.10:
            deviation = target - 0.5
            target = 0.5 + (deviation * 1.2)
            
        target = max(0.0, min(1.0, target))
        
        # One-sided execution (Mallik style)
        if target >= 0.5:
            tqqq = target
            sqqq = 0.0
            cash = 1.0 - target
        else:
            tqqq = 0.0
            sqqq = 1.0 - target
            cash = 1.0 - sqqq
            
        signals.append({
            'date': idx,
            'tqqq': tqqq,
            'sqqq': sqqq,
            'cash': cash
        })
        
    return pd.DataFrame(signals).set_index('date')