"""
Technical Indicators Module for Stock Analysis
Provides trading signals based on common technical analysis methods
Supports multiple data providers: yfinance (live) and Nasdaq Data Link (backtesting)
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# DATA PROVIDER CONFIGURATION
# Change this value to switch data sources globally
# Options: "yfinance" (live data) | "nasdaq" (historical backtesting)
# ============================================================
DATA_PROVIDER = "yfinance"

# Nasdaq Data Link API key - get free key at https://data.nasdaq.com/account/profile
# Or set environment variable: NASDAQ_DATA_LINK_API_KEY
NASDAQ_API_KEY = os.environ.get("NASDAQ_DATA_LINK_API_KEY", "QyY6RDh2MPSLRZZs7HcR")


class TechnicalIndicatorAnalyzer:
    """Technical analysis class that works independently or with app.py"""
    
    def __init__(self, ticker, period='1y'):
        """
        Initialize technical analyzer
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        period : str
            Time period for analysis ('1y', '6mo', '3mo', '1mo')
        """
        self.ticker = ticker
        self.period = period
        self.stock_data = None
        self.current_price = None
        self.indicators = {}
        self.signals = {}
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(self.ticker)
            self.stock_data = stock.history(period=self.period)
            if self.stock_data.empty:
                return False
            self.current_price = self.stock_data['Close'].iloc[-1]
            return True
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            return False
    
    def get_volume_signal_weight(self):
        """
        Return a multiplier (0.5x to 2.0x) that scales how much we trust
        the current trading signal, based on volume.
        """
        if not hasattr(self, "volume_metrics") or not self.volume_metrics:
            return 1.0

        severity = self.volume_metrics["severity"]
        direction = self.volume_metrics["price_direction"]

        # Signal direction alignment check happens in the caller.
        # Here we just return the confidence multiplier.
        if severity == "extreme":
            return 2.0
        if severity == "spike":
            return 1.5
        if severity == "elevated":
            return 1.2
        if severity == "normal":
            return 1.0
        if severity == "low":
            return 0.7
        if severity == "very_low":
            return 0.5
        return 1.0
    
    def calculate_all_indicators(self):
        """Calculate all technical indicators  (MACD uses 8, 17, 9)"""
        if self.stock_data is None or self.stock_data.empty:
            return False
        
        # 1. Moving Averages
        for period in [20, 50, 100, 200]:
            self.stock_data[f'SMA_{period}'] = self.stock_data['Close'].rolling(window=period).mean()
        
        # 2. Exponential Moving Averages
        for period in [12, 26]:
            self.stock_data[f'EMA_{period}'] = self.stock_data['Close'].ewm(span=period, adjust=False).mean()
        
        # 3. MACD
        exp1 = self.stock_data['Close'].ewm(span=8, adjust=False).mean()
        exp2 = self.stock_data['Close'].ewm(span=17, adjust=False).mean()
        self.stock_data['MACD'] = exp1 - exp2
        self.stock_data['Signal_Line'] = self.stock_data['MACD'].ewm(span=9, adjust=False).mean()
        self.stock_data['MACD_Histogram'] = self.stock_data['MACD'] - self.stock_data['Signal_Line']
        
        # 4. Stochastic Oscillator
        low_min = self.stock_data['Low'].rolling(window=14).min()
        high_max = self.stock_data['High'].rolling(window=14).max()
        self.stock_data['%K'] = 100 * ((self.stock_data['Close'] - low_min) / (high_max - low_min))
        self.stock_data['%D'] = self.stock_data['%K'].rolling(window=3).mean()
        
        # 5. RSI
        delta = self.stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.stock_data['RSI'] = 100 - (100 / (1 + rs))
        
        # 6. Bollinger Bands
        self.stock_data['BB_Middle'] = self.stock_data['Close'].rolling(window=20).mean()
        bb_std = self.stock_data['Close'].rolling(window=20).std()
        self.stock_data['BB_Upper'] = self.stock_data['BB_Middle'] + (bb_std * 2)
        self.stock_data['BB_Lower'] = self.stock_data['BB_Middle'] - (bb_std * 2)
        
        # 7. Average True Range (ATR) for volatility
        high_low = self.stock_data['High'] - self.stock_data['Low']
        high_close = abs(self.stock_data['High'] - self.stock_data['Close'].shift())
        low_close = abs(self.stock_data['Low'] - self.stock_data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.stock_data['ATR'] = true_range.rolling(14).mean()
        
        # Store latest indicator values
        self._store_latest_indicators()

        # Compute volume metrics (Levels 1-4)
        self.calculate_volume_metrics()
        
        return True

    def calculate_volume_metrics(self):
        """
        Compute volume analysis metrics: ratio, z-score, classification.
        Populates self.volume_metrics.
        """
        if self.stock_data is None or len(self.stock_data) < 20:
            self.volume_metrics = {}
            return

        df = self.stock_data
        volume = df["Volume"]
        price = df["Close"]

        # 20-day average volume
        avg_20 = volume.rolling(20).mean()
        latest_vol = volume.iloc[-1]
        latest_avg = avg_20.iloc[-1]

        # Volume ratio (Level 1)
        volume_ratio = latest_vol / latest_avg if latest_avg > 0 else 0

        # Z-score vs 60-day window (Level 4)
        window = volume.rolling(60)
        mean_60 = window.mean().iloc[-1]
        std_60 = window.std().iloc[-1]
        volume_zscore = ((latest_vol - mean_60) / std_60) if std_60 > 0 else 0

        # Classification (Level 2)
        if volume_ratio >= 4.0:
            classification = "EXTREME SPIKE"
            severity = "extreme"
        elif volume_ratio >= 2.5:
            classification = "SPIKE"
            severity = "spike"
        elif volume_ratio >= 1.5:
            classification = "ELEVATED"
            severity = "elevated"
        elif volume_ratio >= 1.0:
            classification = "NORMAL"
            severity = "normal"
        elif volume_ratio >= 0.5:
            classification = "BELOW AVERAGE"
            severity = "low"
        else:
            classification = "VERY LOW"
            severity = "very_low"

        # Price direction today (for spike interpretation)
        if len(price) >= 2:
            price_change_pct = (price.iloc[-1] - price.iloc[-2]) / price.iloc[-2] * 100
        else:
            price_change_pct = 0.0

        if price_change_pct > 0.5:
            price_direction = "UP"
        elif price_change_pct < -0.5:
            price_direction = "DOWN"
        else:
            price_direction = "FLAT"

        # Spike interpretation (Level 3)
        interpretation = "Normal trading activity"
        if severity in ("spike", "extreme"):
            if price_direction == "UP":
                interpretation = "Bullish breakout confirmation — buyers in control"
            elif price_direction == "DOWN":
                interpretation = "Bearish breakdown OR capitulation — sellers in control"
            else:
                interpretation = "Indecision on high volume — often precedes a move"
        elif severity == "elevated":
            if price_direction == "UP":
                interpretation = "Building bullish interest"
            elif price_direction == "DOWN":
                interpretation = "Building selling pressure"
            else:
                interpretation = "Elevated activity, no clear direction"
        elif severity in ("low", "very_low"):
            interpretation = "Quiet session — signal reliability reduced"

        # Statistical significance flag
        statistically_significant = volume_zscore >= 2.0

        self.volume_metrics = {
            "latest_volume": latest_vol,
            "avg_volume_20d": latest_avg,
            "volume_ratio": volume_ratio,
            "volume_zscore": volume_zscore,
            "classification": classification,
            "severity": severity,
            "price_change_pct": price_change_pct,
            "price_direction": price_direction,
            "interpretation": interpretation,
            "statistically_significant": statistically_significant,
        }

    def _store_latest_indicators(self):
        """Store latest values of all indicators"""
        latest = self.stock_data.iloc[-1]
        self.indicators = {
            'current_price': self.current_price,
            'SMA_20': latest['SMA_20'] if not pd.isna(latest['SMA_20']) else None,
            'SMA_50': latest['SMA_50'] if not pd.isna(latest['SMA_50']) else None,
            'SMA_100': latest['SMA_100'] if not pd.isna(latest['SMA_100']) else None,
            'SMA_200': latest['SMA_200'] if not pd.isna(latest['SMA_200']) else None,
            'MACD': latest['MACD'],
            'Signal_Line': latest['Signal_Line'],
            'MACD_Histogram': latest['MACD_Histogram'],
            'Stoch_%K': latest['%K'],
            'Stoch_%D': latest['%D'],
            'RSI': latest['RSI'],
            'BB_Upper': latest['BB_Upper'],
            'BB_Lower': latest['BB_Lower'],
            'BB_Middle': latest['BB_Middle'],
            'ATR': latest['ATR'] if not pd.isna(latest['ATR']) else None,
            'volume': latest['Volume'],
            'high': latest['High'],
            'low': latest['Low']
        }
    
    def get_trading_signals(self):
        """
        Generate trading signals based on technical indicators
        
        Returns:
        --------
        dict: Contains buy/sell signals with confidence levels
        """
        if not self.indicators:
            return {'error': 'No indicators calculated'}
        
        signals = {
            'buy_signals': [],
            'sell_signals': [],
            'neutral_signals': [],
            'buy_count': 0,
            'sell_count': 0,
            'total_signals': 0,
            'recommendation': 'NEUTRAL',
            'confidence': 0
        }
        
        price = self.indicators['current_price']
        
        # 1. Moving Average Signals
        if self.indicators['SMA_50'] and self.indicators['SMA_200']:
            if price > self.indicators['SMA_50'] and price > self.indicators['SMA_200']:
                signals['buy_signals'].append('Price above 50 & 200 SMA - Bullish trend')
                signals['buy_count'] += 1
            elif price < self.indicators['SMA_50'] and price < self.indicators['SMA_200']:
                signals['sell_signals'].append('Price below 50 & 200 SMA - Bearish trend')
                signals['sell_count'] += 1
            else:
                signals['neutral_signals'].append('Mixed MA signals - Sideways trend')
        
        # 2. MACD Signal
        if self.indicators['MACD'] > self.indicators['Signal_Line']:
            if self.indicators['MACD_Histogram'] > 0:
                signals['buy_signals'].append('MACD bullish crossover with positive momentum')
                signals['buy_count'] += 1
            else:
                signals['neutral_signals'].append('MACD above signal but losing momentum')
        else:
            if self.indicators['MACD_Histogram'] < 0:
                signals['sell_signals'].append('MACD bearish crossover with negative momentum')
                signals['sell_count'] += 1
            else:
                signals['neutral_signals'].append('MACD below signal but gaining momentum')
        
        # 3. Stochastic Oscillator (Buy <20, Sell >80)
        if self.indicators['Stoch_%K'] < 20 and self.indicators['Stoch_%D'] < 20:
            if self.indicators['Stoch_%K'] > self.indicators['Stoch_%D']:
                signals['buy_signals'].append('Stochastic oversold with bullish crossover')
                signals['buy_count'] += 1
            else:
                signals['neutral_signals'].append('Stochastic oversold - waiting for crossover')
        elif self.indicators['Stoch_%K'] > 80 and self.indicators['Stoch_%D'] > 80:
            if self.indicators['Stoch_%K'] < self.indicators['Stoch_%D']:
                signals['sell_signals'].append('Stochastic overbought with bearish crossover')
                signals['sell_count'] += 1
            else:
                signals['neutral_signals'].append('Stochastic overbought - waiting for crossover')
        else:
            signals['neutral_signals'].append(f'Stochastic at {self.indicators["Stoch_%K"]:.1f} - Neutral zone')
        
        # 4. RSI ( Buy <30, Sell >70)
        if self.indicators['RSI'] < 30:
            signals['buy_signals'].append(f'RSI oversold at {self.indicators["RSI"]:.1f}')
            signals['buy_count'] += 1
        elif self.indicators['RSI'] > 70:
            signals['sell_signals'].append(f'RSI overbought at {self.indicators["RSI"]:.1f}')
            signals['sell_count'] += 1
        else:
            signals['neutral_signals'].append(f'RSI at {self.indicators["RSI"]:.1f} - Neutral')
        
        # 5. Bollinger Bands ( Buy near lower, Sell near upper)
        bb_position = ((price - self.indicators['BB_Lower']) / 
                      (self.indicators['BB_Upper'] - self.indicators['BB_Lower'])) * 100
        
        if bb_position < 20:
            signals['buy_signals'].append(f'Price near lower Bollinger Band ({bb_position:.1f}%)')
            signals['buy_count'] += 1
        elif bb_position > 80:
            signals['sell_signals'].append(f'Price near upper Bollinger Band ({bb_position:.1f}%)')
            signals['sell_count'] += 1
        else:
            signals['neutral_signals'].append(f'Price at {bb_position:.1f}% of BB range - Neutral')
        
        # Calculate totals and recommendation
                # Calculate totals and recommendation
        signals['total_signals'] = signals['buy_count'] + signals['sell_count'] + len(signals['neutral_signals'])

        # --- Level 3: Volume weighting ---
        volume_weight = self.get_volume_signal_weight()
        signals['volume_weight'] = volume_weight

        # Adjust buy/sell counts by volume weight
        weighted_buys = signals['buy_count'] * volume_weight
        weighted_sells = signals['sell_count'] * volume_weight
        weighted_total = weighted_buys + weighted_sells + len(signals['neutral_signals'])

        # Use weighted counts for the verdict
        if weighted_total > 0:
            buy_pct = (weighted_buys / weighted_total) * 100
            sell_pct = (weighted_sells / weighted_total) * 100
        else:
            buy_pct = sell_pct = 0

        if buy_pct >= 60:
            signals['recommendation'] = 'STRONG BUY'
            signals['confidence'] = min(100, buy_pct)
        elif buy_pct >= 40:
            signals['recommendation'] = 'BUY'
            signals['confidence'] = min(100, buy_pct)
        elif sell_pct >= 60:
            signals['recommendation'] = 'STRONG SELL'
            signals['confidence'] = min(100, sell_pct)
        elif sell_pct >= 40:
            signals['recommendation'] = 'SELL'
            signals['confidence'] = min(100, sell_pct)
        else:
            signals['recommendation'] = 'WAIT'
            signals['confidence'] = 50

        # Add a volume note to signals details
        if volume_weight > 1.2:
            signals['details'].append(
                f"📊 High volume ({self.volume_metrics['classification']}) strengthens the signal"
            )
        elif volume_weight < 0.8:
            signals['details'].append(
                f"📊 Low volume ({self.volume_metrics['classification']}) weakens the signal"
            )
        
        # Determine recommendation with confidence
        if signals['buy_count'] >= 3:
            signals['recommendation'] = 'STRONG BUY'
            signals['confidence'] = min(100, (signals['buy_count'] / signals['total_signals']) * 100)
        elif signals['buy_count'] >= 2:
            signals['recommendation'] = 'BUY'
            signals['confidence'] = min(100, (signals['buy_count'] / signals['total_signals']) * 100)
        elif signals['sell_count'] >= 3:
            signals['recommendation'] = 'STRONG SELL'
            signals['confidence'] = min(100, (signals['sell_count'] / signals['total_signals']) * 100)
        elif signals['sell_count'] >= 2:
            signals['recommendation'] = 'SELL'
            signals['confidence'] = min(100, (signals['sell_count'] / signals['total_signals']) * 100)
        else:
            signals['recommendation'] = 'WAIT'
            signals['confidence'] = 50
        
        self.signals = signals
        return signals
    
    def generate_html_chart(self):
        """Generate interactive Plotly chart for web display"""
        if self.stock_data is None or self.stock_data.empty:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Indicators', 'MACD', 'Stochastic Oscillator', 'RSI'),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Row 1: Price with MAs and Bollinger Bands
        fig.add_trace(
            go.Scatter(x=self.stock_data.index, y=self.stock_data['Close'],
                      name='Close Price', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Add Moving Averages
        for ma in [20, 50, 200]:
            if f'SMA_{ma}' in self.stock_data.columns:
                fig.add_trace(
                    go.Scatter(x=self.stock_data.index, y=self.stock_data[f'SMA_{ma}'],
                              name=f'SMA {ma}', line=dict(dash='dash', width=1)),
                    row=1, col=1
                )
        
        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(x=self.stock_data.index, y=self.stock_data['BB_Upper'],
                      name='BB Upper', line=dict(color='gray', dash='dot', width=1),
                      showlegend=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.stock_data.index, y=self.stock_data['BB_Lower'],
                      name='BB Lower', line=dict(color='gray', dash='dot', width=1),
                      fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)',
                      showlegend=True),
            row=1, col=1
        )
        
        # Row 2: MACD
        fig.add_trace(
            go.Scatter(x=self.stock_data.index, y=self.stock_data['MACD'],
                      name='MACD', line=dict(color='blue', width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.stock_data.index, y=self.stock_data['Signal_Line'],
                      name='Signal Line', line=dict(color='red', width=2)),
            row=2, col=1
        )
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in self.stock_data['MACD_Histogram']]
        fig.add_trace(
            go.Bar(x=self.stock_data.index, y=self.stock_data['MACD_Histogram'],
                   name='Histogram', marker_color=colors),
            row=2, col=1
        )
        
        # Row 3: Stochastic
        fig.add_trace(
            go.Scatter(x=self.stock_data.index, y=self.stock_data['%K'],
                      name='%K', line=dict(color='blue', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.stock_data.index, y=self.stock_data['%D'],
                      name='%D', line=dict(color='red', width=2)),
            row=3, col=1
        )
        # Add overbought/oversold lines
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)
        
        # Row 4: RSI
        fig.add_trace(
            go.Scatter(x=self.stock_data.index, y=self.stock_data['RSI'],
                      name='RSI', line=dict(color='purple', width=2)),
            row=4, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text=f"{self.ticker.upper()} Technical Analysis Dashboard",
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="Stochastic", row=3, col=1)
        fig.update_yaxes(title_text="RSI", row=4, col=1)
        
        return fig.to_html(full_html=False)

    def create_interactive_chart(self):
        """Create interactive Plotly chart with all technical indicators"""
        if self.stock_data is None or self.stock_data.empty:
            return None
        
        # Create subplots: Price, MACD, Stochastic
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                'Price & Moving Averages',
                'MACD (8, 17, 9)',
                'Stochastic Oscillator',
                'Volume',
            ),
            row_heights=[0.35, 0.25, 0.2, 0.2],
        )
        
        # Row 1: Price with Moving Averages
        fig.add_trace(
            go.Scatter(x=self.stock_data.index, y=self.stock_data['Close'],
                      name='Close Price', line=dict(color='#00BFFF', width=2)),
            row=1, col=1
        )
        
        # Add SMA lines
        for ma, color in [(20, '#FF6B6B'), (50, '#FFA94D'), (200, '#51CF66')]:
            if f'SMA_{ma}' in self.stock_data.columns:
                fig.add_trace(
                    go.Scatter(x=self.stock_data.index, y=self.stock_data[f'SMA_{ma}'],
                              name=f'SMA {ma}', line=dict(color=color, width=1.5, dash='dash')),
                    row=1, col=1
                )
        
        # Add Bollinger Bands
        if 'BB_Upper' in self.stock_data.columns:
            fig.add_trace(
                go.Scatter(x=self.stock_data.index, y=self.stock_data['BB_Upper'],
                          name='BB Upper', line=dict(color='rgba(128,128,128,0.5)', width=1),
                          showlegend=True),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=self.stock_data.index, y=self.stock_data['BB_Lower'],
                          name='BB Lower', line=dict(color='rgba(128,128,128,0.5)', width=1),
                          fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                          showlegend=True),
                row=1, col=1
            )
        
        # Row 2: MACD
        fig.add_trace(
            go.Scatter(x=self.stock_data.index, y=self.stock_data['MACD'],
                      name='MACD', line=dict(color='#4DABF7', width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.stock_data.index, y=self.stock_data['Signal_Line'],
                      name='Signal Line', line=dict(color='#FF6B6B', width=2)),
            row=2, col=1
        )
        
        # MACD Histogram
        colors = ['#51CF66' if val >= 0 else '#FF6B6B' for val in self.stock_data['MACD_Histogram']]
        fig.add_trace(
            go.Bar(x=self.stock_data.index, y=self.stock_data['MACD_Histogram'],
                   name='Histogram', marker_color=colors, opacity=0.5),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Row 3: Stochastic
        fig.add_trace(
            go.Scatter(x=self.stock_data.index, y=self.stock_data['%K'],
                      name='%K', line=dict(color='#4DABF7', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.stock_data.index, y=self.stock_data['%D'],
                      name='%D', line=dict(color='#FF6B6B', width=2)),
            row=3, col=1
        )

        # Row 4: Volume
        if 'Volume' in self.stock_data.columns:
            # Color bars by up/down day
            colors_vol = [
                '#4CAF50' if self.stock_data['Close'].iloc[i] >= self.stock_data['Open'].iloc[i]
                else '#EF5350'
                for i in range(len(self.stock_data))
            ]
            fig.add_trace(
                go.Bar(
                    x=self.stock_data.index,
                    y=self.stock_data['Volume'],
                    name='Volume',
                    marker_color=colors_vol,
                    opacity=0.6,
                ),
                row=4, col=1,
            )

            # 20-day average volume line
            if 'Volume_SMA' not in self.stock_data.columns:
                self.stock_data['Volume_SMA'] = self.stock_data['Volume'].rolling(20).mean()
            fig.add_trace(
                go.Scatter(
                    x=self.stock_data.index,
                    y=self.stock_data['Volume_SMA'],
                    name='Vol Avg (20d)',
                    line=dict(color='#FFA726', width=1.5),
                ),
                row=4, col=1,
            )

            fig.update_yaxes(title_text="Volume", row=4, col=1)
        
        # Overbought/Oversold lines
        fig.add_hline(y=80, line_dash="dash", line_color="#FF6B6B", row=3, col=1, 
                      annotation_text="Overbought", annotation_position="right")
        fig.add_hline(y=20, line_dash="dash", line_color="#51CF66", row=3, col=1,
                      annotation_text="Oversold", annotation_position="right")
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"{self.ticker.upper()} Technical Analysis Dashboard",
            showlegend=True,
            hovermode='x unified',
            template='plotly_dark'
        )
        
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="Stochastic", row=3, col=1, range=[0, 100])        
        
        return fig

    def get_verdict(self):
        """
        Generate a verdict similar to Stock Analyzer theme
        Returns verdict with color coding and detailed explanation
        """
        if not self.signals:
            self.get_trading_signals()
        
        signals = self.signals
        recommendation = signals['recommendation']
        confidence = signals['confidence']
        
        # Get the specific signals that contributed
        buy_signals = signals.get('buy_signals', [])
        sell_signals = signals.get('sell_signals', [])
        neutral_signals = signals.get('neutral_signals', [])
        
        # Determine verdict color and detailed message
        if "STRONG BUY" in recommendation:
            verdict = "STRONG BUY"
            color = "bargain"
            detail = "Multiple indicators showing strong bullish signals with high confidence"
            icon = "🟢"
            # Get top 3 buy signals that contributed
            contributing_signals = buy_signals[:3] if buy_signals else []
        elif "BUY" in recommendation:
            verdict = "BUY"
            color = "green"
            detail = "Technical indicators suggest favorable entry point"
            icon = "🟢"
            contributing_signals = buy_signals[:3] if buy_signals else []
        elif "STRONG SELL" in recommendation:
            verdict = "STRONG SELL"
            color = "red"
            detail = "Multiple indicators showing strong bearish signals - consider exiting or shorting"
            icon = "🔴"
            contributing_signals = sell_signals[:3] if sell_signals else []
        elif "SELL" in recommendation:
            verdict = "SELL"
            color = "red"
            detail = "Technical indicators suggest unfavorable conditions - consider reducing exposure"
            icon = "🔴"
            contributing_signals = sell_signals[:3] if sell_signals else []
        elif "WAIT" in recommendation:
            verdict = "WATCH"
            color = "orange"
            detail = "Mixed signals - wait for clearer direction before taking action"
            icon = "⏳"
            # Show both buy and sell signals for WATCH
            contributing_signals = buy_signals[:2] + sell_signals[:2] if buy_signals or sell_signals else []
        else:
            verdict = "NEUTRAL"
            color = "gray"
            detail = "No clear signals - market conditions are uncertain"
            icon = "⚪"
            contributing_signals = []
        
        # Get current price and key levels
        price = self.current_price
        ind = self.indicators
        
        # Calculate support and resistance levels
        support = ind['BB_Lower'] if ind['BB_Lower'] else None
        resistance = ind['BB_Upper'] if ind['BB_Upper'] else None
        
        # Build detailed analysis
        analysis_details = {
            'verdict': verdict,
            'color': color,
            'icon': icon,
            'confidence': confidence,
            'detail': detail,
            'contributing_signals': contributing_signals,  # NEW: List of contributing signals
            'buy_signals': buy_signals,                   # NEW: All buy signals
            'sell_signals': sell_signals,                 # NEW: All sell signals
            'neutral_signals': neutral_signals,           # NEW: All neutral signals
            'current_price': price,
            'support_level': support,
            'resistance_level': resistance,
            'signal_summary': {
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'neutral_signals': len(neutral_signals)
            },
            'key_indicators': {
                'rsi': ind['RSI'],
                'macd': ind['MACD'],
                'stoch_k': ind['Stoch_%K'],
                'stoch_d': ind['Stoch_%D'],
                'sma_50': ind['SMA_50'],
                'sma_200': ind['SMA_200']
            },
             'volume_metrics': getattr(self, 'volume_metrics', {}),
        }
        
        return analysis_details
    
    def get_summary_dict(self):
        """Get summary in dictionary format for API integration"""
        if not self.indicators:
            self.calculate_all_indicators()
        
        signals = self.get_trading_signals()
        
        return {
            'ticker': self.ticker,
            'current_price': self.current_price,
            'indicators': self.indicators,
            'signals': signals,
            'timestamp': datetime.now().isoformat()
        }
    
    def print_summary(self):
        """Print a formatted summary to console"""
        summary = self.get_summary_dict()
        
        print("\n" + "="*70)
        print(f"📊 TECHNICAL ANALYSIS: {self.ticker.upper()}")
        print("="*70)
        print(f"Current Price: ${summary['current_price']:.2f}")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*70)
        
        print("\n📈 INDICATORS:")
        ind = summary['indicators']
        print(f"  SMA 20: ${ind['SMA_20']:.2f}" if ind['SMA_20'] else "  SMA 20: N/A")
        print(f"  SMA 50: ${ind['SMA_50']:.2f}" if ind['SMA_50'] else "  SMA 50: N/A")
        print(f"  SMA 200: ${ind['SMA_200']:.2f}" if ind['SMA_200'] else "  SMA 200: N/A")
        print(f"  MACD: {ind['MACD']:.4f}")
        print(f"  Signal Line: {ind['Signal_Line']:.4f}")
        print(f"  Stochastic %K: {ind['Stoch_%K']:.1f}")
        print(f"  Stochastic %D: {ind['Stoch_%D']:.1f}")
        print(f"  RSI: {ind['RSI']:.1f}")
        print(f"  Bollinger Position: {((ind['current_price'] - ind['BB_Lower']) / (ind['BB_Upper'] - ind['BB_Lower']) * 100):.1f}%")
        
        print("\n📊 SIGNALS:")
        sig = summary['signals']
        for signal in sig['buy_signals']:
            print(f"  ✅ {signal}")
        for signal in sig['sell_signals']:
            print(f"  ❌ {signal}")
        for signal in sig['neutral_signals']:
            print(f"  ⏳ {signal}")
        
        print("\n" + "="*70)
        print(f"💡 RECOMMENDATION: {sig['recommendation']} (Confidence: {sig['confidence']:.0f}%)")
        print("="*70 + "\n")

# Standalone function for quick analysis
def quick_technical_analysis(ticker, period='1y'):
    """
    Quick function to run technical analysis on any stock
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    period : str
        Time period ('1y', '6mo', '3mo', '1mo')
    
    Returns:
    --------
    dict: Analysis results
    """
    analyzer = TechnicalIndicatorAnalyzer(ticker, period)
    if analyzer.fetch_data():
        analyzer.calculate_all_indicators()
        analyzer.print_summary()
        return analyzer.get_summary_dict()
    else:
        print(f"Error: Could not fetch data for {ticker}")
        return None

# Example usage for testing
if __name__ == "__main__":
    # Test with Apple
    quick_technical_analysis('AAPL', '1y')