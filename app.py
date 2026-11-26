from __future__ import annotations
import streamlit as st
import pandas as pd
import warnings
import logging

# Import the other modules
from analyzer import render_analyzer
from screener import render_stock_screener
from investments import render_my_investments
from technical_indicators import TechnicalIndicatorAnalyzer
from tqqq import render_tqqq_sqqq_signals

warnings.filterwarnings('ignore')
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

st.set_page_config(page_title="MoatCheck", page_icon=None, layout="wide")

def _inject_tab_styles() -> None:
    st.markdown(
        """
        <style>
        /* ----- Buttons ----- */
        div.stButton > button, div[data-testid="stForm"] button, button {
            background-color: #B0B8C0 !important; color: #1E293B !important;
            border: none !important; font-weight: 700 !important;
            border-radius: 8px !important; padding: 0.5rem 1.5rem !important;
            transition: all 0.2s ease !important;
        }
        div.stButton > button:hover, div[data-testid="stForm"] button:hover, button:hover {
            background-color: #C8D0D8 !important; color: #1E293B !important;
            border: none !important; box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
            transform: translateY(-1px) !important;
        }
        div.stButton > button:active, div[data-testid="stForm"] button:active, button:active {
            background-color: #9AA2AA !important; transform: translateY(0px) !important;
        }

        /* ----- Sidebar navigation links (Yahoo-Mail style) ----- */
        section[data-testid="stSidebar"] {
            background-color: #0b0f14; border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {
            padding: 0.7rem 0.9rem !important; border-radius: 8px !important;
            color: #c5c8cd !important; font-weight: 600 !important;
            font-size: 0.95rem !important; transition: background-color 0.15s ease !important;
            margin-bottom: 2px;
        }
        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {
            background-color: rgba(255,255,255,0.06) !important; color: #e8eaed !important;
        }
        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] {
            background-color: rgba(76, 175, 80, 0.16) !important; color: #b6f0b6 !important;
            border-left: 3px solid #4CAF50 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_technical_analysis():
    """Technical Analysis tab - Trading signals based on technical indicators"""
    st.header("📊 Technical Analysis - Trading Signals")
    st.caption("Technical indicators for identifying potential entry and exit points")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1: tech_ticker = st.text_input("Enter Stock Ticker for Technical Analysis:", "AAPL").upper()
    with col2: tech_period = st.selectbox("Analysis Period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    with col3: tech_analyze = st.button("📊 Analyze Technicals", type="primary", key="tech_analyze")
    
    if tech_analyze or tech_ticker:
        with st.spinner(f"Analyzing {tech_ticker} technical indicators..."):
            analyzer = TechnicalIndicatorAnalyzer(tech_ticker, 'max')            
            if not analyzer.fetch_data():
                st.error(f"Could not fetch data for {tech_ticker}. Please check the ticker symbol.")
                return
            analyzer.calculate_all_indicators()
            analyzer.get_trading_signals()
            summary = analyzer.get_summary_dict()
            verdict_data = analyzer.get_verdict()
            volume_metrics = verdict_data.get('volume_metrics', {})
            
            st.markdown("### Key Indicators")
            key_indicators = verdict_data['key_indicators']
            ind = analyzer.indicators
            price = verdict_data['current_price']
            # Display as a styled dataframe
            indicator_data = {
                'Indicator': ['RSI', 'MACD', 'Stochastic %K', 'Stochastic %D',
                              'SMA 50', 'SMA 200', 'Bollinger Bands', 'Volume'],
                'Value': [
                    f"{key_indicators['rsi']:.1f}",
                    f"{key_indicators['macd']:.4f}",
                    f"{key_indicators['stoch_k']:.1f}",
                    f"{key_indicators['stoch_d']:.1f}",
                    f"${key_indicators['sma_50']:.2f}" if key_indicators['sma_50'] else "N/A",
                    f"${key_indicators['sma_200']:.2f}" if key_indicators['sma_200'] else "N/A",
                    f"${ind['BB_Lower']:.2f} - ${ind['BB_Upper']:.2f}" if ind['BB_Lower'] and ind['BB_Upper'] else "N/A",
                    f"{volume_metrics.get('volume_ratio', 0):.2f}× avg" if volume_metrics else "N/A",
                ],
                'Status': []
            }
            
            # 1. RSI Status
            rsi = key_indicators['rsi']
            if rsi < 30:
                indicator_data['Status'].append('🟢 Oversold (Bullish)')
            elif rsi > 70:
                indicator_data['Status'].append('🔴 Overbought (Bearish)')
            elif 30 <= rsi <= 40:
                indicator_data['Status'].append('🟡 Approaching Oversold')
            elif 60 <= rsi <= 70:
                indicator_data['Status'].append('🟡 Approaching Overbought')
            else:
                indicator_data['Status'].append('🟡 Neutral')
            
            # 2. MACD Status
            macd = key_indicators['macd']
            macd_hist = ind['MACD_Histogram']
            if macd > 0 and macd_hist > 0:
                indicator_data['Status'].append('📈 Bullish (Increasing Momentum)')
            elif macd > 0 and macd_hist < 0:
                indicator_data['Status'].append('🟡 Bullish (Weakening)')
            elif macd < 0 and macd_hist < 0:
                indicator_data['Status'].append('📉 Bearish (Increasing Momentum)')
            elif macd < 0 and macd_hist > 0:
                indicator_data['Status'].append('🟡 Bearish (Weakening)')
            else:
                indicator_data['Status'].append('🟡 Neutral')
            
            # 3. Stochastic %K Status
            stoch_k = key_indicators['stoch_k']
            stoch_d = key_indicators['stoch_d']
            if stoch_k < 20 and stoch_d < 20:
                if stoch_k > stoch_d:
                    indicator_data['Status'].append('🟢 Oversold (Crossover Up)')
                else:
                    indicator_data['Status'].append('🟢 Oversold')
            elif stoch_k > 80 and stoch_d > 80:
                if stoch_k < stoch_d:
                    indicator_data['Status'].append('🔴 Overbought (Crossover Down)')
                else:
                    indicator_data['Status'].append('🔴 Overbought')
            else:
                indicator_data['Status'].append('🟡 Neutral')
            
            # 4. Stochastic %D Status
            if stoch_d < 20:
                indicator_data['Status'].append('🟢 Oversold')
            elif stoch_d > 80:
                indicator_data['Status'].append('🔴 Overbought')
            else:
                indicator_data['Status'].append('🟡 Neutral')
            
            # 5. SMA 50 Status
            sma_50 = key_indicators['sma_50']
            sma_200 = key_indicators['sma_200']
            if sma_50 and sma_200:
                if sma_50 > sma_200:
                    if price > sma_50:
                        indicator_data['Status'].append('📈 Above 200 (Golden Cross)')
                    else:
                        indicator_data['Status'].append('📈 Above 200 (Pullback)')
                else:
                    if price < sma_50:
                        indicator_data['Status'].append('📉 Below 200 (Death Cross)')
                    else:
                        indicator_data['Status'].append('📉 Below 200 (Bounce)')
            else:
                indicator_data['Status'].append('N/A')
            
            # 6. SMA 200 Status
            if sma_200:
                if price > sma_200:
                    percent_above = ((price - sma_200) / sma_200) * 100
                    if percent_above > 20:
                        indicator_data['Status'].append(f'📈 Above (+{percent_above:.0f}%)')
                    else:
                        indicator_data['Status'].append(f'📈 Above ({percent_above:.0f}%)')
                else:
                    percent_below = ((sma_200 - price) / sma_200) * 100
                    if percent_below > 20:
                        indicator_data['Status'].append(f'📉 Below (-{percent_below:.0f}%)')
                    else:
                        indicator_data['Status'].append(f'📉 Below ({percent_below:.0f}%)')
            else:
                indicator_data['Status'].append('N/A')
            
            # 7. Bollinger Bands Status
            if ind['BB_Lower'] and ind['BB_Upper'] and ind['current_price']:
                bb_pos = ((ind['current_price'] - ind['BB_Lower']) / (ind['BB_Upper'] - ind['BB_Lower'])) * 100
                bb_width = ind['BB_Upper'] - ind['BB_Lower']
                
                if bb_pos < 10:
                    indicator_data['Status'].append('🟢 Extreme Lower (Strong Oversold)')
                elif bb_pos < 20:
                    indicator_data['Status'].append('🟢 Near Lower (Oversold)')
                elif bb_pos > 90:
                    indicator_data['Status'].append('🔴 Extreme Upper (Strong Overbought)')
                elif bb_pos > 80:
                    indicator_data['Status'].append('🔴 Near Upper (Overbought)')
                elif 40 <= bb_pos <= 60:
                    indicator_data['Status'].append('🟡 Middle (Neutral)')
                else:
                    indicator_data['Status'].append('🟡 Mid-Range')
                
                # Add bandwidth info if available
                if bb_width and bb_width > 0:
                    avg_price = (ind['BB_Upper'] + ind['BB_Lower']) / 2
                    bandwidth_pct = (bb_width / avg_price) * 100
                    if bandwidth_pct > 20:
                        indicator_data['Status'][-1] += ' 🔸 Wide Volatility'
                    elif bandwidth_pct < 10:
                        indicator_data['Status'][-1] += ' 🔹 Narrow (Squeeze)'
            else:
                indicator_data['Status'].append('N/A')

            # Volume status
            if volume_metrics:
                sev = volume_metrics.get('severity', 'normal')
                direction = volume_metrics.get('price_direction', 'FLAT')
                if sev in ('spike', 'extreme'):
                    if direction == 'UP':
                        indicator_data['Status'].append('🔵 Spike + Price Up (Bullish)')
                    elif direction == 'DOWN':
                        indicator_data['Status'].append('🔴 Spike + Price Down (Bearish)')
                    else:
                        indicator_data['Status'].append('🟠 Spike, Flat Price (Indecision)')
                elif sev == 'elevated':
                    indicator_data['Status'].append('🟢 Elevated Volume')
                elif sev in ('low', 'very_low'):
                    indicator_data['Status'].append('⚪ Low Volume (Weak Conviction)')
                else:
                    indicator_data['Status'].append('🟡 Normal')
            else:
                indicator_data['Status'].append('N/A')
                        
            # Create DataFrame and style it
            df_indicators = pd.DataFrame(indicator_data)
            
            # Style the dataframe to match Stock Analyzer theme
            st.dataframe(
                df_indicators.style.set_properties(**{
                    'background-color': 'rgba(255,255,255,0.05)',
                    'border-color': 'rgba(255,255,255,0.1)',
                    'color': '#e8eaed'
                }).set_table_styles([
                    {'selector': 'thead th', 'props': [('background-color', '#1e1e1e'), ('color', '#b6f0b6')]}
                ]),
                width="stretch",
                hide_index=True
            )
            
            # --- Display Verdict (moved below Key Indicators table) ---
            st.markdown("### Verdict")
            
            # Create verdict display with color matching Stock Analyzer
            verdict_color = verdict_data['color']
            verdict_icon = verdict_data['icon']
            verdict_text = verdict_data['verdict']
            confidence = verdict_data['confidence']
            
            # Color mapping to match Stock Analyzer
            color_map = {
                'bargain': {'bg': '#1e4620', 'text': '#b6f0b6', 'border': '#00E676'},
                'green': {'bg': '#1e4620', 'text': '#b6f0b6', 'border': '#4CAF50'},
                'orange': {'bg': '#4a3a1e', 'text': '#f0d8a6', 'border': '#FFA726'},
                'red': {'bg': '#4b1e1e', 'text': '#f0b6b6', 'border': '#EF5350'},
                'gray': {'bg': '#2a2a2a', 'text': '#9aa0a6', 'border': '#9AA0A6'}
            }
            
            colors = color_map.get(verdict_color, color_map['gray'])
            
            # Get contributing signals
            contributing_signals = verdict_data.get('contributing_signals', [])
            buy_signals = verdict_data.get('buy_signals', [])
            sell_signals = verdict_data.get('sell_signals', [])
            
            # Create verdict card matching Stock Analyzer style
            st.markdown(f"""
            <div style="
                background-color: {colors['bg']};
                border: 2px solid {colors['border']};
                border-radius: 10px;
                padding: 1.5rem;
                margin: 1rem 0;
            ">
                <div style="display: flex; align-items: flex-start; gap: 1rem;">
                    <div style="font-size: 3rem;">{verdict_icon}</div>
                    <div style="flex: 1;">
                        <div style="font-size: 2rem; font-weight: 700; color: {colors['text']};">
                            {verdict_text}
                            <span style="font-size: 1rem; font-weight: 400; color: #9aa0a6;">
                                (Confidence: {confidence:.0f}%)
                            </span>
                        </div>
                        <div style="color: #9aa0a6; font-size: 0.9rem; margin-top: 0.25rem;">
                            {verdict_data['detail']}
                        </div>
            """, unsafe_allow_html=True)
            
            # Add contributing signals
            if contributing_signals:
                st.markdown(f"""
                        <div style="margin-top: 0.75rem;">
                            <div style="color: #9aa0a6; font-size: 0.8rem; margin-bottom: 0.25rem;">
                                <strong>Key contributing indicators:</strong>
                            </div>
                            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                """, unsafe_allow_html=True)
                
                # Display each contributing signal as a badge
                for signal in contributing_signals:
                    # Determine badge color based on signal type
                    if "bullish" in signal.lower() or "above" in signal.lower() or "oversold" in signal.lower():
                        badge_color = "#4CAF50"  # Green for bullish
                    elif "bearish" in signal.lower() or "below" in signal.lower() or "overbought" in signal.lower():
                        badge_color = "#EF5350"  # Red for bearish
                    else:
                        badge_color = "#FFA726"  # Orange for neutral/wait
                    
                    # Shorten long signals for display
                    display_signal = signal
                    if len(signal) > 60:
                        display_signal = signal[:57] + "..."
                    
                    st.markdown(f"""
                        <span style="
                            background-color: rgba(255,255,255,0.08);
                            border-left: 3px solid {badge_color};
                            padding: 0.3rem 0.6rem;
                            border-radius: 4px;
                            font-size: 0.75rem;
                            color: #e8eaed;
                            display: inline-block;
                            margin: 2px 0;
                        ">
                            {display_signal}
                        </span>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                            </div>
                        </div>
                """, unsafe_allow_html=True)
            
            # Close the div
            st.markdown("""
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- Add expandable section for all signals ---
            with st.expander("📋 See all signals that contributed to this verdict"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🟢 Buy Signals**")
                    if buy_signals:
                        for signal in buy_signals:
                            st.markdown(f"<span style='color: #b6f0b6;'>✅ {signal}</span>", unsafe_allow_html=True)
                    else:
                        st.info("No buy signals detected")
                
                with col2:
                    st.markdown("**🔴 Sell Signals**")
                    if sell_signals:
                        for signal in sell_signals:
                            st.markdown(f"<span style='color: #f0b6b6;'>❌ {signal}</span>", unsafe_allow_html=True)
                    else:
                        st.info("No sell signals detected")
                
                # Show neutral signals if any
                neutral_signals = verdict_data.get('neutral_signals', [])
                if neutral_signals:
                    st.markdown("**⏳ Neutral / Wait Signals**")
                    for signal in neutral_signals[:5]:  # Show first 5 to keep it clean
                        st.markdown(f"<span style='color: #f0d8a6;'>⏳ {signal}</span>", unsafe_allow_html=True)
                    if len(neutral_signals) > 5:
                        st.caption(f"... and {len(neutral_signals) - 5} more neutral signals")

            # ============================================================
            # VOLUME ANALYSIS (Levels 1-4)
            # ============================================================
            #volume_metrics = verdict_data.get('volume_metrics', {})

            if volume_metrics:
                st.markdown("### 📊 Volume Analysis")

                vm = volume_metrics
                severity = vm.get('severity', 'normal')
                interpretation = vm.get('interpretation', '')
                statistically_significant = vm.get('statistically_significant', False)

                # Color mapping
                if severity in ('spike', 'extreme'):
                    sev_color, sev_bg = '#FFA726', '#4a3a1e'
                elif severity == 'elevated':
                    sev_color, sev_bg = '#8BC34A', '#2a3d1e'
                elif severity in ('low', 'very_low'):
                    sev_color, sev_bg = '#9AA0A6', '#2a2a2a'
                else:
                    sev_color, sev_bg = '#4CAF50', '#1e4620'

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown(f"""
                    <div style="background-color:{sev_bg}; border:2px solid {sev_color};
                                border-radius:10px; padding:1rem; text-align:center;">
                        <div style="font-size:1.25rem; font-weight:700; color:{sev_color};">
                            {vm['classification']}
                        </div>
                        <div style="color:#9aa0a6; font-size:0.8rem; margin-top:0.25rem;">
                            {vm['volume_ratio']:.2f}× 20-day avg
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"**Interpretation**: {interpretation}")

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Today's Volume", f"{vm['latest_volume']:,.0f}")
                    mc2.metric("20-day Avg", f"{vm['avg_volume_20d']:,.0f}")
                    mc3.metric(
                        "Z-Score",
                        f"{vm['volume_zscore']:+.2f}",
                        delta="Significant" if statistically_significant else "Normal",
                        delta_color="normal" if statistically_significant else "off",
                    )

                    if statistically_significant:
                        st.caption(
                            f"🔬 Volume is {vm['volume_zscore']:.1f} standard deviations above the "
                            f"60-day mean — statistically unusual for this stock."
                        )

                # Show volume weight applied to the signal
                volume_weight = summary['signals'].get('volume_weight', 1.0)
                if abs(volume_weight - 1.0) > 0.1:
                    direction = "increased" if volume_weight > 1.0 else "reduced"
                    st.info(
                        f"⚖️ Signal confidence {direction} to "
                        f"**{volume_weight:.2f}×** due to volume conditions."
                    )
            
            # --- Key Levels (similar to Stock Analyzer) ---
            st.markdown("### Key Levels")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${verdict_data['current_price']:.2f}")
            with col2:
                support = verdict_data['support_level']
                if support:
                    _sup_src = (verdict_data.get('support_info') or {}).get('source', '')
                    st.metric("Support Level", f"${support:.2f}",
                             delta=f"{((verdict_data['current_price'] - support) / verdict_data['current_price'] * 100):.1f}% below price",
                             delta_color="off")
                    if _sup_src:
                        st.caption(f"Nearest support below price: {_sup_src}")
                else:
                    st.metric("Support Level", "N/A")
                    st.caption("No support below the price (trading at new lows)")
            with col3:
                resistance = verdict_data['resistance_level']
                if resistance:
                    _res_src = (verdict_data.get('resistance_info') or {}).get('source', '')
                    st.metric("Resistance Level", f"${resistance:.2f}",
                             delta=f"{((resistance - verdict_data['current_price']) / verdict_data['current_price'] * 100):.1f}% above price",
                             delta_color="off")
                    if _res_src:
                        st.caption(f"Nearest resistance above price: {_res_src}")
                else:
                    st.metric("Resistance Level", "N/A")
            _bbl, _bbu = verdict_data.get('bb_lower'), verdict_data.get('bb_upper')
            if _bbl and _bbu:
                _pos = "below the lower band" if verdict_data['current_price'] < _bbl else (
                    "above the upper band" if verdict_data['current_price'] > _bbu else "inside the bands")
                st.caption(f"Bollinger Bands: ${_bbl:.2f} - ${_bbu:.2f} (price is {_pos}). The bands move with price; "
                           "they are shown for reference and are not used as support.")

                        # --- Suggested entry price (from inverting the technical verdict) ---
            st.markdown("### 🎯 Technical BUY Price")

            entry_info = analyzer.find_entry_price()

            if entry_info is None:
                st.info("Not enough data to compute a BUY price.")
            elif entry_info["entry_price"] is None:
                st.warning(
                    "No technical BUY price: the model verdict never turns to BUY at lower prices and there is "
                    "no support level below the current price (the stock is at its lows). Wait for a base to form."
                )
            else:
                tgt = entry_info["entry_price"]
                cur = entry_info["current_price"]
                drop_pct = (tgt - cur) / cur * 100

                # Color depends on how close the current price is to the target
                _by_support = entry_info.get("method") == "support"
                _bearish = "SELL" in str(entry_info.get("current_verdict", ""))
                if drop_pct >= -0.5 and _by_support and _bearish:
                    # At a support level, but the model is bearish: not a clean buy
                    box_bg, box_color = "#4a3a1e", "#FFA726"
                    status_text = "AT SUPPORT, MODEL BEARISH"
                elif drop_pct >= -0.5:
                    # Already there — bright green
                    box_bg, box_color = "#1e4620", "#00E676"
                    status_text = "AT SUPPORT" if _by_support else "READY NOW"
                elif drop_pct >= -3.0:
                    # Close — light green
                    box_bg, box_color = "#2a3d1e", "#8BC34A"
                    status_text = "NEAR ENTRY"
                elif drop_pct >= -8.0:
                    # Moderate pullback needed — amber
                    box_bg, box_color = "#4a3a1e", "#FFA726"
                    status_text = "WAIT FOR PULLBACK"
                else:
                    # Deep pullback needed — red
                    box_bg, box_color = "#4b1e1e", "#EF5350"
                    status_text = "DEEP PULLBACK NEEDED"

                # Big highlighted card
                st.markdown(f"""
                <div style="background-color:{box_bg}; border:2px solid {box_color};
                            border-radius:10px; padding:1.5rem; text-align:center;
                            margin:0.5rem 0;">
                    <div style="color:#9aa0a6; font-size:0.75rem;
                                text-transform:uppercase; letter-spacing:0.05em;
                                margin-bottom:0.35rem;">
                        Technical BUY Price
                    </div>
                    <div style="font-size:3rem; font-weight:700; color:{box_color};
                                line-height:1.1;">
                        ${tgt:.2f}
                    </div>
                    <div style="color:{box_color}; font-size:1rem;
                                font-weight:600; margin-top:0.35rem;">
                        {status_text}
                    </div>
                    <div style="color:#9aa0a6; font-size:0.85rem; margin-top:0.25rem;">
                        {drop_pct:+.2f}% from current (${cur:.2f})
                    </div>
                </div>
                """, unsafe_allow_html=True)

                _method = entry_info.get("method")
                _sup = entry_info.get("support") or {}
                if _method == "current":
                    st.success("✅ The technical verdict is already BUY at the current price.")
                elif _method == "model":
                    st.info(f"⏳ The technical model flips to BUY at about ${tgt:.2f}.")
                else:
                    st.info(
                        f"📍 Support-based entry: nearest support is **${tgt:.2f}** "
                        f"({_sup.get('source', 'support')}, touched {_sup.get('touches', 1)}x). The model verdict is "
                        f"**{entry_info['current_verdict']}**, so this is a price-action level, not a model BUY signal."
                    )
                if entry_info.get("stop_price") and _method != "current":
                    st.caption(
                        f"Suggested stop: ${entry_info['stop_price']:.2f} (one ATR under the nearest level below the entry), "
                        f"risking about {entry_info['risk_pct']:.1f}% from the entry."
                    )

            # --- 52-week range ---
            st.markdown("### 52-Week Range")
            try:
                _df = analyzer.stock_data
                if _df is not None and len(_df) >= 200:
                    _high_52 = float(_df["High"].tail(252).max())
                    _low_52 = float(_df["Low"].tail(252).min())
                    _cur = float(verdict_data["current_price"])

                    if _high_52 > _low_52 and _cur == _cur:
                        _pos_pct = max(0.0, min(100.0,
                            (_cur - _low_52) / (_high_52 - _low_52) * 100))

                        st.markdown(
                            f"""
                            <div style="margin: 0.5rem 0 1rem;">
                                <div style="display:flex; justify-content:space-between;
                                            color:#9aa0a6; font-size:0.85rem;">
                                    <span>Low ${_low_52:,.2f}</span>
                                    <span>Current <strong style="color:#e8eaed;">${_cur:,.2f}</strong></span>
                                    <span>High ${_high_52:,.2f}</span>
                                </div>
                                <div style="position:relative; height:8px; background:rgba(255,255,255,0.08); border-radius:4px; margin-top:0.5rem; overflow:hidden;">
                                    <div style="position:absolute; left:0; top:0; width:{_pos_pct:.1f}%; height:100%; background:#4DABF7; border-radius:4px 0 0 4px;"></div>
                                    <div style="position:absolute; left:{_pos_pct:.1f}%; top:-4px; width:3px; height:16px; background:#00E676; border-radius:2px; transform:translateX(-50%);"></div>
                                </div>
                                <div style="color:#9aa0a6; font-size:0.75rem; margin-top:0.5rem;">
                                    Current price is at {_pos_pct:.0f}% of the 52-week range.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("52-week range unavailable.")
                else:
                    st.caption("Not enough history for 52-week range.")
            except Exception:
                st.caption("52-week range unavailable.")

            # --- Signal Summary (matching Stock Analyzer table style) ---
            st.markdown("### Signal Summary")
            
            signal_summary = verdict_data['signal_summary']
            
            # Create metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Buy Signals", signal_summary['buy_signals'], 
                         delta="Bullish" if signal_summary['buy_signals'] > 0 else None)
            with col2:
                st.metric("📉 Sell Signals", signal_summary['sell_signals'],
                         delta="Bearish" if signal_summary['sell_signals'] > 0 else None)
            with col3:
                st.metric("⏳ Neutral Signals", signal_summary['neutral_signals'])
            
            # --- Display Chart ---
            st.markdown("### Technical Charts")

            # --- Chart window selector ---
            st.markdown("**Chart window**")
            period_labels = {
                "1M": 30,
                "2M": 60,
                "3M": 90,
                "6M": 180,
                "1Y": 365,
                "2Y": 730,
                "3Y": 1095,
                "5Y": 1825,
                "10Y": 3650,
                "MAX": None,
            }
            selected_period = st.radio(
                "Chart period:",
                options=list(period_labels.keys()),
                index=3,
                horizontal=True,
                key="chart_period",
                label_visibility="collapsed",
            )
            lookback_days = period_labels[selected_period]

            # --- Price overlay selector ---
            st.markdown("**Overlays**")
            cb1, cb2, cb3, cb4 = st.columns(4)
            with cb1:
                show_sma20 = st.checkbox("SMA 20", value=False, key="cb_sma20")
            with cb2:
                show_sma50 = st.checkbox("SMA 50", value=False, key="cb_sma50")
            with cb3:
                show_sma200 = st.checkbox("SMA 200", value=False, key="cb_sma200")
            with cb4:
                show_bb = st.checkbox("Bollinger Bands", value=False, key="cb_bb")

            try:
                fig = analyzer.create_interactive_chart(
                    show_sma20=show_sma20,
                    show_sma50=show_sma50,
                    show_sma200=show_sma200,
                    show_bb=show_bb,
                    lookback_days=lookback_days
                )
                
                if fig:
                    st.plotly_chart(fig, width="stretch")

                    # --- Historical signal performance ---
                    st.markdown("#### 📊 Signal Performance (Historical)")

                    cycles = analyzer.compute_signal_cycles(lookback_days=lookback_days)

                    for label, key in [("MACD (8, 17, 9)", "macd"),
                                    ("Stochastic Oscillator", "stoch")]:
                        st.markdown(f"**{label}**")
                        c_list = cycles.get(key, [])

                        if not c_list:
                            st.caption("No completed BUY→SELL cycles in this window.")
                            continue

                        rows = []
                        for c in c_list:
                            rows.append({
                                "Buy Date": c["buy_date"].strftime("%Y-%m-%d"),
                                "Buy Price": f"${c['buy_price']:.2f}",
                                "Sell Date": c["sell_date"].strftime("%Y-%m-%d"),
                                "Sell Price": f"${c['sell_price']:.2f}",
                                "% Move": f"{c['pct_change']:+.2f}%",
                            })
                        df = pd.DataFrame(rows)

                        # Summary stats
                        pcts = [c["pct_change"] for c in c_list]
                        wins = sum(1 for p in pcts if p > 0)
                        losses = sum(1 for p in pcts if p < 0)
                        avg_win = sum(p for p in pcts if p > 0) / wins if wins else 0
                        avg_loss = sum(p for p in pcts if p < 0) / losses if losses else 0
                        hit_rate = wins / len(pcts) * 100 if pcts else 0

                        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                        sc1.metric("Trades", len(pcts))
                        sc2.metric("Hit Rate", f"{hit_rate:.0f}%")
                        sc3.metric("Avg Win", f"{avg_win:+.2f}%")
                        sc4.metric("Avg Loss", f"{avg_loss:+.2f}%")
                        sc5.metric("Cumulative", f"{sum(pcts):+.2f}%")

                        st.dataframe(df, width="stretch", hide_index=True)
                else:
                    st.warning("Chart method returned None")
            except Exception as e:
                    st.error(f"Chart error: {type(e).__name__}: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # --- Display detailed signals (expandable) ---
            with st.expander("📋 Detailed Signal Breakdown"):
                signals = summary['signals']
                
                if signals.get('buy_signals'):
                    st.markdown("**Buy Signals**")
                    for signal in signals['buy_signals']:
                        st.success(f"✅ {signal}")
                
                if signals.get('sell_signals'):
                    st.markdown("**Sell Signals**")
                    for signal in signals['sell_signals']:
                        st.error(f"❌ {signal}")
                
                if signals.get('neutral_signals'):
                    st.markdown("**Neutral / Wait Signals**")
                    for signal in signals['neutral_signals']:
                        st.warning(f"⏳ {signal}")
                # --- Volume Details (separate section) ---
                volume_details = signals.get('volume_details', [])
                if volume_details:
                    st.markdown("---")
                    st.markdown("**📊 Volume Analysis**")
                    for detail in volume_details:
                        dtype = detail.get('type', 'neutral')
                        text = detail.get('text', '')
                        if dtype == 'positive':
                            st.success(f"✅ {text}")
                        elif dtype == 'caution':
                            st.warning(f"⚠️ {text}")
                        else:
                            st.info(f"ℹ️ {text}")
    
            # ============================================================
            # MEAN-REVERSION STRATEGY FAMILY (7 sub-strategies)
            # Independent view — not part of the main verdict
            # ============================================================
            st.markdown("### 🔀 Mean-Reversion Strategy Family")
            st.caption(
                "Seven trend + mean-reversion rules, weighted independently. "
                "This is a second opinion, not a replacement for the Verdict above."
            )

            mr_signal = analyzer.get_mr_signal()

            if "error" in mr_signal:
                st.warning(mr_signal["error"])
            else:
                score = mr_signal["score"]
                label = mr_signal["label"]
                regime = mr_signal["regime"]
                multiplier = mr_signal["multiplier"]

                # Color by label
                if "STRONG BULLISH" in label:
                    color, bg = "#00E676", "#1e4620"
                elif "BULLISH" in label:
                    color, bg = "#4CAF50", "#1e4620"
                elif "STRONG BEARISH" in label:
                    color, bg = "#EF5350", "#4b1e1e"
                elif "BEARISH" in label:
                    color, bg = "#FF7043", "#3d241e"
                else:
                    color, bg = "#FFA726", "#4a3a1e"

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown(f"""
                    <div style="background-color:{bg}; border:2px solid {color};
                                border-radius:10px; padding:1.25rem; text-align:center;">
                        <div style="font-size:1.5rem; font-weight:700; color:{color};">
                            {label}
                        </div>
                        <div style="color:#9aa0a6; font-size:0.85rem; margin-top:0.4rem;">
                            Score: <strong>{score:.1f}</strong> / 100
                        </div>
                        <div style="color:#9aa0a6; font-size:0.75rem;">
                            Regime: {regime}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Raw Score", f"{mr_signal['raw_score']:.1f}")
                    mc2.metric("BB Multiplier", f"{multiplier:.1f}×")
                    mc3.metric("Adjusted Score", f"{score:.1f}")

                    st.caption(
                        f"BB width: {mr_signal['bb_width']*100:.1f}% · "
                        f"Distance from SMA 50: {mr_signal['dist_sma50']*100:+.1f}% · "
                        f"RSI: {mr_signal['rsi']:.1f}"
                    )

                # Per-strategy breakdown
                with st.expander("📋 Sub-Strategy Votes (7 rules)"):
                    vote_df = pd.DataFrame(
                        mr_signal["details"],
                        columns=["Strategy", "Vote (TQQQ-equivalent %)", "Reason"],
                    )
                    st.dataframe(vote_df, width="stretch", hide_index=True)

                    st.caption(
                        "Each rule votes a score from 0 to 100. The average is the raw score. "
                        "The Bollinger Band width multiplier adjusts for trend strength."
                    )

                # Disagreement note vs. main verdict
                main_verdict = verdict_data["verdict"]
                if ("BULLISH" in label and "SELL" in main_verdict) or \
                   ("BEARISH" in label and "BUY" in main_verdict):
                    st.warning(
                        f"⚠️ **Divergence detected**: the main Verdict says **{main_verdict}** "
                        f"while this strategy family says **{label}**. "
                        f"When they disagree, treat the setup as ambiguous — "
                        f"reduce position size or wait for alignment."
                    )

            # ============================================================
            # BNF REVERSAL SETUP SCANNER
            # Takashi Kotegawa style — mean-reversion on extreme dislocation
            # ============================================================
            st.markdown("---")
            st.markdown("### 🔄 BNF Reversal Setup")
            st.caption(
                "Takashi Kotegawa (BNF) style mean-reversion check. "
                "Looks for stocks dislocated 20%+ below their 25-day MA with RSI/BB confirmation. "
                "This is a bear-market / panic setup — not a trend-following signal."
            )

            bnf = analyzer.check_bnf_reversal_setup()

            # Color and icon by setup state
            if bnf.get("setup"):
                bnf_color, bnf_bg = "#00E676", "#1e4620"
                bnf_label = "⚡ SETUP DETECTED"
            else:
                bnf_color, bnf_bg = "#9AA0A6", "#2a2a2a"
                bnf_label = "No setup"

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                <div style="background-color:{bnf_bg}; border:2px solid {bnf_color};
                            border-radius:10px; padding:1.25rem; text-align:center;">
                    <div style="font-size:1.25rem; font-weight:700; color:{bnf_color};">
                        {bnf_label}
                    </div>
                    <div style="color:#9aa0a6; font-size:0.8rem; margin-top:0.4rem;">
                        {bnf.get('reason', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Show the dislocation as the headline metric
                dislocation = bnf.get("dislocation_pct")
                if dislocation is not None:
                    # Color delta by severity
                    if dislocation <= -35:
                        delta_color = "normal"   # extreme — highlighted
                    elif dislocation <= -20:
                        delta_color = "normal"   # setup zone
                    else:
                        delta_color = "off"      # not dislocated

                    st.metric(
                        "Dislocation from MA25",
                        f"{dislocation:+.2f}%",
                        delta="Extreme" if dislocation <= -35 else
                              "In setup zone" if dislocation <= -20 else
                              "Normal range",
                        delta_color=delta_color,
                    )

                # Confirmation metrics
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("RSI", f"{bnf.get('rsi', 0):.1f}",
                          delta="Oversold" if bnf.get("rsi", 50) < 30 else "Normal",
                          delta_color="normal" if bnf.get("rsi", 50) < 30 else "off")
                mc2.metric("MA 25", f"${bnf.get('ma_25', 0):.2f}")
                mc3.metric("Current Price", f"${bnf.get('current_price', 0):.2f}")

            # Thresholds explanation (collapsed by default)
            with st.expander("📖 How this setup works"):
                st.markdown("""
                **The BNF Reversal Setup** (Takashi Kotegawa style):

                - **Anchor**: 25-day moving average
                - **Dislocation threshold**: price at least 20% below MA25
                - **Extreme panic threshold**: price 35%+ below MA25
                - **Confirmation**: RSI < 30 **OR** price below lower Bollinger Band
                - **Holding period**: 2–6 days, exit as price reverts to the mean

                **When it works**: high-volatility bear markets and panic-driven selloffs.
                **When it fails**: strong trending bull markets — "oversold" stocks keep grinding lower.

                This is a **short-term swing setup**, not a buy-and-hold signal. Cut losses
                if the bounce doesn't materialize within 2–3 days.
                """)
            # --- Download Data ---
            with st.expander("📥 Download Data"):
                csv = analyzer.stock_data.to_csv()
                st.download_button(
                    label="Download Full Technical Data (CSV)",
                    data=csv,
                    file_name=f"{tech_ticker}_technical_data.csv",
                    mime="text/csv"
                )

_inject_tab_styles()

def render_home():
    st.markdown(
        '<h1 style="text-align: center; margin-top: 3rem;">🏰 Welcome to MoatCheck</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #9aa0a6;">'
        'Separating Wonderful Businesses from the Merely Good.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 🔍 Stock Analyzer")
        st.markdown("Enter a ticker symbol to calculate the Big 5 Growth rates, intrinsic value, and Buffett-style health checks.  \nIdentify if it is a :green[Wonderful Company.] " \
        "  \nSeparating Wonderful Businesses from the Merely Good")
    with col2:
        st.markdown("### 🏦 Stock Screener")
        st.markdown("Scan the entire US market for high-quality companies trading at a Margin of Safety." \
        "  \nIdentify potential investment opportunities with a built-in safety margin.  " \
        "\n Scores every US-listed stock on the Big 5 Growth rates and estimates its value. Your latest results are saved and shown whenever you open this page.")
    with col3:
        st.markdown("### 📈 Technical Analysis")
        st.markdown("Identify potential entry and exit points using moving averages, RSI, MACD, Volume Analysis, and Bollinger Bands." \
        "  \nEnhance your trading decisions with Comprehensive Technical Analysis.")
    with col4:
        st.markdown("### 🚀 TQQQ and SQQQ Strategy")
        st.markdown("Leveraged ETF trading strategy using technical indicators to time entries and exits.  \nSignal Source: QQQ Nasdaq 100 Index, Trading Vehicle: TQQQ and SQQQ")
    
    st.info("Use the sidebar to navigate to a specific tool.")

    # --- ADD THIS FOOTER HERE ---
    st.markdown("""
        <div style="margin-top: 5rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.1); 
                    text-align: center; color: #6c6f75; font-size: 0.8rem; line-height: 1.6;">
            <p style="font-weight: 600; color: #9aa0a6;">MoatCheck™</p>
            <p>&copy; 2026 MoatCheck. All rights reserved.</p>
            <p style="font-size: 0.75rem; margin-top: 0.5rem; max-width: 600px; margin-left: auto; margin-right: auto;">
                <em>Disclaimer: MoatCheck is for educational and informational purposes only. It does not constitute financial advice. 
                Always cross-check with SEC filings and consult a qualified financial advisor before making any investment decisions.</em>
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- Router ---
def page_home(): render_home()
def page_analyzer(): render_analyzer()
def page_screener(): render_stock_screener()
def page_technical(): render_technical_analysis()
def page_tqqq(): render_tqqq_sqqq_signals()
def page_investments(): render_my_investments()


# --- Navigation ---
pg = st.navigation([
    st.Page(page_home, title="Home", icon="🏠", default=True),
    st.Page(page_analyzer, title="Stock Analyzer", icon="🔍"),
    st.Page(page_screener, title="Stock Screener", icon="🏦"),
    st.Page(page_technical, title="Technical Analysis", icon="📈"),
    st.Page(page_tqqq, title="TQQQ / SQQQ", icon="🚀"),
    st.Page(page_investments, title="My Investments", icon="💰"),
])

pg.run()