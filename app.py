import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

from database import DatabaseManager
from technical_analysis import TechnicalAnalysis
from portfolio_manager import PortfolioManager
from stock_groups import StockGroupManager
from utils import format_currency, get_signal_color, create_candlestick_chart

# Initialize session state
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

if 'tech_analysis' not in st.session_state:
    st.session_state.tech_analysis = TechnicalAnalysis()

if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = PortfolioManager(st.session_state.db_manager)

if 'stock_group_manager' not in st.session_state:
    st.session_state.stock_group_manager = StockGroupManager(st.session_state.db_manager)

def main():
    st.set_page_config(
        page_title="Stock Analysis Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📈 Stock Analysis Dashboard")

    # Sidebar Navigation
    with st.sidebar:
        st.header("Navigation")

        # Module selection
        module = st.radio(
            "Select Module",
            ["Stock Group Management", "Portfolio Management"],
            key="module_selector"
        )

        # Refresh data button
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            refresh_all_data()

        #st.write(f"Last Updated: {datetime.now().strftime('%H:%M')}")

        st.divider()

        # Module-specific controls
        if module == "Stock Group Management":
            st.subheader("Stock Group Controls")

            # Group selection
            selected_group = st.selectbox(
                "Select Stock Group",
                ["V40", "V40 Next", "V200"],
                key="group_selector"
            )

            # Add stock to group
            st.write("**Add Stock to Group**")
            new_stock = st.text_input("Stock Symbol (without .NS)", key="new_stock_group")
            if st.button("Add Stock", key="add_stock_group", use_container_width=True):
                if new_stock:
                    symbol = new_stock.upper() + ".NS"
                    success = st.session_state.stock_group_manager.add_stock_to_group(selected_group, symbol)
                    if success:
                        st.success(f"Added {new_stock.upper()} to {selected_group}")
                        st.rerun()
                    else:
                        st.error(f"Stock {new_stock.upper()} already exists in {selected_group}")
                else:
                    st.error("Please enter a stock symbol")

            # Remove stock from group
            group_stocks = st.session_state.stock_group_manager.get_group_stocks(selected_group)
            if group_stocks:
                st.write("**Remove Stock from Group**")
                stock_to_delete = st.selectbox(
                    "Select stock to remove",
                    [stock['symbol'] for stock in group_stocks],
                    format_func=lambda x: x.replace('.NS', ''),
                    key="delete_stock_selector"
                )
                if st.button("Remove Stock", key="delete_stock_group", use_container_width=True):
                    if stock_to_delete:
                        st.session_state.stock_group_manager.remove_stock_from_group(selected_group, stock_to_delete)
                        st.success(f"Removed {stock_to_delete.replace('.NS', '')} from {selected_group}")
                        st.rerun()

        else:  # Portfolio Management
            st.subheader("Portfolio Controls")

            # Add stock to portfolio
            st.write("**Add Stock to Portfolio**")
            portfolio_stock = st.text_input("Stock Symbol (without .NS)", key="portfolio_stock")
            quantity = st.number_input("Quantity", min_value=1, value=1, key="quantity")
            buy_price = st.number_input("Buy Price (₹)", min_value=0.01, value=100.0, key="buy_price")

            if st.button("Add to Portfolio", key="add_portfolio_stock", use_container_width=True):
                if portfolio_stock and quantity > 0 and buy_price > 0:
                    symbol = portfolio_stock.upper() + ".NS"
                    st.session_state.portfolio_manager.add_stock(symbol, quantity, buy_price)
                    st.success(f"Added {quantity} shares of {portfolio_stock.upper()} at ₹{buy_price}")
                    st.rerun()
                else:
                    st.error("Please fill all fields correctly")

            # Remove stock from portfolio
            portfolio_stocks = st.session_state.portfolio_manager.get_portfolio_stocks()
            if portfolio_stocks:
                st.write("**Remove Stock from Portfolio**")
                stock_to_delete = st.selectbox(
                    "Select stock to remove",
                    options=[stock['symbol'] for stock in portfolio_stocks],
                    format_func=lambda x: x.replace('.NS', ''),
                    key="stock_to_delete"
                )
                if st.button("Remove Stock", key="remove_portfolio_stock", use_container_width=True):
                    if stock_to_delete:
                        st.session_state.portfolio_manager.remove_stock(stock_to_delete)
                        st.success(f"Removed {stock_to_delete.replace('.NS', '')} from portfolio")
                        st.rerun()

    # Main content area
    if module == "Stock Group Management":
        stock_group_interface()
    else:
        portfolio_interface()

def refresh_all_data():
    """Refresh all stock data from yFinance"""
    with st.spinner("Refreshing data from yFinance..."):
        try:
            # Get all unique stocks from groups and portfolio
            all_stocks = set()

            # Get stocks from groups
            for group in ['V40', 'V40 Next', 'V200']:
                stocks = st.session_state.stock_group_manager.get_group_stocks(group)
                all_stocks.update([stock['symbol'] for stock in stocks])

            # Get stocks from portfolio
            portfolio_stocks = st.session_state.portfolio_manager.get_portfolio_stocks()
            all_stocks.update([stock['symbol'] for stock in portfolio_stocks])

            # Update data for each stock
            for symbol in all_stocks:
                update_stock_data(symbol)

            st.success(f"✅ Successfully refreshed data for {len(all_stocks)} stocks")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error refreshing data: {str(e)}")

def update_stock_data(symbol):
    """Update stock data for a given symbol"""
    try:
        ticker = yf.Ticker(symbol)

        # Get historical data (2 years)
        hist_data = ticker.history(period="2y")
        if hist_data.empty:
            return

        # Get current info
        info = ticker.info

        # Store historical data
        st.session_state.db_manager.store_historical_data(symbol, hist_data)

        # Store current stock info
        stock_info = {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'current_price': info.get('currentPrice', hist_data['Close'].iloc[-1]),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'debt_to_equity': info.get('debtToEquity', 0),
            'roe': info.get('returnOnEquity', 0),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
            'lifetime_high': hist_data['High'].max(),
            'updated_at': datetime.now()
        }

        st.session_state.db_manager.store_stock_info(stock_info)

        # Calculate and store technical signals
        calculate_and_store_signals(symbol, hist_data)

    except Exception as e:
        st.error(f"Error updating {symbol}: {str(e)}")

def calculate_and_store_signals(symbol, hist_data):
    """Calculate technical analysis signals and store them"""
    try:
        # Calculate all technical signals
        sma_signal = st.session_state.tech_analysis.calculate_sma_signal(hist_data)
        green_candle_signal = st.session_state.tech_analysis.calculate_green_candle_signal(hist_data)
        range_bound_signal = st.session_state.tech_analysis.calculate_range_bound_signal(hist_data)
        rhs_signal = st.session_state.tech_analysis.calculate_rhs_signal(hist_data)
        cup_handle_signal = st.session_state.tech_analysis.calculate_cup_handle_signal(hist_data)

        # Calculate fundamental score (simplified)
        fundamental_score = calculate_fundamental_score(symbol)

        # Store signals
        signals = {
            'symbol': symbol,
            'sma_signal': sma_signal['signal'],
            'green_candle_signal': green_candle_signal['signal'],
            'range_bound_signal': range_bound_signal['signal'],
            'rhs_signal': rhs_signal['signal'],
            'cup_handle_signal': cup_handle_signal['signal'],
            'fundamental_score': fundamental_score,
            'average_rating': calculate_average_rating([sma_signal['signal'], green_candle_signal['signal'],
                                                      range_bound_signal['signal'], rhs_signal['signal'],
                                                      cup_handle_signal['signal']]),
            'updated_at': datetime.now()
        }

        st.session_state.db_manager.store_signals(signals)

    except Exception as e:
        st.error(f"Error calculating signals for {symbol}: {str(e)}")

def calculate_fundamental_score(symbol):
    """Calculate a simplified fundamental score"""
    try:
        stock_info = st.session_state.db_manager.get_stock_info(symbol)
        if not stock_info:
            return 50

        score = 50  # Base score

        # PE ratio scoring
        if stock_info['pe_ratio'] > 0:
            if stock_info['pe_ratio'] < 15:
                score += 20
            elif stock_info['pe_ratio'] < 25:
                score += 10
            elif stock_info['pe_ratio'] > 40:
                score -= 10

        # ROE scoring
        if stock_info['roe'] > 0:
            if stock_info['roe'] > 0.15:
                score += 15
            elif stock_info['roe'] > 0.10:
                score += 10
            elif stock_info['roe'] < 0.05:
                score -= 10

        # Debt to equity scoring
        if stock_info['debt_to_equity'] > 0:
            if stock_info['debt_to_equity'] < 0.5:
                score += 10
            elif stock_info['debt_to_equity'] > 2:
                score -= 15

        return max(0, min(100, score))
    except:
        return 50

def calculate_average_rating(signals):
    """Calculate average rating from signals"""
    signal_values = {'BUY': 1, 'SELL': -1, 'WATCH': 0, 'NEUTRAL': 0}
    total = sum(signal_values.get(signal, 0) for signal in signals)
    avg = total / len(signals)

    if avg > 0.3:
        return 'BUY'
    elif avg < -0.3:
        return 'SELL'
    else:
        return 'NEUTRAL'

def stock_group_interface():
    """Interface for Stock Group Management"""
    # Get selected group from sidebar
    selected_group = st.session_state.get('group_selector', 'V40')

    st.header(f"📊 Stock Group Management - :green[{selected_group}]")

    # Display group stocks
    group_stocks = st.session_state.stock_group_manager.get_group_stocks(selected_group)

    if group_stocks:
        # Tabs for Summary and Detail Analysis
        tab1, tab2 = st.tabs(["📋 Summary", "🔍 Detail Analysis"])

        with tab1:
            display_summary_tab(group_stocks, selected_group)

        with tab2:
            display_detail_analysis_tab(group_stocks)
    else:
        st.info(f"No stocks found in {selected_group} group. Add some stocks using the sidebar controls.")

# Style the portfolio dataframe
def style_pnl(val):
    if val > 0:
        return ' color: #155724;font-weight: bold;'
    elif val < 0:
        return 'color: #721c24; font-weight: bold;'
    return ''

def display_summary_tab(group_stocks, selected_group):
    """Display summary tab with stock overview"""
    st.subheader("Stock Summary")

    # Create summary dataframe
    summary_data = []
    for stock in group_stocks:
        stock_info = st.session_state.db_manager.get_stock_info(stock['symbol'])
        signals = st.session_state.db_manager.get_signals(stock['symbol'])

        if stock_info and signals:
            # Calculate daily change
            hist_data = st.session_state.db_manager.get_historical_data(stock['symbol'])
            daily_change = 0
            if len(hist_data) >= 2:
                daily_change = ((hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-2]) /
                               hist_data['Close'].iloc[-2]) * 100

            # Calculate overall signal based on majority vote
            all_signals = [
                signals['sma_signal'],
                signals['green_candle_signal'],
                signals['range_bound_signal'],
                signals['rhs_signal'],
                signals['cup_handle_signal']
            ]

            signal_counts = {}
            for signal in all_signals:
                signal_counts[signal] = signal_counts.get(signal, 0) + 1

            # Determine overall signal
            overall_signal = max(signal_counts.items(), key=lambda x: x[1])[0]
            if overall_signal=='NEUTRAL':
                overall_signal='HOLD'

            summary_data.append({
                'Stock': stock['symbol'].replace('.NS', ''),
                'Company': stock_info['company_name'],
                'Price': stock_info['current_price'],
                'Daily Change %': daily_change,
                'Fundamental Score': signals['fundamental_score'],
                'Overall Signal': overall_signal,
                'SMA': signals['sma_signal'],
                'Green Candle': signals['green_candle_signal'],
                'Range Bound': signals['range_bound_signal'],
                'RHS': signals['rhs_signal'],
                'Cup Handle': signals['cup_handle_signal']
            })

    if summary_data:
        df = pd.DataFrame(summary_data)



        # Style the dataframe
        def style_signals(val):
            color,text_color = get_signal_color(val)
            return f'background-color: {color}; color: {text_color}; font-weight: bold;'

        signal_columns = ['Overall Signal', 'SMA', 'Green Candle', 'Range Bound', 'RHS', 'Cup Handle']
        #styled_df = df.style.map(style_pnl, subset='Daily Change %')
        styled_df = df.style.map(style_signals, subset=signal_columns)
        styled_df = styled_df.map(style_pnl, subset='Daily Change %')
        styled_df = styled_df.format({
            'Price': lambda x: format_currency(x),
            'Daily Change %': '{:.2f}%',
            'Fundamental Score': '{:.0f}'
        })

        st.dataframe(styled_df,hide_index=True, use_container_width=True)



        # Performance highlights
        st.subheader("🏆 Performance Highlights")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Top 5 by Fundamental Score**")
            top_fundamental = df.nlargest(5, 'Fundamental Score')[['Company', 'Fundamental Score']]
            st.dataframe(top_fundamental,hide_index=True, use_container_width=True)

        with col2:
            st.write("**Best Daily Performers**")
            best_performers = df.nlargest(5, 'Daily Change %')[['Company', 'Daily Change %']]
            st.dataframe(best_performers,hide_index=True, use_container_width=True)

    else:
        st.info("No data available. Please refresh data first.")

def display_detail_analysis_tab(group_stocks):
    """Display detailed analysis tab"""
    st.subheader("Detailed Analysis")

    # Filters
    col1, col2 = st.columns([1, 1])

    with col1:
        selected_stock = st.selectbox(
            "Select Stock",
            [stock['symbol'] for stock in group_stocks],
            format_func=lambda x: x.replace('.NS', '')
        )

    with col2:
        period = st.selectbox("Period", ["1 Year", "2 Years"])

    if selected_stock:
        display_stock_analysis(selected_stock, period)

def display_stock_analysis(symbol, period):
    """Display comprehensive stock analysis"""
    stock_info = st.session_state.db_manager.get_stock_info(symbol)
    signals = st.session_state.db_manager.get_signals(symbol)

    if not stock_info or not signals:
        st.error("No data available for this stock. Please refresh data.")
        return

    # Fundamental Information
    st.subheader("📊 Fundamental Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Basic Information**")
        st.write(f"**Company:** {stock_info['company_name']}")
        st.write(f"**Current Price:** {format_currency(stock_info['current_price'])}")
        st.write(f"**Market Cap:** {format_currency(stock_info['market_cap'])}")
        st.write(f"**Sector:** {stock_info['sector']}")
        st.write(f"**Industry:** {stock_info['industry']}")

    with col2:
        st.write("**Key Ratios**")
        st.write(f"**PE Ratio:** {stock_info['pe_ratio']:.2f}" if stock_info['pe_ratio'] else "PE Ratio: N/A")
        st.write(f"**ROE:** {stock_info['roe']:.2%}" if stock_info['roe'] else "ROE: N/A")
        st.write(f"**Debt to Equity:** {stock_info['debt_to_equity']:.2f}" if stock_info['debt_to_equity'] else "Debt to Equity: N/A")
        st.write(f"**52W High:** {format_currency(stock_info['fifty_two_week_high'])}")
        st.write(f"**52W Low:** {format_currency(stock_info['fifty_two_week_low'])}")
        st.write(f"**Lifetime High:** {format_currency(stock_info['lifetime_high'])}")

    # Overall View
    st.subheader("🔍 Overall View")
    display_overall_view(stock_info, signals)

    # Technical Strategy Section
    st.subheader("📈 Technical Analysis")
    display_technical_strategies(symbol, period)

def display_overall_view(stock_info, signals):
    """Display overall stock view"""
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Strength & Risk Factors**")

        strengths = []
        risks = []

        if signals['fundamental_score'] > 70:
            strengths.append("Strong fundamental score")
        elif signals['fundamental_score'] < 40:
            risks.append("Weak fundamental score")

        if stock_info['pe_ratio'] and stock_info['pe_ratio'] < 15:
            strengths.append("Attractive valuation (Low PE)")
        elif stock_info['pe_ratio'] and stock_info['pe_ratio'] > 40:
            risks.append("High valuation (High PE)")

        if stock_info['roe'] and stock_info['roe'] > 0.15:
            strengths.append("Strong profitability (High ROE)")
        elif stock_info['roe'] and stock_info['roe'] < 0.05:
            risks.append("Low profitability (Low ROE)")

        if stock_info['debt_to_equity'] and stock_info['debt_to_equity'] < 0.5:
            strengths.append("Low debt levels")
        elif stock_info['debt_to_equity'] and stock_info['debt_to_equity'] > 2:
            risks.append("High debt levels")

        if strengths:
            st.write("**Strengths:**")
            for strength in strengths:
                st.write(f"✅ {strength}")

        if risks:
            st.write("**Risks:**")
            for risk in risks:
                st.write(f"❌ {risk}")

    with col2:
        st.write("**Future Growth & Projections**")

        # Calculate price position
        current_price = stock_info['current_price']
        lifetime_high = stock_info['lifetime_high']
        fifty_two_week_low = stock_info['fifty_two_week_low']

        position_from_high = ((lifetime_high - current_price) / lifetime_high) * 100
        position_from_low = ((current_price - fifty_two_week_low) / fifty_two_week_low) * 100

        st.write(f"**Price Position:**")
        st.write(f"• {position_from_high:.1f}% below lifetime high")
        st.write(f"• {position_from_low:.1f}% above 52-week low")

        # Technical outlook
        buy_signals = sum(1 for signal in [signals['sma_signal'], signals['green_candle_signal'],
                                          signals['range_bound_signal'], signals['rhs_signal'],
                                          signals['cup_handle_signal']] if signal == 'BUY')

        st.markdown("### 🎯 **Technical Outlook:**")
        st.write(f"• {buy_signals}/5 strategies showing BUY signal")
        st.write(f"• Average rating: {signals['average_rating']}")

def display_technical_strategies(symbol, period):
    """Display technical analysis strategies"""
    # Get historical data
    hist_data = st.session_state.db_manager.get_historical_data(symbol)

    if hist_data is None or len(hist_data) == 0:
        st.error("No historical data available")
        return

    # Filter data based on period
    if period == "1 Year":
        hist_data = hist_data.last('365D')
    else:  # 2 Years
        hist_data = hist_data.last('730D')

    # Create tabs for each strategy
    strategy_tabs = st.tabs(["📈 SMA", "🟢 Green Candle", "📊 Range Bound", "👥 Reverse H&S", "☕ Cup Handle"])

    with strategy_tabs[0]:
        display_sma_strategy(symbol, hist_data)

    with strategy_tabs[1]:
        display_green_candle_strategy(symbol, hist_data)

    with strategy_tabs[2]:
        display_range_bound_strategy(symbol, hist_data)

    with strategy_tabs[3]:
        display_rhs_strategy(symbol, hist_data)

    with strategy_tabs[4]:
        display_cup_handle_strategy(symbol, hist_data)

def display_sma_strategy(symbol, hist_data):

    """Display SMA strategy analysis"""
    st.subheader("Simple Moving Average Strategy")

    # Calculate SMA signals
    sma_analysis = st.session_state.tech_analysis.calculate_sma_signal(hist_data)
    signal_color, text_color = get_signal_color(sma_analysis['signal'])

    # Create chart
    fig = create_candlestick_chart(hist_data, f"{symbol.replace('.NS', '')} - SMA Analysis")

    # Add SMA lines
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=sma_analysis['sma_20'],
        name='SMA 20',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=sma_analysis['sma_50'],
        name='SMA 50',
        line=dict(color='orange', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=sma_analysis['sma_200'],
        name='SMA 200',
        line=dict(color='red', width=2)
    ))

    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    # Display strategy rules and details side by side in cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 **Simple Moving Average Strategy**")
        st.markdown(f"""
                    <div style="
                        border: 2px solid #17a2b8; 
                        border-radius: 10px; 
                        padding: 15px; 
                        background-color: #f9f9f9;
                        margin: 10px 0;
                    ">
                        <strong> **Strategy Rules**</strong><br>
                        • Use daily candlestick charts<br>
                        • No stop-loss (hold until target is hit)<br>
                        • Position sizing: 3% of total portfolio per trade<br>
                        • Maximum 3 trades per stock (total 9% of portfolio per stock)<br><br>
                        <strong>**Signal Conditions:**</strong><br>
                        • **SELL:** Price above 20 SMA > 50 SMA > 200 SMA (strong bullish alignment)<br>
                        • **BUY:** Price below 20 SMA < 50 SMA < 200 SMA (strong bearish alignment)<br>
                        • **NEUTRAL:** SMAs not aligned for clear signal<br><br>
                        <strong>**Averaging Rules:**</strong><br>
                        • an average if second entry is at least 10% lower than first entry<br>
                        • Maximum 3 trades per stock allowed<br>
                    </div>
                    """, unsafe_allow_html=True)


    with col2:
        st.markdown("### 🕯️ Strategy Details")
        #signal_color = get_signal_color(signal)
        signal_color, text_color = get_signal_color(sma_analysis['signal'])
        #print({sma_analysis['signal']})
        st.markdown(f"""
                    <div style="
                        border: 2px solid {signal_color}; 
                        border-radius: 10px; 
                        padding: 15px; 
                        background-color: #f9f9f9;
                        margin: 10px 0;
                    ">
                        <strong>Signal:</strong> <span style="color: {text_color}; font-weight: bold;">{sma_analysis['signal']}</span><br>
                        <strong>Current Price:</strong> ₹{format_currency(sma_analysis['current_price'])}<br>
                        <strong>20-Day SMA:</strong> ₹{format_currency(sma_analysis['sma_20'].iloc[-1]) if len(sma_analysis['sma_20']) > 0 else 'N/A'}<br>
                        <strong>50-Day SMA:</strong> ₹{format_currency(sma_analysis['sma_50'].iloc[-1]) if len(sma_analysis['sma_50']) > 0 else 'N/A'}<br>
                        <strong>200-Day SMA:</strong> ₹{format_currency(sma_analysis['sma_200'].iloc[-1]) if len(sma_analysis['sma_200']) > 0 else 'N/A'}<br>
                        <strong>Reasoning:</strong><br>
                        {sma_analysis['reasoning']}
                    </div>
                    """, unsafe_allow_html=True)
        #with st.container():
        #    st.markdown("### 🎯 **Strategy Details**")
        #    st.write(f"**Signal:** {sma_analysis['signal']}")
        #    st.write(f"**Current Price:** {format_currency(sma_analysis['current_price'])}")
        #    st.write(f"**20-Day SMA:** {format_currency(sma_analysis['sma_20'].iloc[-1]) if len(sma_analysis['sma_20']) > 0 else 'N/A'}")
        #    st.write(f"**50-Day SMA:** {format_currency(sma_analysis['sma_50'].iloc[-1]) if len(sma_analysis['sma_50']) > 0 else 'N/A'}")
        #    st.write(f"**200-Day SMA:** {format_currency(sma_analysis['sma_200'].iloc[-1]) if len(sma_analysis['sma_200']) > 0 else 'N/A'}")
        #    st.write(f"**Reasoning:** {sma_analysis['reasoning']}")

    st.divider()





def display_green_candle_strategy(symbol, hist_data):
    """Display Green Candle strategy analysis"""
    st.subheader("Green Candle Strategy (V20)")

    # Calculate Green Candle signals
    gc_analysis = st.session_state.tech_analysis.calculate_green_candle_signal(hist_data)

    # Create chart
    fig = create_candlestick_chart(hist_data, f"{symbol.replace('.NS', '')} - Green Candle Analysis")

    # Add range lines and parent formation markers
    if gc_analysis.get('upper_line') and gc_analysis.get('lower_line'):
        fig.add_hline(y=gc_analysis['upper_line'], line_dash="dash", line_color="green",
                      annotation_text="Upper Line (Parent Formation Top)")
        fig.add_hline(y=gc_analysis['lower_line'], line_dash="dash", line_color="red",
                      annotation_text="Lower Line (Parent Formation Bottom)")

        # Add markers for parent formation extremes
        if gc_analysis.get('parent_formation'):
            formation = gc_analysis['parent_formation']
            fig.add_scatter(x=[formation['start_date']], y=[formation['bottom']],
                            mode='markers', marker=dict(color='red', size=12, symbol='triangle-down'),
                            name='Formation Bottom')
            fig.add_scatter(x=[formation['end_date']], y=[formation['top']],
                            mode='markers', marker=dict(color='green', size=12, symbol='triangle-up'),
                            name='Formation Top')

    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    # Display strategy rules and details side by side in cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 **20%+ Green Candle Strategy**")
        st.markdown(f"""
                            <div style="
                                border: 2px solid #17a2b8; 
                                border-radius: 10px; 
                                padding: 15px; 
                                background-color: #f9f9f9;
                                margin: 10px 0;
                            ">
                                <strong> **Strategy Rules**</strong><br>
                                • Use daily candlestick charts<br>
                                • No stop-loss (hold until target is hit)<br>
                                • Position sizing: 3% of total portfolio per trade<br>
                                • Maximum 3 trades per stock (total 9% of portfolio per stock)<br><br>
                                <strong> **Identify a 'Bunch of Green Candles**</strong><br>
                                • Look for bunch green candles (no red candles in between)<br>
                                • Total price movement from lowest to highest point must be ≥20%<br>
                                      -Lower Line: Lowest point of the green candle bunch<br>
                                      -Upper Line: Highest point of the green candle bunch<br><br>      
                                <strong> **Buy at Lower Line**</strong><br>
                                • When stock retraces and touches the lower line, buy the stock<br>
                                • If it falls further (≥10% below first buy), average (second buy)<br><br>
                                <strong> **Sell at Upper Line**</strong><br>
                                • When stock rises back to the upper line, sell for profit<br>
                                • Each range is independent—once completed, wait for new 20% move<br><br>
                                
                            </div>
                            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎯 **Strategy Details**")
        signal_color, text_color = get_signal_color(gc_analysis['signal'])

        # Build the parent formation HTML separately
        parent_formation_html = "<span></span>"
        if gc_analysis.get('parent_formation'):
            formation = gc_analysis['parent_formation']
            parent_formation_html = f"""
                        <strong>Parent Formation Bottom:</strong> ₹{format_currency(formation['bottom'])}<br>
                        <strong>Parent Formation Top:</strong> ₹{format_currency(formation['top'])}<br>
                        <strong>Formation Range:</strong> {formation['range_percent']:.1f}%<br>
                        <strong>Formation Period:</strong> {formation['start_date']} to {formation['end_date']}<br>
            """

        st.markdown(f"""
                    <div style="
                        border: 2px solid {signal_color}; 
                        border-radius: 10px; 
                        padding: 15px; 
                        background-color: #f9f9f9;
                        margin: 10px 0;">
                        <strong>Signal:</strong> <span style='color: {text_color}; font-weight: bold;'>{gc_analysis['signal']}</span><br>
                        <strong>Current Price:</strong> ₹{format_currency(gc_analysis['current_price'])}<br>
                        <span>{parent_formation_html}</span>
                        <strong>Reasoning:</strong><br>
                        {gc_analysis['reasoning']}
                    </div>
                    """, unsafe_allow_html=True)

    st.divider()





def display_range_bound_strategy(symbol, hist_data):
    """Display Range Bound strategy analysis"""
    st.subheader("Range Bound Trading Strategy")

    # Calculate Range Bound signals
    rb_analysis = st.session_state.tech_analysis.calculate_range_bound_signal(hist_data)

    # Create chart
    fig = create_candlestick_chart(hist_data, f"{symbol.replace('.NS', '')} - Range Bound Analysis")

    # Add support and resistance lines
    if rb_analysis.get('support_level') and rb_analysis.get('resistance_level'):
        fig.add_hline(y=rb_analysis['support_level'], line_dash="dash", line_color="green",
                      annotation_text="Support")
        fig.add_hline(y=rb_analysis['resistance_level'], line_dash="dash", line_color="red",
                      annotation_text="Resistance")

    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    # Display strategy rules and details side by side in cards
    col1, col2 = st.columns(2)
    with col1:
        with col1:
            st.markdown("### 📋 **Range-Bound Trading Strategy**")
            st.markdown(f"""
                                <div style="
                                    border: 2px solid #17a2b8; 
                                    border-radius: 10px; 
                                    padding: 15px; 
                                    background-color: #f9f9f9;
                                    margin: 10px 0;
                                ">
                                    <strong>Chart Setup:</strong><br>
                                    • Use daily timeframe for range-bound trading<br>
                                    • Each level (Support/Resistance) must be touched at least twice<br>
                                    • Price must alternate between Support and Resistance (zig-zag movement)<br>
                                    • Trade only if range is at least 20% (avoid smaller ranges below 15%)<br><br>
                                    <strong>Entry and Exit Rules:</strong><br>
                                    • <strong>Buy Near Support:</strong> Enter when price touches Support (with volume confirmation)<br>
                                    • <strong>Sell Near Resistance:</strong> Exit near Resistance or if fundamentals deteriorate<br>
                                    • <strong>Invalid Range:</strong> If only one side is touched twice without the other<br><br>
                                    <strong>Fundamental Check:</strong><br>
                                    • During range-bound period, company financials should improve<br>
                                    • Rising revenue, net profit, and declining NPA (Non-Performing Assets)<br><br>
                                    <strong>Trend Reversal Signs:</strong><br>
                                    • Look for Higher Highs & Higher Lows after a downtrend<br>
                                    • Historically, if previous ranges took 8-10 months to break, expect similar timeframe<br>
                                </div>
                                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 🎯 **Strategy Details**")
            signal_color, text_color = get_signal_color(rb_analysis['signal'])

            # Build the range levels HTML separately
            range_levels_html = "<span></span>"
            if rb_analysis.get('support_level') and rb_analysis.get('resistance_level'):
                range_levels_html = f"""
                            <strong>Support Level:</strong> ₹{format_currency(rb_analysis['support_level'])}<br>
                            <strong>Resistance Level:</strong> ₹{format_currency(rb_analysis['resistance_level'])}<br>
                            <strong>Range Strength:</strong> {rb_analysis.get('range_strength', 0):.1f}%<br>"""

            st.markdown(f"""
                        <div style="
                            border: 2px solid {signal_color}; 
                            border-radius: 10px; 
                            padding: 15px; 
                            background-color: #f9f9f9;
                            margin: 10px 0;">
                            <strong>Signal:</strong> <span style='color: {text_color}; font-weight: bold;'>{rb_analysis['signal']}</span><br>
                            <strong>Current Price:</strong> ₹{format_currency(rb_analysis['current_price'])}<br>
                            <span>{range_levels_html}</span>
                            <strong>Reasoning:</strong><br>
                            {rb_analysis['reasoning']}
                        </div>
                        """, unsafe_allow_html=True)

    st.divider()



    # Display strategy details
    #st.write("**Strategy Details:**")
    #st.write(f"**Signal:** {rb_analysis['signal']}")

    #if rb_analysis.get('support_level') and rb_analysis.get('resistance_level'):
        #st.write(f"**Support Level:** {format_currency(rb_analysis['support_level'])}")
        #st.write(f"**Resistance Level:** {format_currency(rb_analysis['resistance_level'])}")
        #st.write(f"**Range Strength:** {rb_analysis.get('range_strength', 0):.1f}%")



def display_rhs_strategy(symbol, hist_data):
    """Display Reverse Head and Shoulder strategy analysis"""
    st.subheader("Reverse Head and Shoulder Pattern")

    # Calculate RHS signals
    rhs_analysis = st.session_state.tech_analysis.calculate_rhs_signal(hist_data)
    # Create chart
    fig = create_candlestick_chart(hist_data, f"{symbol.replace('.NS', '')} - RHS Pattern Analysis")

    # Add pattern lines and points
    if rhs_analysis.get('pattern_points'):
        points = rhs_analysis['pattern_points']

        # Add neckline
        if points.get('neckline'):
            fig.add_hline(y=points['neckline'], line_dash="dash", line_color="blue",
                          annotation_text="Neckline")

        # Add target line
        if points.get('target_price'):
            fig.add_hline(y=points['target_price'], line_dash="dot", line_color="green",
                          annotation_text="Target Price")

        # Mark pattern points
        if points.get('left_shoulder'):
            fig.add_scatter(x=[points['left_shoulder']['date']], y=[points['left_shoulder']['price']],
                            mode='markers', marker=dict(color='orange', size=12, symbol='diamond'),
                            name='Left Shoulder')

        if points.get('head'):
            fig.add_scatter(x=[points['head']['date']], y=[points['head']['price']],
                            mode='markers', marker=dict(color='red', size=12, symbol='triangle-down'), name='Head')

        if points.get('right_shoulder'):
            fig.add_scatter(x=[points['right_shoulder']['date']], y=[points['right_shoulder']['price']],
                            mode='markers', marker=dict(color='orange', size=12, symbol='diamond'),
                            name='Right Shoulder')

    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    # Display strategy rules and details side by side in cards
    col1, col2 = st.columns(2)

    with col1:
        with col1:
            st.markdown("### 📋 **Pattern Rules**")
            st.markdown(f"""
                                <div style="
                                    border: 2px solid #17a2b8; 
                                    border-radius: 10px; 
                                    padding: 15px; 
                                    background-color: #f9f9f9;
                                    margin: 10px 0;
                                ">
                                    <strong>Pattern Shape:</strong><br>
                                    • <strong>Left Shoulder:</strong> Price falls, then rises<br>
                                    • <strong>Head:</strong> Price falls deeper than the left shoulder, then rises<br>
                                    • <strong>Right Shoulder:</strong> Price falls again (but not as deep as the head), then rises<br>
                                    • <strong>Neckline:</strong> Horizontal line connecting start of left shoulder, end of left shoulder, and end of head<br>
                                    • <strong>Key Points:</strong> Neckline must be horizontal, right shoulder cannot be deeper than head<br><br>
                                    <strong>Buying Conditions:</strong><br>
                                    • Right shoulder forms a base (price consolidates in tight range)<br>
                                    • Breakout occurs above the base range<br>
                                    • Breakout candle is green (bullish) and closes above the range<br>
                                    • Buy next day after confirmation (or same day at closing if monitoring live)<br><br>
                                    <strong>Target Calculation:</strong><br>
                                    • Measure depth from lowest point of head to neckline<br>
                                    • Add this depth above neckline to get target<br><br>
                                    <strong>Sell Rules:</strong><br>
                                    • If technical target < lifetime high, sell at lifetime high<br>
                                    • If technical target > lifetime high, sell at technical target<br>
                                </div>
                                """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎯 **Strategy Details**")
        signal_color, text_color = get_signal_color(rhs_analysis['signal'])

        # Build the pattern points HTML separately
        pattern_points_html = "<span></span>"
        if rhs_analysis.get('pattern_points'):
            points = rhs_analysis['pattern_points']
            pattern_details = []

            if points.get('left_shoulder'):
                pattern_details.append(
                    f"<strong>Left Shoulder:</strong> {points['left_shoulder']['date']} - {format_currency(points['left_shoulder']['price'])}")

            if points.get('head'):
                pattern_details.append(
                    f"<strong>Head:</strong> {points['head']['date']} - {format_currency(points['head']['price'])}")

            if points.get('right_shoulder'):
                pattern_details.append(
                    f"<strong>Right Shoulder:</strong> {points['right_shoulder']['date']} - {format_currency(points['right_shoulder']['price'])}")

            if points.get('neckline'):
                pattern_details.append(f"<strong>Neckline:</strong> {format_currency(points['neckline'])}")

            if points.get('target_price'):
                pattern_details.append(f"<strong>Target Price:</strong> {format_currency(points['target_price'])}")

            if pattern_details:
                pattern_points_html = "<br>".join(pattern_details) + "<br>"

        st.markdown(f"""
                    <div style="
                        border: 2px solid {signal_color}; 
                        border-radius: 10px; 
                        padding: 15px; 
                        background-color: #f9f9f9;
                        margin: 10px 0;">
                        <strong>Signal:</strong> <span style='color: {text_color}; font-weight: bold;'>{rhs_analysis['signal']}</span><br>
                        <strong>Status:</strong> {rhs_analysis.get('status', 'No Pattern found')}<br>
                        <span>{pattern_points_html}</span>
                        <strong>Reasoning:</strong><br>
                        {rhs_analysis['reasoning']}
                    </div>
                    """, unsafe_allow_html=True)
    st.divider()



    # Display pattern details
    #st.write("**Pattern Details:**")
    #st.write(f"**Signal:** {rhs_analysis['signal']}")
    #st.write(f"**Status:** {rhs_analysis.get('status', 'N/A')}")

    #if rhs_analysis.get('pattern_points'):
    #    points = rhs_analysis['pattern_points']
    #    st.write("**Pattern Points:**")

    #    if points.get('left_shoulder'):
    #        st.write(f"**Left Shoulder:** {points['left_shoulder']['date']} - {format_currency(points['left_shoulder']['price'])}")

    #    if points.get('head'):
    #        st.write(f"**Head:** {points['head']['date']} - {format_currency(points['head']['price'])}")

    #    if points.get('right_shoulder'):
    #        st.write(f"**Right Shoulder:** {points['right_shoulder']['date']} - {format_currency(points['right_shoulder']['price'])}")

    #    if points.get('neckline'):
    #        st.write(f"**Neckline:** {format_currency(points['neckline'])}")

    #    if points.get('target_price'):
    #        st.write(f"**Target Price:** {format_currency(points['target_price'])}")

def display_cup_handle_strategy(symbol, hist_data):
    """Display Cup with Handle strategy analysis"""
    st.subheader("Cup with Handle Pattern")

    # Calculate Cup Handle signals
    ch_analysis = st.session_state.tech_analysis.calculate_cup_handle_signal(hist_data)

    # Create chart
    fig = create_candlestick_chart(hist_data, f"{symbol.replace('.NS', '')} - Cup with Handle Analysis")

    # Add pattern lines
    if ch_analysis.get('pattern_points'):
        points = ch_analysis['pattern_points']

        # Add neckline
        if points.get('neckline'):
            fig.add_hline(y=points['neckline'], line_dash="dash", line_color="blue",
                          annotation_text="Neckline")

        # Add target line
        if points.get('target_price'):
            fig.add_hline(y=points['target_price'], line_dash="dot", line_color="green",
                          annotation_text="Target Price")

        # Mark cup and handle points
        if points.get('cup_start'):
            fig.add_scatter(x=[points['cup_start']['date']], y=[points['cup_start']['price']],
                            mode='markers', marker=dict(color='blue', size=12, symbol='circle'), name='Cup Start')

        if points.get('cup_low'):
            fig.add_scatter(x=[points['cup_low']['date']], y=[points['cup_low']['price']],
                            mode='markers', marker=dict(color='red', size=12, symbol='triangle-down'), name='Cup Low')

        if points.get('handle_low'):
            fig.add_scatter(x=[points['handle_low']['date']], y=[points['handle_low']['price']],
                            mode='markers', marker=dict(color='orange', size=12, symbol='square'), name='Handle Low')

    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    # Display strategy rules and details side by side in cards
    col1, col2 = st.columns(2)

    with col1:
        with col1:
            st.markdown("### 📋 **Pattern Rules**")
            st.markdown(f"""
                                <div style="
                                    border: 2px solid #17a2b8; 
                                    border-radius: 10px; 
                                    padding: 15px; 
                                    background-color: #f9f9f9;
                                    margin: 10px 0;
                                ">
                                    <strong>Pattern Shape:</strong><br>
                                    • <strong>Cup:</strong> U-shaped or V-shaped decline + recovery<br>
                                    • <strong>Handle:</strong> Small pullback on right side (must be smaller than the cup)<br>
                                    • <strong>Neckline:</strong> Connects start and end of the cup and handle (must be horizontal)<br><br>
                                    <strong>Buying Conditions:</strong><br>
                                    ✅ Handle forms a base (consolidation)<br>
                                    ✅ Breakout above the base with a green closing candle<br>
                                    ✅ Buy next day (or at closing)<br><br>
                                    <strong>Target Calculation:</strong><br>
                                    • Measure depth from cup low to neckline<br>
                                    • Add this depth above neckline = Target<br><br>
                                    <strong>Sell Rule:</strong><br>
                                    • Always sell at the technical target (unlike RHS, do not adjust for lifetime high)<br>
                                </div>
                                """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎯 **Strategy Details**")
        signal_color, text_color = get_signal_color(ch_analysis['signal'])

        # Build the pattern points HTML separately
        pattern_points_html = "<span></span>"
        if ch_analysis.get('pattern_points'):
            points = ch_analysis['pattern_points']
            pattern_details = []

            if points.get('cup_low'):
                pattern_details.append(
                    f"<strong>Cup Low:</strong> {points['cup_low']['date']} - {format_currency(points['cup_low']['price'])}")

            if points.get('handle_low'):
                pattern_details.append(
                    f"<strong>Handle Low:</strong> {points['handle_low']['date']} - {format_currency(points['handle_low']['price'])}")

            if points.get('neckline'):
                pattern_details.append(f"<strong>Neckline:</strong> {format_currency(points['neckline'])}")

            if points.get('target_price'):
                pattern_details.append(f"<strong>Target Price:</strong> {format_currency(points['target_price'])}")

            if pattern_details:
                pattern_points_html = "<br>".join(pattern_details) + "<br>"

        st.markdown(f"""
                    <div style="
                        border: 2px solid {signal_color}; 
                        border-radius: 10px; 
                        padding: 15px; 
                        background-color: #f9f9f9;
                        margin: 10px 0;">
                        <strong>Signal:</strong> <span style='color: {text_color}; font-weight: bold;'>{ch_analysis['signal']}</span><br>
                        <strong>Status:</strong> {ch_analysis.get('status', 'No Pattern found')}<br>
                        <span>{pattern_points_html}</span>
                        <strong>Reasoning:</strong><br>
                        {ch_analysis['reasoning']}
                    </div>
                    """, unsafe_allow_html=True)

    st.divider()



    # Display pattern details
    #st.write("**Pattern Details:**")
    #st.write(f"**Signal:** {ch_analysis['signal']}")
    #st.write(f"**Status:** {ch_analysis.get('status', 'N/A')}")

    #if ch_analysis.get('pattern_points'):
    #    points = ch_analysis['pattern_points']
    #    st.write("**Pattern Points:**")

    #    if points.get('cup_start'):
    #        st.write(f"**Cup Start:** {points['cup_start']['date']} - {format_currency(points['cup_start']['price'])}")

    #    if points.get('cup_bottom'):
    #        st.write(f"**Cup Bottom:** {points['cup_bottom']['date']} - {format_currency(points['cup_bottom']['price'])}")

    #    if points.get('handle_low'):
    #        st.write(f"**Handle Low:** {points['handle_low']['date']} - {format_currency(points['handle_low']['price'])}")

    #    if points.get('neckline'):
    #        st.write(f"**Neckline:** {format_currency(points['neckline'])}")

    #    if points.get('target_price'):
    #        st.write(f"**Target Price:** {format_currency(points['target_price'])}")

def portfolio_interface():
    """Interface for Portfolio Management"""
    st.header("💼 Portfolio Management")

    # Display portfolio
    st.subheader("Portfolio Dashboard")

    portfolio_stocks = st.session_state.portfolio_manager.get_portfolio_stocks()

    if portfolio_stocks:
        # Create portfolio dataframe
        portfolio_data = []
        total_investment = 0
        total_current_value = 0

        for stock in portfolio_stocks:
            stock_info = st.session_state.db_manager.get_stock_info(stock['symbol'])
            signals = st.session_state.db_manager.get_signals(stock['symbol'])

            if stock_info and signals:
                current_price = stock_info['current_price']
                investment = stock['quantity'] * stock['avg_buy_price']
                current_value = stock['quantity'] * current_price
                pnl = current_value - investment
                pnl_percent = (pnl / investment) * 100

                total_investment += investment
                total_current_value += current_value

                portfolio_data.append({
                    'Company': stock_info['company_name'],
                    'Symbol': stock['symbol'].replace('.NS', ''),
                    'Quantity': stock['quantity'],
                    'Buy Price': stock['avg_buy_price'],
                    'Current Price': current_price,
                    'Investment': investment,
                    'Current Value': current_value,
                    'P&L': pnl,
                    'P&L %': pnl_percent,
                    'SMA': signals['sma_signal'],
                    'Green Candle': signals['green_candle_signal'],
                    'Range Bound': signals['range_bound_signal'],
                    'RHS': signals['rhs_signal'],
                    'Cup Handle': signals['cup_handle_signal']
                })

        if portfolio_data:
            df = pd.DataFrame(portfolio_data)

            # Display portfolio summary
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Investment", format_currency(total_investment))

            with col2:
                st.metric("Current Value", format_currency(total_current_value))

            with col3:
                total_pnl = total_current_value - total_investment
                total_pnl_percent = (total_pnl / total_investment) * 100 if total_investment > 0 else 0
                #st.metric("Total P&L", format_currency(total_pnl))
                st.metric("Total P&L", format_currency(total_pnl), f"{total_pnl_percent:.2f}%")

                #st.metric("Total P&L %", f"{total_pnl_percent:.2f}%")

            with col4:
                st.metric("Portfolio Stocks", len(portfolio_data))

            # Style the portfolio dataframe
            #def style_pnl(val):
            #    if val > 0:
            #        return 'color: green; font-weight: bold;'
            #    elif val < 0:
            #        return 'color: red; font-weight: bold;'
            #    return ''

            # Style the dataframe
            def style_signals(val):
                color, text_color = get_signal_color(val)
                return f'background-color: {color}; color: {text_color}; font-weight: bold;'
            #def style_signals(val):
            #    color = get_signal_color(val)
            #    return f'background-color: {color}; color: black; font-weight: bold;'

            signal_columns = ['SMA', 'Green Candle', 'Range Bound', 'RHS', 'Cup Handle']
            pnl_columns = ['P&L', 'P&L %']

            styled_df = df.style.map(style_signals, subset=signal_columns)
            styled_df = styled_df.map(style_pnl, subset=pnl_columns)
            styled_df = styled_df.format({
                'Buy Price': lambda x: format_currency(x),
                'Current Price': lambda x: format_currency(x),
                'Investment': lambda x: format_currency(x),
                'Current Value': lambda x: format_currency(x),
                'P&L': lambda x: format_currency(x),
                'P&L %': '{:.2f}%'
            })

            st.dataframe(styled_df, hide_index=True,use_container_width=True)

            # Performance highlights
            st.subheader("🏆 Portfolio Highlights")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Top 5 by Fundamental Score**")
                # Get fundamental scores for portfolio stocks
                portfolio_with_scores = []
                for stock in portfolio_stocks:
                    signals = st.session_state.db_manager.get_signals(stock['symbol'])
                    if signals:
                        stock_info = st.session_state.db_manager.get_stock_info(stock['symbol'])
                        portfolio_with_scores.append({
                            'Company': stock_info['company_name'] if stock_info else stock['symbol'],
                            'Fundamental Score': signals['fundamental_score']
                        })

                if portfolio_with_scores:
                    scores_df = pd.DataFrame(portfolio_with_scores)
                    top_scores = scores_df.nlargest(5, 'Fundamental Score')
                    st.dataframe(top_scores, hide_index=True,use_container_width=True)

            with col2:
                st.write("**Top 5 Performers by P&L %**")
                top_performers = df.nlargest(5, 'P&L %')[['Company', 'P&L %']]
                st.dataframe(top_performers,hide_index=True, use_container_width=True)


    else:
        st.info("No stocks in portfolio. Add some stocks to get started.")

if __name__ == "__main__":
    main()
