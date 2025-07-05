import plotly.graph_objects as go
from plotly.subplots import make_subplots

def format_currency(amount):
    """Format currency with Indian Rupee symbol"""
    if amount >= 10000000:  # 1 crore
        return f"₹{amount/10000000:.2f}Cr"
    elif amount >= 100000:  # 1 lakh
        return f"₹{amount/100000:.2f}L"
    elif amount >= 1000:  # 1 thousand
        return f"₹{amount/1000:.2f}K"
    else:
        return f"₹{amount:.2f}"

def get_signal_color(signal):
    """Get color code for signal"""
    if signal == 'BUY':
        return '#d4edda', '#155724' # Green
    elif signal == 'SELL':
        return '#f8d7da','#721c24' # Red
    elif signal == 'WATCH':
        return '#e0eb8d','#839407'  # Yellow
    elif signal == 'HOLD':
        return '#bef9fa','#1aa8ab'  # Yellow
    else:  # NEUTRAL
        return '#fbf8f8','#8a8686' # Gray

def create_candlestick_chart(hist_data, title):
    """Create a candlestick chart"""
    fig = go.Figure(data=go.Candlestick(
        x=hist_data.index,
        open=hist_data['Open'],
        high=hist_data['High'],
        low=hist_data['Low'],
        close=hist_data['Close'],
        name='Price'
    ))
    
    fig.update_layout(
        title=title,
        yaxis_title='Price (₹)',
        xaxis_title='Date',
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    return fig

def calculate_daily_change(hist_data):
    """Calculate daily price change"""
    if len(hist_data) < 2:
        return 0
    
    current_price = hist_data['Close'].iloc[-1]
    previous_price = hist_data['Close'].iloc[-2]
    
    return ((current_price - previous_price) / previous_price) * 100

def calculate_returns(hist_data, period_days=30):
    """Calculate returns over a specified period"""
    if len(hist_data) < period_days:
        return 0
    
    current_price = hist_data['Close'].iloc[-1]
    past_price = hist_data['Close'].iloc[-period_days]
    
    return ((current_price - past_price) / past_price) * 100

def get_price_trend(hist_data, period_days=20):
    """Get price trend over specified period"""
    if len(hist_data) < period_days:
        return 'NEUTRAL'
    
    recent_data = hist_data.tail(period_days)
    
    # Calculate simple trend
    start_price = recent_data['Close'].iloc[0]
    end_price = recent_data['Close'].iloc[-1]
    
    change_percent = ((end_price - start_price) / start_price) * 100
    
    if change_percent > 5:
        return 'UPTREND'
    elif change_percent < -5:
        return 'DOWNTREND'
    else:
        return 'SIDEWAYS'

def validate_stock_symbol(symbol):
    """Validate stock symbol format"""
    if not symbol:
        return False
    
    # Check if symbol ends with .NS for Indian stocks
    if not symbol.endswith('.NS'):
        return False
    
    # Check if symbol has valid format (letters and numbers)
    base_symbol = symbol[:-3]  # Remove .NS
    if not base_symbol.isalnum():
        return False
    
    return True

def format_percentage(value):
    """Format percentage values"""
    if value > 0:
        return f"+{value:.2f}%"
    else:
        return f"{value:.2f}%"

def get_risk_level(pe_ratio, debt_to_equity, roe):
    """Calculate risk level based on fundamental metrics"""
    risk_score = 0
    
    # PE ratio risk
    if pe_ratio > 30:
        risk_score += 2
    elif pe_ratio > 20:
        risk_score += 1
    
    # Debt to equity risk
    if debt_to_equity > 1.5:
        risk_score += 2
    elif debt_to_equity > 1.0:
        risk_score += 1
    
    # ROE risk (inverse)
    if roe < 0.05:
        risk_score += 2
    elif roe < 0.10:
        risk_score += 1
    
    if risk_score >= 4:
        return 'HIGH'
    elif risk_score >= 2:
        return 'MEDIUM'
    else:
        return 'LOW'
