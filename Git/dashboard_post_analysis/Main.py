
"""
Trading Strategy Dashboard - Main Application

A comprehensive dashboard for analyzing trading strategy performance,
risk metrics, and market conditions with real-time data visualization.
"""

import os
import streamlit as st
import pandas as pd
from typing import Optional, Tuple
import unicodedata

from display_class import Display
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

REQUIRED_FILES = [
    BASE_DIR / 'Trade_report.xlsx',
    BASE_DIR / 'Trade_report_backtest.xlsx',
    BASE_DIR / 'Market_data.xlsx'
]

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apple-style CSS with single color palette
st.markdown("""
<style>
    /* Apple-style Design System - Single Color Palette */
    :root {
        --primary-50: #f0f9ff;
        --primary-100: #e0f2fe;
        --primary-200: #bae6fd;
        --primary-300: #7dd3fc;
        --primary-400: #38bdf8;
        --primary-500: #0ea5e9;
        --primary-600: #0284c7;
        --primary-700: #0369a1;
        --primary-800: #075985;
        --primary-900: #0c4a6e;
        --primary-950: #082f49;
        
        --gray-50: #f8fafc;
        --gray-100: #f1f5f9;
        --gray-200: #e2e8f0;
        --gray-300: #cbd5e1;
        --gray-400: #94a3b8;
        --gray-500: #64748b;
        --gray-600: #475569;
        --gray-700: #334155;
        --gray-800: #1e293b;
        --gray-900: #0f172a;
        --gray-950: #020617;
    }
    
    .main {
        background: var(--gray-50);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stApp {
        background: var(--gray-50);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--gray-200);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--gray-100);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-400);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-500);
    }
    
    /* Streamlit elements */
    .stSelectbox, .stSlider, .stRadio {
        border: 1px solid var(--gray-200);
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox > div > div {
        color: var(--gray-900);
        border-radius: 6px;
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        background: transparent !important;
    }
    
    /* Custom container styling */
    .metric-container {
        background: rgba(255,255,255,0.9);
        border: 1px solid var(--gray-200);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Animation for cards */
    .metric-container:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.2s ease;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: var(--primary-600);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stButton > button:hover {
        background: var(--primary-700);
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.25);
    }
    
    /* Sidebar navigation styling */
    .css-1lcbmhc {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(20px);
    }
    
    /* Page title styling */
    .css-10trblm {
        color: var(--gray-900);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-weight: 600;
    }
    
    /* Label styling */
    .stMarkdown p {
        color: var(--gray-700);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    /* Sidebar-specific date input styling */
    .css-1d391kg .stDateInput > div > div {
        background: rgba(255,255,255,0.95) !important;
        border: 1px solid var(--gray-200) !important;
        border-radius: 8px !important;
    }
    
    .css-1d391kg .stDateInput input {
        background: rgba(255,255,255,0.95) !important;
        color: var(--gray-900) !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    
    .css-1d391kg .stDateInput label {
        color: var(--gray-700) !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# Constants
REQUIRED_FILES = ['Market_data.xlsx', 'Trade_report.xlsx', 'Trade_report_backtest.xlsx']
EXCEL_ENGINES = ['openpyxl', 'xlrd', 'odf']
CACHE_TTL = 3600  # 1 hour


@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for file in REQUIRED_FILES:
        if not file.exists():
            st.error(f"File {file.name} does not exist.")
            st.stop()

        if file.stat().st_size == 0:
            st.warning(f"File {file.name} is empty.")

    try:
        filename = "Trade_report.xlsx"
        normalized_filename = unicodedata.normalize('NFC', filename)
        trade_book = pd.read_excel(normalized_filename)
        filename = "Trade_report_backtest.xlsx"
        normalized_filename = unicodedata.normalize('NFC', filename)
        trade_book_backtest = pd.read_excel(normalized_filename)
        filename = "Market_data.xlsx"
        normalized_filename = unicodedata.normalize('NFC', filename)
        market_data = pd.read_excel(normalized_filename)

        return trade_book, trade_book_backtest, market_data

    except Exception as e:
        st.error(str(e))
        st.stop()

def _load_market_data(engine: Optional[str] = None) -> pd.DataFrame:
    """
    Load market data with error handling.
    
    Args:
        engine: Excel engine to use (optional)
        
    Returns:
        Market data DataFrame or empty DataFrame if loading fails
    """
    try:
        if engine:
            market_data = pd.read_excel('Market_data.xlsx', engine=engine)
        else:
            market_data = pd.read_excel('Market_data.xlsx')
            
        if market_data.empty:
            st.warning("Market_data.xlsx is empty. Creating empty DataFrame.")
            
        return market_data
        
    except Exception as market_error:
        st.warning(f"Could not load Market_data.xlsx: {str(market_error)}")
        st.warning("Creating empty market data DataFrame.")
        return pd.DataFrame()

def main():
    """Main application function."""
    # Load and preprocess data
    trade_book_raw, trade_book_backtest_raw, market_data_raw = load_data()
    tb, md = Display.preprocess(trade_book_raw, market_data_raw)
    tb_backtest, _ = Display.preprocess(trade_book_backtest_raw, market_data_raw)
    
    # Store data in session state for pages to access
    st.session_state['trade_book'] = tb
    st.session_state['trade_book_backtest'] = tb_backtest
    st.session_state['market_data'] = md
    
    # Render sidebar with filters and metrics
    _render_sidebar(tb, md)
    
    # Render main content
    _render_main_content()


def _render_sidebar(tb: pd.DataFrame, md: pd.DataFrame) -> None:
    """Render the sidebar with filters and quick stats."""
    with st.sidebar:
        _render_cache_button()
        selected_range, selected_symbols = _render_filters(tb)
        _apply_filters(tb, md, selected_range, selected_symbols)
        _render_quick_stats(tb)


def _render_cache_button() -> None:
    """Render the cache refresh button."""
    if st.button("🔄 Refresh Data Cache", help="Clear cached data to reload with latest changes"):
        st.cache_data.clear()
        st.rerun()


def _render_filters(tb: pd.DataFrame) -> Tuple[Optional[Tuple[pd.Timestamp, pd.Timestamp]], list]:
    """Render date and symbol filters."""
    # Date filters
    min_date = pd.to_datetime(tb["Close_time"].min()) if not tb.empty else None
    max_date = pd.to_datetime(tb["Close_time"].max()) if not tb.empty else None
    selected_range = None
    
    if min_date and max_date and not pd.isna(min_date) and not pd.isna(max_date):
        start_date = st.date_input(
            "Start Date",
            value=min_date.date(),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        
        end_date = st.date_input(
            "End Date",
            value=max_date.date(),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        
        if start_date and end_date:
            selected_range = (pd.Timestamp(start_date), pd.Timestamp(end_date))
    
    # Symbols filter
    symbols = sorted([s for s in tb["Symbol"].dropna().unique()]) if "Symbol" in tb.columns else []
    selected_symbols = st.multiselect("Select symbols", options=symbols, default=symbols)
    
    return selected_range, selected_symbols


def _apply_filters(tb: pd.DataFrame, md: pd.DataFrame, 
                  selected_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]], 
                  selected_symbols: list) -> None:
    """Apply filters to data and update session state."""
    # Store filters in session state
    st.session_state['date_range'] = selected_range
    st.session_state['selected_symbols'] = selected_symbols
    
    # Apply filters
    filtered_tb = tb.copy()
    filtered_md = md.copy()
    
    if selected_symbols:
        filtered_tb = filtered_tb[filtered_tb["Symbol"].isin(selected_symbols)].copy()
    
    if selected_range is not None:
        start, end = pd.to_datetime(selected_range[0]), pd.to_datetime(selected_range[1])
        filtered_tb = filtered_tb[(filtered_tb["Close_time"] >= start) & (filtered_tb["Close_time"] <= end)].copy()
        filtered_md = filtered_md[(filtered_md["timestamp"] >= start) & (filtered_md["timestamp"] <= end)].copy()
    
    # Update session state with filtered data
    st.session_state['trade_book'] = filtered_tb
    st.session_state['market_data'] = filtered_md


def _render_quick_stats(tb: pd.DataFrame) -> None:
    """Render quick statistics cards in sidebar."""
    # Use gross PnL for total PnL calculation
    total_pnl = tb["gross_pnl"].sum() if "gross_pnl" in tb.columns else tb["pnl"].sum()
    win_rate = tb["is_win"].mean() * 100
    total_trades = len(tb)
    
    stats_data = [
        (total_pnl, "Total PnL", "#34c759", "#30d158"),
        (win_rate, "Win Rate", "#007aff", "#5856d6"),
        (total_trades, "Total Trades", "#ff9500", "#ff9f0a")
    ]
    
    for value, label, color1, color2 in stats_data:
        _render_stat_card(value, label, color1, color2)


def _render_stat_card(value: float, label: str, color1: str, color2: str) -> None:
    """Render a single statistics card."""
    if label == "Win Rate":
        display_value = f"{value:.1f}%"
    elif label == "Total PnL":
        display_value = f"${value:,.0f}"
    else:
        display_value = f"{value}"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color1} 0%, {color2} 100%);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(52,199,89,0.2);
        border: 1px solid rgba(52,199,89,0.1);
        margin-bottom: 1rem;">
        <h4 style="
            color: white;
            margin: 0;
            font-size: 1.2rem;
            font-weight: 600;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            {display_value}
        </h4>
        <p style="
            color: rgba(255,255,255,0.9);
            margin: 0;
            font-size: 0.8rem;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            {label}
        </p>
    </div>
    """, unsafe_allow_html=True)


def _render_main_content() -> None:
    """Render the main content area with header and page overview."""
    _render_header()
    _render_page_overview()


def _render_header() -> None:
    """Render the main dashboard header."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, var(--gray-50) 0%, #ffffff 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid var(--gray-200);">
        <h1 style="
            color: var(--gray-900);
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            Trading Strategy Dashboard
        </h1>
        <p style="
            color: var(--gray-600);
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            text-align: center;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            Comprehensive Analysis & Risk Management Platform
        </p>
    </div>
    """, unsafe_allow_html=True)


def _render_page_overview() -> None:
    """Render overview cards for each analysis page."""
    col1, col2 = st.columns(2)
    
    page_info = [
        ("Global Performance", "Quick assessment of your strategy's profitability, activity, and behavior."),
        ("Strategy Optimization", "Optimize and scale your strategy with Kelly criterion and Monte Carlo analysis."),
        ("Statistical Insight", "Deep statistical analysis to understand why your strategy works or fails."),
        ("Survivability & Risk", "Assess long-term durability and exposure to blow-up risk.")
    ]
    
    with col1:
        for title, description in page_info[:2]:
            _render_page_card(title, description)
    
    with col2:
        for title, description in page_info[2:]:
            _render_page_card(title, description)


def _render_page_card(title: str, description: str) -> None:
    """Render a single page overview card."""
    st.markdown(f"""
    <div style="
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid var(--gray-200);">
        <h3 style="
            color: var(--primary-600);
            margin: 0 0 1rem 0;
            font-size: 1.3rem;
            font-weight: 600;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            {title}
        </h3>
        <p style="
            color: var(--gray-600);
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            {description}
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()