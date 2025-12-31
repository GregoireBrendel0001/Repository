
"""
Trading Strategy Dashboard - Main Application

A comprehensive dashboard for analyzing trading strategy performance,
risk metrics, and market conditions with real-time data visualization.
"""

import os
import streamlit as st
import pandas as pd
from typing import Optional, Tuple
import json

from display_class import Display

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


def get_all_trade_book_ids() -> list:
    """
    Query MongoDB to get all available trade book IDs with metadata.
    
    Returns:
        List of dictionaries with trade_book_id, symbol, created_at, and nb_trades
    """
    try:
        from dotenv import dotenv_values
        import pymongo
        
        config = dotenv_values(os.path.join(os.path.dirname(__file__), '.env'))
        mongo_client = pymongo.MongoClient(
            config["MONGO_URI"], 
            int(config["MONGO_PORT"]), 
            username=config["MONGO_USER"], 
            password=config["MONGO_PASSWORD"]
        )
        db = mongo_client.get_database(config["MONGO_DB"])
        
        # Find all documents sorted by created_at descending
        docs = db["BookOfTrade"].find(
            {},
            {"_id": 1, "symbol": 1, "created_at": 1, "nb_trades": 1}
        ).sort("created_at", -1)
        
        trade_books = []
        for doc in docs:
            trade_books.append({
                "trade_book_id": doc.get("_id"),
                "symbol": doc.get("symbol", "N/A"),
                "created_at": doc.get("created_at"),
                "nb_trades": doc.get("nb_trades", 0)
            })
        
        mongo_client.close()
        return trade_books
    except Exception as e:
        st.error(f"Error querying MongoDB for trade book IDs: {e}")
        return []


def get_latest_trade_book_id(symbol: str = None) -> Optional[str]:
    """
    Query MongoDB to get the latest trade book ID.
    
    Args:
        symbol: Optional symbol to filter by (e.g., 'FTTUSDT')
    
    Returns:
        The latest trade book ID or None if not found
    """
    try:
        from dotenv import dotenv_values
        import pymongo
        
        config = dotenv_values(os.path.join(os.path.dirname(__file__), '.env'))
        mongo_client = pymongo.MongoClient(
            config["MONGO_URI"], 
            int(config["MONGO_PORT"]), 
            username=config["MONGO_USER"], 
            password=config["MONGO_PASSWORD"]
        )
        db = mongo_client.get_database(config["MONGO_DB"])
        
        # Build query
        query = {}
        if symbol:
            query["symbol"] = symbol
        
        # Find the latest document by created_at
        latest_doc = db["BookOfTrade"].find_one(
            query,
            sort=[("created_at", -1)]  # Sort by created_at descending
        )
        
        mongo_client.close()
        
        if latest_doc:
            return latest_doc.get("_id")
        return None
    except Exception as e:
        st.error(f"Error querying MongoDB for latest trade book ID: {e}")
        return None


def load_data(trade_book_ids: list, strategies_config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess trade book and market data from multiple trade book IDs.
    
    Args:
        trade_book_ids: List of trade book IDs to load
        strategies_config: Dictionary of strategy configurations
        
    Returns:
        Tuple of (trade_book_df, market_data_df) combining all selected trade books
    """
    display = Display()
    all_tb = []
    all_md = []
    
    if not strategies_config:
        st.error("No strategies found in config file")
        return pd.DataFrame(), pd.DataFrame()
    
    if not trade_book_ids:
        st.warning("No trade book IDs selected")
        return pd.DataFrame(), pd.DataFrame()
    
    first_strategy = list(strategies_config.keys())[0]
    first_config = strategies_config[first_strategy]
    
    # Load data for each trade_book_id
    for trade_book_id in trade_book_ids:
        tb, md = display.load_and_preprocess(trade_book_id, first_config)
        
        if not tb.empty:
            all_tb.append(tb)
            if not md.empty:
                all_md.append(md)
        else:
            st.warning(f"⚠️ No trade book data found for ID: {trade_book_id}")
    
    # Combine all trade books
    if all_tb:
        combined_tb = pd.concat(all_tb, ignore_index=True)
        # Sort by Close_time if available
        if "Close_time" in combined_tb.columns:
            combined_tb = combined_tb.sort_values("Close_time").reset_index(drop=True)
    else:
        combined_tb = pd.DataFrame()
        st.error("❌ **No trade book data found for any selected IDs**")
        st.warning("Please check:")
        st.markdown("""
        - The IDs exist in MongoDB
        - The documents have a 'data' field with trade records
        - MongoDB connection is working correctly
        """)
    
    # Combine all market data
    if all_md:
        combined_md = pd.concat(all_md, ignore_index=True)
        # Remove duplicates based on timestamp
        if "timestamp" in combined_md.columns:
            combined_md = combined_md.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    else:
        combined_md = pd.DataFrame()

    return combined_tb, combined_md


def main():
    """Main application function."""
    base_dir = "/Users/gregoirebrendel/Documents/Code/quantdev/PROD/Generic utils/Backtest/utils"
    strategies_config_path = os.path.join(base_dir, "configs", "strategies.json")

    with open(strategies_config_path, "r") as file:
        strategies_config = json.load(file)

    # Get all available trade book IDs from database
    available_trade_books = get_all_trade_book_ids()
    
    if not available_trade_books:
        st.error("❌ **No trade books found in database**")
        st.stop()
        return
    
    # Initialize session state for selected trade_book_ids if not exists
    if 'selected_trade_book_ids' not in st.session_state:
        # Default to the latest trade book ID
        latest_id = get_latest_trade_book_id()
        st.session_state['selected_trade_book_ids'] = [latest_id] if latest_id else [available_trade_books[0]["trade_book_id"]]
    
    # Render sidebar with trade book ID selection first
    with st.sidebar:
        _render_cache_button()
        selected_trade_book_ids = _render_trade_book_selection(available_trade_books)
        
        # Update session state with selected IDs
        st.session_state['selected_trade_book_ids'] = selected_trade_book_ids

    # Load trade book and market data for selected IDs
    tb, md = load_data(selected_trade_book_ids, strategies_config)
    
    # Store data in session state for pages to access / Initialize state
    st.session_state['trade_book'] = tb
    st.session_state['market_data'] = md
    
    # Render sidebar filters and metrics (if data is available)
    if not tb.empty:
        _render_sidebar(tb, md)
    
    # Render main content
    _render_main_content()


def _render_sidebar(tb: pd.DataFrame, md: pd.DataFrame) -> None:
    """Render the sidebar with filters and quick stats."""
    with st.sidebar:
        selected_range = _render_filters(tb)
        _apply_filters(tb, md, selected_range)
        _render_quick_stats(tb)


def _render_cache_button() -> None:
    """Render the cache refresh button."""
    pass


def _render_trade_book_selection(available_trade_books: list) -> list:
    """
    Render trade book ID selection widget in sidebar.
    
    Args:
        available_trade_books: List of dictionaries with trade book metadata
        
    Returns:
        List of selected trade_book_ids
    """
    
    # Create display labels with metadata
    options = []
    trade_book_map = {}
    
    for tb_info in available_trade_books:
        tb_id = tb_info["trade_book_id"]
        symbol = tb_info.get("symbol", "N/A")
        created_at = tb_info.get("created_at")
        nb_trades = tb_info.get("nb_trades", 0)
        
        # Format created_at if available
        if created_at:
            if isinstance(created_at, str):
                date_str = created_at.split()[0] if ' ' in created_at else created_at
            else:
                date_str = str(created_at).split()[0] if ' ' in str(created_at) else str(created_at)
        else:
            date_str = "N/A"
        
        # Create display label
        label = f"{tb_id} | {symbol} | {nb_trades} trades | {date_str}"
        options.append(label)
        trade_book_map[label] = tb_id
    
    # Get default selection from session state or use latest
    selected_ids = st.session_state.get('selected_trade_book_ids', [])
    
    # Map selected IDs to labels
    default_labels = []
    for option in options:
        if trade_book_map[option] in selected_ids:
            default_labels.append(option)
    
    # If no defaults, select the first one (latest)
    if not default_labels and options:
        default_labels = [options[0]]
    
    # Multi-select widget with key for state management
    selected_labels = st.multiselect(
        "Choose trade book IDs to analyze",
        options=options,
        default=default_labels,
        key="trade_book_selection",
        help="Select one or more trade books to combine and analyze"
    )
    
    # Convert labels back to trade_book_ids
    selected_trade_book_ids = [trade_book_map[label] for label in selected_labels] if selected_labels else []
    
    # Ensure at least one is selected
    if not selected_trade_book_ids and options:
        selected_trade_book_ids = [trade_book_map[options[0]]]
    
    return selected_trade_book_ids


def _render_filters(tb: pd.DataFrame) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Render date filters."""
    
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
    
    return selected_range


def _apply_filters(tb: pd.DataFrame, md: pd.DataFrame, 
                  selected_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]]) -> None:
    """Apply filters to data and update session state."""
    # Store filters in session state
    st.session_state['date_range'] = selected_range
    
    # Apply filters
    filtered_tb = tb.copy()
    filtered_md = md.copy()
    
    if selected_range is not None:
        start, end = pd.to_datetime(selected_range[0]), pd.to_datetime(selected_range[1])
        filtered_tb = filtered_tb[(filtered_tb["Close_time"] >= start) & (filtered_tb["Close_time"] <= end)].copy()
        if not filtered_md.empty and "timestamp" in filtered_md.columns:
            mask = (filtered_md["timestamp"] >= start) & (filtered_md["timestamp"] <= end)
            filtered_md = filtered_md[mask].copy()

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