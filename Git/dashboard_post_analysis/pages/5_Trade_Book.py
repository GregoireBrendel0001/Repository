import streamlit as st
import pandas as pd
from display_class import Display

def render_trade_book():

    
    # Get data from session state
    if 'trade_book' not in st.session_state:
        st.error("No trade book data available. Please load data first.")
        return
    
    tb = st.session_state['trade_book']
    
    if tb.empty:
        st.warning("Trade book is empty.")
        return
    
    # Create a copy for display - keep all original columns
    display_df = tb.copy()
    

    
    # Time range filter
    if 'Open_time' in tb.columns:
        # Convert to datetime for date picker
        tb_dates = pd.to_datetime(tb['Open_time'])
        min_date = tb_dates.min().date()
        max_date = tb_dates.max().date()
        
        date_range = st.date_input(
            "Time Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        date_range = None

    # Apply time range filter only
    filtered_df = display_df.copy()
    
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date and end_date:
            # Convert display_df dates back to datetime for comparison
            display_df_dates = pd.to_datetime(display_df['Open_time'])
            mask = (display_df_dates.dt.date >= start_date) & (display_df_dates.dt.date <= end_date)
            filtered_df = filtered_df[mask]
    

    

    
    # Configure dataframe display
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=600,
        hide_index=True
    )
    


if __name__ == "__main__":
    render_trade_book()
