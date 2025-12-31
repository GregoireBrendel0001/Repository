import streamlit as st
import pandas as pd
from display_class import Display

def render_global_performance():
    """Render the Global Performance page with Apple-style design."""
    
    # Load data from session state
    filtered_tb = st.session_state.get('trade_book', pd.DataFrame())
    filtered_md = st.session_state.get('market_data', pd.DataFrame())
    
    if filtered_tb.empty:
        st.error("No trading data available. Please check your data files.")
        return
    
    
    # Calculate strategy vs asset performance and volatility
    strategy_return, asset_return = Display.calculate_strategy_vs_asset_performance(filtered_tb, filtered_md)
    strategy_volatility, asset_volatility = Display.calculate_strategy_vs_asset_volatility(filtered_tb, filtered_md)
    
    # Key metrics with Apple-style cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Use gross PnL for total PnL calculation
    total_pnl = filtered_tb["gross_pnl"].sum() if "gross_pnl" in filtered_tb.columns else filtered_tb["pnl"].sum()
    win_rate = filtered_tb["is_win"].mean() * 100
    avg_duration = filtered_tb["duration_min"].mean() / 60  # Convert minutes to hours
    total_trades = len(filtered_tb)
    liquidity_cost_buy = filtered_tb["Liquidity cost buy"].sum()
    total_fees = filtered_tb["Fees"].sum()
    
    # Apple-style metric cards
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #34c759 0%, #30d158 100%);
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(52,199,89,0.2);
            border: 1px solid rgba(52,199,89,0.1);">
            <h3 style="
                color: white;
                margin: 0;
                font-size: 1.8rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                ${total_pnl:,.0f}
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Total PnL
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #007aff 0%, #5856d6 100%);
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0,122,255,0.2);
            border: 1px solid rgba(0,122,255,0.1);">
            <h3 style="
                color: white;
                margin: 0;
                font-size: 1.8rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                {win_rate:.1f}%
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Win Rate
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #5856d6 0%, #af52de 100%);
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(88,86,214,0.2);
            border: 1px solid rgba(88,86,214,0.1);">
            <h3 style="
                color: white;
                margin: 0;
                font-size: 1.8rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                {avg_duration:.2f}h
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Avg Duration
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #ff3b30 0%, #ff453a 100%);
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(255,59,48,0.2);
            border: 1px solid rgba(255,59,48,0.1);">
            <h3 style="
                color: white;
                margin: 0;
                font-size: 1.8rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                {total_trades}
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Total Trades
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #ff9500 0%, #ff9f0a 100%);
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(255,149,0,0.2);
            border: 1px solid rgba(255,149,0,0.1);">
            <h3 style="
                color: white;
                margin: 0;
                font-size: 1.8rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                {liquidity_cost_buy:.2f}
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                TCA
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(255,107,107,0.2);
            border: 1px solid rgba(255,107,107,0.1);">
            <h3 style="
                color: white;
                margin: 0;
                font-size: 1.8rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                ${total_fees:,.0f}
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Total Fees
            </p>
        </div>
        """, unsafe_allow_html=True)

    
    # Equity curve with drawdown
    st.markdown("""
    <h3 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Equity Curve with Drawdown
    </h3>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    
    fig = Display.equity_curve_with_backtest(filtered_tb, filtered_md, None)
    fig.update_layout(
        height=600,  # Increased height for two-panel layout
        plot_bgcolor='rgb(38, 39, 47)',
        paper_bgcolor='rgb(38, 39, 47)',
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color='#ffffff',
            size=11
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        # Apply rounded corners using CSS
        uirevision=True
    )
    
    # Apply rounded corners to the plot using custom CSS
    st.markdown("""
    <style>
    .js-plotly-plot .plotly .main-svg {
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    .js-plotly-plot .plotly .bg {
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Strategy vs Asset Performance - 4 separate boxes
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(255,107,107,0.2);
            border: 1px solid rgba(255,107,107,0.1);">
            <h3 style="
                color: white;
                margin: 0;
                font-size: 1.8rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                {strategy_return:.1f}%
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Strategy Return
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(78,205,196,0.2);
            border: 1px solid rgba(78,205,196,0.1);">
            <h3 style="
                color: white;
                margin: 0;
                font-size: 1.8rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                {asset_return:.1f}%
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Asset Return
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #FF9500 0%, #FF9F0A 100%);
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(255,149,0,0.2);
            border: 1px solid rgba(255,149,0,0.1);">
            <h3 style="
                color: white;
                margin: 0;
                font-size: 1.8rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                {strategy_volatility:.1f}%
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Strategy Drawdown
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #5856d6 0%, #af52de 100%);
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(88,86,214,0.2);
            border: 1px solid rgba(88,86,214,0.1);">
            <h3 style="
                color: white;
                margin: 0;
                font-size: 1.8rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                {asset_volatility:.1f}%
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Asset Volatility
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Trade Size vs PnL plot in a column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <h3 style="
            color: #ffffff;
            margin: 0 0 0.2rem 0;
            font-size: 1.3rem;
            font-weight: 600;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            Trade Size vs PnL
        </h3>
        <hr style="
            border: none;
            height: 1px;
            background-color: #e2e8f0;
            margin: 0 0 0.5rem 0;">
        """, unsafe_allow_html=True)
        
        fig = Display.trade_size_vs_pnl(filtered_tb)
        fig.update_layout(
            height=300,
            plot_bgcolor='rgb(38, 39, 47)',
            paper_bgcolor='rgb(38, 39, 47)',
            font=dict(
                family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                color='#ffffff',
                size=11
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#ffffff'),
                titlefont=dict(color='#ffffff')
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#ffffff'),
                titlefont=dict(color='#ffffff')
            ),
            # Apply rounded corners using CSS
            uirevision=True
        )
        
        # Apply rounded corners to the plot using custom CSS
        st.markdown("""
        <style>
        .js-plotly-plot .plotly .main-svg {
            border-radius: 12px !important;
            overflow: hidden !important;
        }
        .js-plotly-plot .plotly .bg {
            border-radius: 12px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown("""
        <h3 style="
            color: #ffffff;
            margin: 0 0 0.5rem 0;
            font-size: 1.3rem;
            font-weight: 600;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            PnL Distribution
        </h3>
        <hr style="
            border: none;
            height: 1px;
            background-color: #e2e8f0;
            margin: 0 0 1rem 0;">
        """, unsafe_allow_html=True)
        
        fig = Display.pnl_histogram(filtered_tb)
        fig.update_layout(
            height=300,
            plot_bgcolor='rgb(38, 39, 47)',
            paper_bgcolor='rgb(38, 39, 47)',
            font=dict(
                family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                color='#ffffff',
                size=11
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#ffffff'),
                titlefont=dict(color='#ffffff')
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#ffffff'),
                titlefont=dict(color='#ffffff')
            ),
            # Apply rounded corners using CSS
            uirevision=True
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Trade Frequency by Time - Full width below
    st.markdown("""
    <h3 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Trade Frequency by Time
    </h3>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    # Create the trade frequency plot showing individual trades
    fig = Display.trade_frequency_by_time(filtered_tb)
    
    fig.update_layout(
        height=400,  # Increased height for full width
        plot_bgcolor='rgb(38, 39, 47)',
        paper_bgcolor='rgb(38, 39, 47)',
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color='#ffffff',
            size=11
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff'),
            titlefont=dict(color='#ffffff')
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff'),
            titlefont=dict(color='#ffffff')
        ),
        # Apply rounded corners using CSS
        uirevision=True
    )
    
    # Render the plot
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # PnL Timeline with Markers (below)
    st.markdown("""
    <h3 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        PnL Timeline with Markers
    </h3>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    fig = Display.pnl_timeline_with_markers(filtered_tb)
    fig.update_layout(
        height=350,
        plot_bgcolor='rgb(38, 39, 47)',
        paper_bgcolor='rgb(38, 39, 47)',
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color='#ffffff',
            size=11
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff'),
            titlefont=dict(color='#ffffff')
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff'),
            titlefont=dict(color='#ffffff')
        )
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Expectation vs Reality (if available)
    if "Confidence" in filtered_tb.columns or "Score" in filtered_tb.columns or "Signal" in filtered_tb.columns:
        st.markdown("""
        <h3 style="
            color: #ffffff;
            margin: 0 0 0.5rem 0;
            font-size: 1.3rem;
            font-weight: 600;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            Expectation vs Reality
        </h3>
        <hr style="
            border: none;
            height: 1px;
            background-color: #e2e8f0;
            margin: 0 0 1rem 0;">
        """, unsafe_allow_html=True)
        
        fig = Display.expectation_vs_reality(filtered_tb)
        if fig is not None:
            fig.update_layout(
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(
                    family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                    color='#1d1d1f',
                    size=11
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(gridcolor='rgba(0,0,0,0.05)'),
                yaxis=dict(gridcolor='rgba(0,0,0,0.05)')
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

if __name__ == "__main__":
    render_global_performance()
