import streamlit as st
import pandas as pd
from display_class import Display

def render_statistical_insight():
    """Render the Statistical Insight page with Apple-style design."""
    
    # Load data from session state
    tb = st.session_state.get('trade_book', pd.DataFrame())
    md = st.session_state.get('market_data', pd.DataFrame())
    date_range = st.session_state.get('date_range', None)
    
    if tb.empty:
        st.error("No trading data available. Please check your data files.")
        return
    
    # Apply rounded corners to all plots using custom CSS
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
    
    # Statistical significance cards
    col1, col2, col3 = st.columns(3)
    
    # Calculate statistical metrics
    mean_pnl = tb["pnl"].mean()
    std_pnl = tb["pnl"].std()
    sharpe_ratio = mean_pnl / std_pnl if std_pnl > 0 else 0
    win_rate = tb["is_win"].mean() * 100
    
    # Calculate t-test statistics
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(tb["pnl"], 0)
    
    # Format p-value for display
    p_value_formatted = f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}"
    
    with col1:
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
                {sharpe_ratio:.2f}
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Sharpe Ratio
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
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
                {t_stat:.2f}
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                T-Statistic
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
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
                {p_value_formatted}
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                P-Value
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Market conditions analysis
    st.markdown("""
    <h2 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Market Conditions Analysis
    </h2>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <h3 style="
            color: #ffffff;
            margin: 0 0 0.5rem 0;
            font-size: 1.3rem;
            font-weight: 600;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            Profit vs Volatility
        </h3>
        <hr style="
            border: none;
            height: 1px;
            background-color: #e2e8f0;
            margin: 0 0 1rem 0;">
        """, unsafe_allow_html=True)
        
        fig = Display.profit_vs_volatility(tb)
        fig.update_layout(
            height=600,
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
                tickfont=dict(color='#ffffff', size=11),
                titlefont=dict(color='#ffffff', size=11)
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#ffffff', size=11),
                titlefont=dict(color='#ffffff', size=11)
            ),
            uirevision=True
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown("""
        <h3 style="
            color: #ffffff;
            margin: 0 0 0.5rem 0;
            font-size: 1.3rem;
            font-weight: 600;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            Profit vs Market Trend
        </h3>
        <hr style="
            border: none;
            height: 1px;
            background-color: #e2e8f0;
            margin: 0 0 1rem 0;">
        """, unsafe_allow_html=True)
        
        fig = Display.profit_vs_trend(tb)
        fig.update_layout(
            height=600,
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
                tickfont=dict(color='#ffffff', size=11),
                titlefont=dict(color='#ffffff', size=11)
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#ffffff', size=11),
                titlefont=dict(color='#ffffff', size=11)
            ),
            uirevision=True
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Trade efficiency analysis
    st.markdown("""
    <h2 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Trade Efficiency Analysis
    </h2>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    # Profit velocity
    st.markdown("""
    <h3 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Profit Velocity
    </h3>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    fig, mean_val, median_val = Display.profit_velocity(tb)
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
            tickfont=dict(color='#ffffff', size=11),
            titlefont=dict(color='#ffffff', size=11)
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff', size=11),
            titlefont=dict(color='#ffffff', size=11)
        ),
        uirevision=True
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Add metric boxes for mean and median
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #FFB019 0%, #FF9500 100%);
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(255,176,25,0.2);
            border: 1px solid rgba(255,176,25,0.1);
            margin: 1rem 0;">
            <h4 style="
                color: white;
                margin: 0;
                font-size: 1.2rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                {mean_val:.4f}%
            </h4>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.8rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Mean Return/Min
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #00E396 0%, #00D4AA 100%);
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0,227,150,0.2);
            border: 1px solid rgba(0,227,150,0.1);
            margin: 1rem 0;">
            <h4 style="
                color: white;
                margin: 0;
                font-size: 1.2rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                {median_val:.4f}%
            </h4>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.8rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Median Return/Min
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Edge ratio analysis
    st.markdown("""
    <h3 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Edge Ratio Analysis
    </h3>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    fig = Display.edge_ratio_plot(tb)
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
            tickfont=dict(color='#ffffff', size=11),
            titlefont=dict(color='#ffffff', size=11)
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff', size=11),
            titlefont=dict(color='#ffffff', size=11)
        ),
        uirevision=True
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Statistical significance analysis
    st.markdown("""
    <h2 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Statistical Significance
    </h2>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Autocorrelation
    </h3>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    fig = Display.autocorr_plot(tb)
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
            tickfont=dict(color='#ffffff', size=11),
            titlefont=dict(color='#ffffff', size=11)
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff', size=11),
            titlefont=dict(color='#ffffff', size=11)
        ),
        uirevision=True
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Rolling metrics
    st.markdown("""
    <h3 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Rolling Sharpe & Sortino Ratios
    </h3>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    fig = Display.rolling_sharpe_sortino(tb)
    fig.update_layout(
        height=400,
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
            tickfont=dict(color='#ffffff', size=11),
            titlefont=dict(color='#ffffff', size=11)
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff', size=11),
            titlefont=dict(color='#ffffff', size=11)
        ),
        uirevision=True
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Regime split equity curve
    st.markdown("""
    <h3 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Regime-Split Equity Curve
    </h3>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    fig, high_vol_performance, stable_performance, vol_threshold = Display.regime_split_equity(tb)
    fig.update_layout(
        height=400,
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
            tickfont=dict(color='#ffffff', size=11),
            titlefont=dict(color='#ffffff', size=11)
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff', size=11),
            titlefont=dict(color='#ffffff', size=11)
        ),
        uirevision=True
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Add metric boxes for volatility regime performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #FF4560 0%, #FF6B6B 100%);
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(255,69,96,0.2);
            border: 1px solid rgba(255,69,96,0.1);
            margin: 1rem 0;">
            <h4 style="
                color: white;
                margin: 0;
                font-size: 1.2rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                {high_vol_performance:.2f} USDT
            </h4>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.8rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                High Volatility Regime
            </p>
            <p style="
                color: rgba(255,255,255,0.7);
                margin: 0.2rem 0 0 0;
                font-size: 0.7rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                ≥ {vol_threshold:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #00E396 0%, #00D4AA 100%);
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0,227,150,0.2);
            border: 1px solid rgba(0,227,150,0.1);
            margin: 1rem 0;">
            <h4 style="
                color: white;
                margin: 0;
                font-size: 1.2rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                {stable_performance:.2f} USDT
            </h4>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.8rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Stable Regime
            </p>
            <p style="
                color: rgba(255,255,255,0.7);
                margin: 0.2rem 0 0 0;
                font-size: 0.7rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                < {vol_threshold:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Trade counts by volatility regime
    st.markdown("<br>", unsafe_allow_html=True)
    positive_trades_by_vol, negative_trades_by_vol, vol_threshold = Display.get_trade_counts_by_volatility_regime(tb)
    
    # Format the data for display
    def format_trade_counts(df, vol_threshold_val):
        """Format trade counts and average performance DataFrame into a readable string"""
        if df.empty:
            return "No data"
        
        lines = []
        for _, row in df.iterrows():
            regime = row['regime']
            count = int(row['count'])
            avg_pnl = row['avg_pnl']
            
            # Shorten regime names for display
            if regime == 'High Volatility Regime':
                regime_short = 'High Vol'
                threshold_text = f"≥ {vol_threshold_val:.4f}"
            else:
                regime_short = 'Stable'
                threshold_text = f"< {vol_threshold_val:.4f}"
            
            # Format display based on whether there are trades
            if count == 0:
                lines.append(f"<strong>{regime_short}</strong> ({threshold_text}): {count} trades")
            else:
                lines.append(f"<strong>{regime_short}</strong> ({threshold_text}): {count} trades | Avg: {avg_pnl:.2f} USDT")
        
        return "<br>".join(lines)
    
    positive_counts_str = format_trade_counts(positive_trades_by_vol, vol_threshold)
    negative_counts_str = format_trade_counts(negative_trades_by_vol, vol_threshold)
    total_positive = positive_trades_by_vol['count'].sum() if not positive_trades_by_vol.empty else 0
    total_negative = negative_trades_by_vol['count'].sum() if not negative_trades_by_vol.empty else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #00E396 0%, #00D4AA 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,227,150,0.2);
            border: 1px solid rgba(0,227,150,0.1);
            margin: 1rem 0;">
            <h4 style="
                color: white;
                margin: 0 0 1rem 0;
                font-size: 1.3rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                text-align: center;">
                Positive Trades
            </h4>
            <h3 style="
                color: white;
                margin: 0 0 1rem 0;
                font-size: 2rem;
                font-weight: 700;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                text-align: center;">
                {total_positive}
            </h3>
            <div style="
                color: rgba(255,255,255,0.95);
                font-size: 0.85rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                background: rgba(0,0,0,0.2);
                padding: 1rem;
                border-radius: 8px;
                margin-top: 0.5rem;">
                {positive_counts_str}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #FF4560 0%, #FF6B6B 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(255,69,96,0.2);
            border: 1px solid rgba(255,69,96,0.1);
            margin: 1rem 0;">
            <h4 style="
                color: white;
                margin: 0 0 1rem 0;
                font-size: 1.3rem;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                text-align: center;">
                Negative Trades
            </h4>
            <h3 style="
                color: white;
                margin: 0 0 1rem 0;
                font-size: 2rem;
                font-weight: 700;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                text-align: center;">
                {total_negative}
            </h3>
            <div style="
                color: rgba(255,255,255,0.95);
                font-size: 0.85rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                background: rgba(0,0,0,0.2);
                padding: 1rem;
                border-radius: 8px;
                margin-top: 0.5rem;">
                {negative_counts_str}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Rolling Win Rate
    st.markdown("""
    <h3 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Rolling Win Rate
    </h3>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    fig = Display.rolling_win_rate(tb)
    fig.update_layout(
        height=400,
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
            tickfont=dict(color='#ffffff', size=11),
            titlefont=dict(color='#ffffff', size=11)
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff', size=11),
            titlefont=dict(color='#ffffff', size=11)
        ),
        uirevision=True
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

if __name__ == "__main__":
    render_statistical_insight()
