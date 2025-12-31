import streamlit as st
import pandas as pd
from display_class import Display

def render_survivability_risk():
    """Render the Survivability & Risk page with Apple-style design."""
    
    # Load data from session state
    tb = st.session_state.get('trade_book', pd.DataFrame())
    
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
    
    # Risk metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate risk metrics
    total_pnl = tb["pnl"].sum()
    max_drawdown = tb["drawdown"].max() if "drawdown" in tb.columns else 0  # Changed from .min() to .max()
    
    # Calculate performance volatility as percentage of average trade size
    avg_trade_size = tb["size_usdt"].mean() if "size_usdt" in tb.columns and len(tb) > 0 else 1
    volatility_pct = (tb["pnl"].std() / avg_trade_size * 100) if avg_trade_size > 0 else 0
    
    var_95 = tb["pnl"].quantile(0.05) if len(tb) > 0 else 0
    
    # Risk index calculation (0-100 scale)
    risk_index = min(100, max(0, (abs(max_drawdown) / abs(total_pnl) * 100) if total_pnl != 0 else 50))
    
    with col1:
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
                {risk_index:.0f}
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Risk Index
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
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
                {max_drawdown:.0f}
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Max Drawdown
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
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
                {volatility_pct:.1f}%
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Performance Volatility
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
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
                {var_95:.0f}
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                VaR (95%)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Drawdown analysis
    st.markdown("""
    <h2 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Drawdown Analysis
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
        Max Drawdown Over Time
    </h3>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    fig = Display.max_drawdown_over_time(tb)
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
        ),
        uirevision=True
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Drawdown recovery time distribution
    st.markdown("""
    <h3 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Drawdown Recovery Time Distribution
    </h3>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    fig, recovery_insights = Display.drawdown_recovery_distribution(tb)
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
        ),
        uirevision=True
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Add recovery time insights if available
    if recovery_insights:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
                padding: 1rem;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 4px 16px rgba(0,122,255,0.2);
                border: 1px solid rgba(0,122,255,0.1);
                margin: 1rem 0;">
                <h4 style="
                    color: white;
                    margin: 0;
                    font-size: 1.2rem;
                    font-weight: 600;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                    {recovery_insights['p75_recovery_hours']/24:.1f}d
                </h4>
                <p style="
                    color: rgba(255,255,255,0.9);
                    margin: 0;
                    font-size: 0.8rem;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                    75th Percentile
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #34C759 0%, #30D158 100%);
                padding: 1rem;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 4px 16px rgba(52,199,89,0.2);
                border: 1px solid rgba(52,199,89,0.1);
                margin: 1rem 0;">
                <h4 style="
                    color: white;
                    margin: 0;
                    font-size: 1.2rem;
                    font-weight: 600;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                    {recovery_insights['max_recovery_hours']/24:.1f}d
                </h4>
                <p style="
                    color: rgba(255,255,255,0.9);
                    margin: 0;
                    font-size: 0.8rem;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                    Max
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #FF9500 0%, #FF9F0A 100%);
                padding: 1rem;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 4px 16px rgba(255,149,0,0.2);
                border: 1px solid rgba(255,149,0,0.1);
                margin: 1rem 0;">
                <h4 style="
                    color: white;
                    margin: 0;
                    font-size: 1.2rem;
                    font-weight: 600;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                    {recovery_insights['total_recoveries']}
                </h4>
                <p style="
                    color: rgba(255,255,255,0.9);
                    margin: 0;
                    font-size: 0.8rem;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                    Number of Recovery
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Risk distribution analysis
    st.markdown("""
    <h2 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Risk Distribution Analysis
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
            VaR Distribution
        </h3>
        <hr style="
            border: none;
            height: 1px;
            background-color: #e2e8f0;
            margin: 0 0 1rem 0;">
        """, unsafe_allow_html=True)
        
        fig = Display.var_plot(tb)
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
            Trade Streaks
        </h3>
        <hr style="
            border: none;
            height: 1px;
            background-color: #e2e8f0;
            margin: 0 0 1rem 0;">
        """, unsafe_allow_html=True)
        
        fig = Display.trade_streaks(tb)
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
            ),
            uirevision=True
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    

    


if __name__ == "__main__":
    render_survivability_risk()
