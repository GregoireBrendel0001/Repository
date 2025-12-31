import streamlit as st
import pandas as pd
from display_class import Display

def render_strategy_optimization():
    """Render the Strategy Optimization page with Apple-style design."""
    
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
    

    
    col1, col2, col3 = st.columns(3)
    
    # Calculate Kelly metrics using Display class
    kelly_result = Display.kelly_estimator(tb)
    p, b, kelly_fraction = kelly_result
    
    # Calculate win rate and average win/loss from pnl data
    wins = tb[tb["pnl"] > 0]
    losses = tb[tb["pnl"] <= 0]
    win_rate = len(wins) / len(tb) if len(tb) > 0 else 0
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["pnl"].mean()) if len(losses) > 0 else 1
    
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
                {kelly_fraction:.2f}
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Kelly Fraction
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
                {avg_win:.0f}
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Avg Win
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
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
                {avg_loss:.0f}
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Avg Loss
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Kelly Growth Rate Analysis
    st.markdown("""
    <h2 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Kelly Criterion Growth Rate
    </h2>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    # Kelly growth rate plot
    fig_kelly_growth = Display.kelly_growth_rate_plot(tb)
    fig_kelly_growth.update_layout(
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
    st.plotly_chart(fig_kelly_growth, use_container_width=True, config={'displayModeBar': False})
    

    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Monte Carlo Simulation
    st.markdown("""
    <h2 style="
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        Monte Carlo Simulation
    </h2>
    <hr style="
        border: none;
        height: 1px;
        background-color: #e2e8f0;
        margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)
    

    

    

    
    fig, bankruptcy_percentage = Display.monte_carlo_equity(tb)
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
    
    # Bankruptcy risk metric box
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
                {bankruptcy_percentage:.1f}%
            </h4>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 0;
                font-size: 0.8rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Bankruptcy Risk
            </p>
            <p style="
                color: rgba(255,255,255,0.7);
                margin: 0.2rem 0 0 0;
                font-size: 0.7rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                Based on {1000} Monte Carlo simulations
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_strategy_optimization()
