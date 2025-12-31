"""
Display class for Trading Strategy Dashboard

This module contains all plotting and analysis functions for the dashboard,
including data preprocessing, feature engineering, and visualization components.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Constants
DEFAULT_PLOT_HEIGHT = 600
DEFAULT_PLOT_WIDTH = 800
BUCKET_COUNT = 6
CONFIDENCE_LEVEL = 0.95
DEFAULT_ALPHA = 0.05

# Color palettes
COLORS = {
    'primary': '#2563eb',
    'success': '#10b981',
    'warning': '#f59e0b',
    'error': '#ef4444',
    'info': '#06b6d4',
    'neutral': '#6b7280'
}

VOLATILITY_COLORS = {
    'Very Low': '#00E396',
    'Low': '#FFB019', 
    'Medium': '#FF6B6B',
    'High': '#FF4560',
    'Very High': '#FF1744',
    'Extreme': '#D50000'
}

TREND_COLORS = {
    'Very Low': '#00E396',
    'Low': '#FFB019',
    'Medium': '#FF6B6B', 
    'High': '#FF4560',
    'Very High': '#FF1744',
    'Extreme': '#D50000',
    'Neutral': '#8E8E93'
}


class Display:
    """Encapsulates all plotting/analysis functions for the dashboard."""

    # --------- Preprocessing and feature engineering ---------
    @staticmethod
    def preprocess(trade_book: pd.DataFrame, market_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tb = trade_book.copy()
        md = market_data.copy()

        # Parse timestamps for trade book - handle different formats
        for col in ["Open_time", "Close_time"]:
            if col in tb.columns:
                # Check if timestamp is numeric (milliseconds)
                if tb[col].dtype in ['int64', 'float64']:
                    tb[col] = pd.to_datetime(tb[col], unit='ms')
                elif tb[col].dtype == 'object':
                    # Try different formats
                    try:
                        # First try the expected format
                        tb[col] = pd.to_datetime(tb[col], format="%d/%m/%Y %H:%M", errors="coerce")
                    except:
                        try:
                            # If that fails, try standard conversion
                            tb[col] = pd.to_datetime(tb[col], errors="coerce")
                        except:
                            # If that fails, try with milliseconds
                            tb[col] = pd.to_datetime(tb[col], unit='ms', errors="coerce")
        
        # Parse timestamps for market data - handle different formats
        if not md.empty and "timestamp" in md.columns:
            # Check if timestamp is numeric (milliseconds)
            if md["timestamp"].dtype in ['int64', 'float64']:
                md["timestamp"] = pd.to_datetime(md["timestamp"], unit='ms')
            elif md["timestamp"].dtype == 'object':
                # Try different formats
                try:
                    md["timestamp"] = pd.to_datetime(md["timestamp"])
                except:
                    # If that fails, try with milliseconds
                    md["timestamp"] = pd.to_datetime(md["timestamp"], unit='ms')
            
            # Sort market data by timestamp
            md = md.sort_values("timestamp")

        # Handle Duration - convert from days to timedelta if it's a float
        if "Duration" in tb.columns:
            if tb["Duration"].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Duration is in days, convert to timedelta
                tb["Duration"] = pd.to_timedelta(tb["Duration"], unit='D')
            elif tb["Duration"].dtype == object:
                # Duration is a string, try to parse as timedelta
                tb["Duration"] = pd.to_timedelta(tb["Duration"], errors="coerce")

        # Basic trade features
        tb = tb.sort_values("Close_time")
        tb["pnl"] = tb.get("Profit & Loss")
        tb["size_usdt"] = tb.get("USDT value open")
        tb["is_win"] = (tb.get("Positive") == 1).astype(int) if "Positive" in tb.columns else (tb["pnl"] > 0).astype(int)
        tb["duration_sec"] = tb["Duration"].dt.total_seconds()
        tb["duration_min"] = tb["duration_sec"] / 60.0
        tb["return_frac"] = np.where(tb["size_usdt"] > 0, tb["pnl"] / tb["size_usdt"], np.nan)
        # Calculate percentage return per minute to remove size bias
        tb["profit_per_minute"] = np.where(
            (tb["duration_min"] > 0) & (tb["size_usdt"] > 0), 
            (tb["pnl"] / tb["size_usdt"]) / tb["duration_min"], 
            np.nan
        )
        tb["hour"] = tb["Open_time"].dt.hour
        tb["dow"] = tb["Open_time"].dt.day_name()
        tb["month"] = tb["Open_time"].dt.month_name()

        # Equity curve on trade-close timeline
        tb["cum_pnl"] = tb["pnl"].cumsum()
        
        # Calculate gross PnL (before fees) for each trade
        # First, calculate individual trade fees
        tb["trade_fee"] = 0.0
        if "Fee" in tb.columns:
            tb["trade_fee"] = tb["Fee"]
        elif "Commission" in tb.columns:
            tb["trade_fee"] = tb["Commission"]
        elif "Fees" in tb.columns:
            tb["trade_fee"] = tb["Fees"]
        else:
            # Estimate fees based on typical trading fees (0.1% per trade)
            estimated_fee_rate = 0.001  # 0.1% per trade
            tb["trade_fee"] = tb["size_usdt"] * estimated_fee_rate
        
        # Calculate gross PnL by subtracting the fee for each trade
        # gross_pnl = pnl - trade_fee (since pnl is net of fees, we subtract individual trade fees to get gross)
        tb["gross_pnl"] = tb["pnl"] - tb["trade_fee"]
        tb["cum_gross_pnl"] = tb["gross_pnl"].cumsum()
        
        # Calculate drawdown correctly as percentage of total account value (same as max_drawdown_over_time)
        initial_investment = tb["size_usdt"].iloc[0] if len(tb) > 0 else 0
        tb["total_value"] = tb["cum_pnl"] + initial_investment
        tb["running_max_value"] = tb["total_value"].cummax()
        
        # Calculate drawdown as percentage of running maximum total value
        tb["drawdown"] = np.where(
            (tb["running_max_value"] > 0) & (tb["running_max_value"] > tb["total_value"]),
            (tb["running_max_value"] - tb["total_value"]) / tb["running_max_value"] * 100,
            0
        )

        # Market features: returns, rolling volatility and trend proxies
        if not md.empty and "close" in md.columns:
            md["ret"] = md["close"].pct_change()
            bars_vol_window = 30  # number of bars for realized vol/trend
            md["roll_vol"] = md["ret"].rolling(bars_vol_window, min_periods=5).std()
            md["roll_trend"] = md["close"].pct_change(bars_vol_window)

            # Align market features at trade open
            aligned = pd.merge_asof(
                tb.sort_values("Open_time"),
                md[["timestamp", "roll_vol", "roll_trend"]].sort_values("timestamp"),
                left_on="Open_time",
                right_on="timestamp",
                direction="backward",
            )
            tb["entry_vol"] = aligned["roll_vol"].values
            tb["entry_trend"] = aligned["roll_trend"].values
        else:
            # If no market data, set default values
            tb["entry_vol"] = np.nan
            tb["entry_trend"] = np.nan

        # Rolling win rate and Kelly criterion calculations
        window = 30  # 30-trade rolling window
        
        # Rolling win rate (percentage)
        tb["rolling_win_rate"] = tb["is_win"].rolling(window, min_periods=5).mean() * 100
        
        # Rolling Kelly criterion calculations
        def calculate_rolling_kelly(series, window_size):
            kelly_values = []
            for i in range(len(series)):
                start_idx = max(0, i - window_size + 1)
                end_idx = i + 1
                window_data = series.iloc[start_idx:end_idx]
                
                if len(window_data) < 5:  # Need minimum data points
                    kelly_values.append(np.nan)
                    continue
                
                wins = window_data[window_data > 0]
                losses = window_data[window_data <= 0]
                
                if len(wins) == 0 or len(losses) == 0:
                    kelly_values.append(0.0)
                    continue
                
                p = len(wins) / len(window_data)  # Win probability
                avg_win = wins.mean()
                avg_loss = abs(losses.mean())
                
                if avg_loss == 0:
                    kelly_values.append(0.0)
                    continue
                
                b = avg_win / avg_loss  # Win/loss ratio
                kelly = p - (1 - p) / b if b > 0 else 0.0
                
                # Cap Kelly between 0 and 1 (0% to 100% allocation)
                kelly = max(0.0, min(1.0, kelly))
                kelly_values.append(kelly)
            
            return pd.Series(kelly_values, index=series.index)
        
        # Calculate rolling Kelly fraction
        tb["rolling_kelly_fraction"] = calculate_rolling_kelly(tb["pnl"], window)
        
        # Calculate suggested position size based on Kelly and current account value
        # Use a conservative approach: Kelly * 0.25 (quarter Kelly) for safety
        tb["kelly_position_size"] = np.where(
            tb["rolling_kelly_fraction"].notna() & (tb["rolling_kelly_fraction"] > 0),
            tb["total_value"] * tb["rolling_kelly_fraction"] * 0.25,  # Quarter Kelly
            tb["size_usdt"]  # Fallback to actual size if no Kelly signal
        )

        return tb, md

    @staticmethod
    def calculate_tracking_error(tb_real: pd.DataFrame, tb_backtest: pd.DataFrame) -> float:
        """Calculate tracking error between real-time and backtest PnL values."""
        if tb_real.empty or tb_backtest.empty:
            return 0.0
        
        # Ensure both dataframes have the same length and are aligned
        min_length = min(len(tb_real), len(tb_backtest))
        real_pnl = tb_real["pnl"].iloc[:min_length].values
        backtest_pnl = tb_backtest["pnl"].iloc[:min_length].values
        
        # Calculate tracking error as the standard deviation of the difference
        tracking_error = np.std(real_pnl - backtest_pnl)
        
        return tracking_error

    @staticmethod
    def calculate_strategy_vs_asset_performance(tb: pd.DataFrame, md: pd.DataFrame) -> Tuple[float, float]:
        """Calculate strategy performance vs asset performance."""
        if tb.empty or md.empty:
            return 0.0, 0.0
        
        # Strategy performance (cumulative PnL / initial investment)
        # Use gross PnL for strategy return calculation
        cumulative_pnl = tb['gross_pnl'].sum() if 'gross_pnl' in tb.columns else tb['pnl'].sum()
        initial_investment = tb['size_usdt'].iloc[0]  # First trade position size as initial investment
        strategy_return = (cumulative_pnl / initial_investment) * 100 if initial_investment > 0 else 0.0
        
        # Asset performance (buy and hold)
        # Align market data with trade period
        start_time = tb['Open_time'].min()
        end_time = tb['Close_time'].max()
        
        # Filter market data to trade period
        md_filtered = md[(md['timestamp'] >= start_time) & (md['timestamp'] <= end_time)]
        
        if len(md_filtered) < 2:
            return strategy_return, 0.0
        
        # Calculate asset return
        asset_start_price = md_filtered.iloc[0]['close']
        asset_end_price = md_filtered.iloc[-1]['close']
        asset_return = (asset_end_price - asset_start_price) / asset_start_price * 100
        
        return strategy_return, asset_return

    @staticmethod
    def calculate_strategy_vs_asset_volatility(tb: pd.DataFrame, md: pd.DataFrame) -> Tuple[float, float]:
        """Calculate strategy volatility vs asset volatility."""
        if tb.empty or md.empty:
            return 0.0, 0.0
        
        # Strategy volatility (max drawdown percentage)
        strategy_volatility = tb['drawdown'].max() / (tb['cum_pnl'].max() + tb['size_usdt'].iloc[0]) * 100
        
        # Asset volatility (over the entire market data period)
        # Calculate asset returns and volatility for the full period
        asset_returns = md['close'].pct_change().dropna()
        
        # Annualize the volatility (assuming 3-minute intervals for crypto data)
        # Convert to annual volatility: std * sqrt(number of periods per year)
        periods_per_year = 365 * 24 * 20  # 20 periods per hour (3-minute intervals)
        asset_volatility = np.std(asset_returns) * np.sqrt(periods_per_year) * 100
        

        
        return strategy_volatility, asset_volatility

    @staticmethod
    def equity_curve_with_backtest(tb_real: pd.DataFrame, tb_backtest: pd.DataFrame, md: pd.DataFrame, date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None) -> go.Figure:
        """Create equity curve chart with both real-time and backtest curves, plus market price."""
        tb_real_plot = tb_real.copy()
        tb_backtest_plot = tb_backtest.copy()
        
        # Get market data from session state if not provided
        if md.empty:
            md = st.session_state.get('market_data', pd.DataFrame())
        
        md_plot = md.copy()
        
        if date_range is not None:
            start, end = date_range
            tb_real_plot = tb_real_plot[(tb_real_plot["Close_time"] >= start) & (tb_real_plot["Close_time"] <= end)]
            tb_backtest_plot = tb_backtest_plot[(tb_backtest_plot["Close_time"] >= start) & (tb_backtest_plot["Close_time"] <= end)]
            md_plot = md_plot[(md_plot["timestamp"] >= start) & (md_plot["timestamp"] <= end)]

        # Create two vertically stacked subplots with shared x-axis and secondary y-axis for upper plot
        fig = make_subplots(
            rows=2, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],  # 2:1 ratio (upper plot larger)
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )

        # Upper subplot: Equity Curves (Real-time and Backtest) + Market Price
        fig.add_trace(
            go.Scatter(
                x=tb_real_plot["Close_time"], 
                y=tb_real_plot["cum_pnl"], 
                name="Real-time", 
                mode="lines", 
                line=dict(color="#8A2BE2", width=1),
                hovertemplate="<b>Date:</b> %{x}<br><b>Real-time PnL:</b> %{y:.2f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=tb_backtest_plot["Close_time"], 
                y=tb_backtest_plot["cum_pnl"], 
                name="Backtest", 
                mode="lines", 
                line=dict(color="#00E396", width=1),
                hovertemplate="<b>Date:</b> %{x}<br><b>Backtest PnL:</b> %{y:.2f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Add cumulative gross PnL curve (real-time data)
        if 'cum_gross_pnl' in tb_real_plot.columns:
            fig.add_trace(
                go.Scatter(
                x=tb_real_plot["Close_time"], 
                    y=tb_real_plot["cum_gross_pnl"], 
                    name="Real-time (Gross)", 
                    mode="lines", 
                    line=dict(color="#FF6B6B", width=1, dash="dot"),
                    hovertemplate="<b>Date:</b> %{x}<br><b>Gross PnL:</b> %{y:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        else:
            # Fallback: calculate gross PnL inline if preprocessing didn't create the column
            print("cum_gross_pnl column not found, calculating inline...")
            total_fees = 0.0
            if "Fee" in tb_real_plot.columns:
                total_fees = tb_real_plot["Fee"].sum()
            elif "Commission" in tb_real_plot.columns:
                total_fees = tb_real_plot["Commission"].sum()
            elif "Fees" in tb_real_plot.columns:
                total_fees = tb_real_plot["Fees"].sum()
            else:
                # Estimate fees based on typical trading fees (0.1% per trade)
                total_volume = tb_real_plot["size_usdt"].sum() if "size_usdt" in tb_real_plot.columns else 0
                estimated_fee_rate = 0.001  # 0.1% per trade
                total_fees = total_volume * estimated_fee_rate
            
            # Calculate gross PnL trade by trade, then cumulative
            # First calculate individual trade fees
            trade_fees = 0.0
            if "Fee" in tb_real_plot.columns:
                trade_fees = tb_real_plot["Fee"]
            elif "Commission" in tb_real_plot.columns:
                trade_fees = tb_real_plot["Commission"]
            elif "Fees" in tb_real_plot.columns:
                trade_fees = tb_real_plot["Fees"]
            else:
                # Estimate fees based on typical trading fees (0.1% per trade)
                estimated_fee_rate = 0.001  # 0.1% per trade
                trade_fees = tb_real_plot["size_usdt"] * estimated_fee_rate
            
            # Calculate gross PnL trade by trade: gross_pnl = pnl - trade_fee
            gross_pnl = tb_real_plot["pnl"] - trade_fees
            gross_cum_pnl = gross_pnl.cumsum()
            
            fig.add_trace(
                go.Scatter(
                    x=tb_real_plot["Close_time"], 
                    y=gross_cum_pnl, 
                    name="Real-time (Gross)", 
                    mode="lines", 
                    line=dict(color="#FF6B6B", width=1, dash="dot"),
                    hovertemplate="<b>Date:</b> %{x}<br><b>Gross PnL:</b> %{y:.2f}<extra></extra>"
                ),
                row=1, col=1
            )



        # Upper subplot: Asset Price Evolution (second y-axis)
        if not md_plot.empty:
            # Debug: Print market data info
            print(f"Market data columns: {list(md_plot.columns)}")
            print(f"Market data shape: {md_plot.shape}")
            print(f"Market data head: {md_plot.head()}")
            
            # Check for common price column names
            price_column = None
            if "close" in md_plot.columns:
                price_column = "close"
            
            print(f"Selected price column: {price_column}")
            
            if price_column:
                # Align market data with trade timeline
                start_time = tb_real_plot["Close_time"].min()
                end_time = tb_real_plot["Close_time"].max()
                print(f"Trade timeline: {start_time} to {end_time}")
                
                md_filtered = md_plot[(md_plot["timestamp"] >= start_time) & (md_plot["timestamp"] <= end_time)]
                print(f"Filtered market data shape: {md_filtered.shape}")
                
                if not md_filtered.empty:
                    print(f"Price range: {md_filtered[price_column].min()} - {md_filtered[price_column].max()}")
                    # Ensure timestamps are properly formatted and aligned
                    md_filtered = md_filtered.copy()
                    md_filtered["formatted_timestamp"] = pd.to_datetime(md_filtered["timestamp"])
                    
                    print(f"Trade Close_time range: {tb_real_plot['Close_time'].min()} - {tb_real_plot['Close_time'].max()}")
                    print(f"Market timestamp range: {md_filtered['formatted_timestamp'].min()} - {md_filtered['formatted_timestamp'].max()}")
                    
                    fig.add_trace(
                        go.Scatter(
                            x=md_filtered["formatted_timestamp"], 
                            y=md_filtered[price_column], 
                            name="Asset Price", 
                            mode="lines", 
                            line=dict(color="#FFA500", width=2),
                            hovertemplate="<b>Date:</b> %{x}<br><b>Asset Price:</b> %{y:.2f}<extra></extra>"
                        ),
                        row=1, col=1, secondary_y=True
                    )
                else:
                    print("No market data in trade timeline range")
            else:
                print("No price column found in market data")
        else:
            print("Market data is empty")
   # Lower subplot: Drawdown bars (using real-time data)
        fig.add_trace(
            go.Bar(
                x=tb_real_plot["Close_time"], 
                y=tb_real_plot["drawdown"], 
                name="Drawdown", 
                marker=dict(
                    color="#3a3a3a",  # Medium grey bars
                    opacity=0.8,
                    line=dict(color="#3a3a3a", width=0)
                ),
                width=0.8,
                hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Lower subplot: Drawdown points on top of bars
        fig.add_trace(
            go.Scatter(
                x=tb_real_plot["Close_time"], 
                y=tb_real_plot["drawdown"], 
                name="Drawdown Points",
                mode="markers",
                marker=dict(
                    color="#FF4560",  # Red dots
                    size=4,
                    line=dict(color="#FF4560", width=1)
                ),
                showlegend=False,
                hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>"
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            template="plotly_dark",
            height=600,  # Increased height for two panels
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1,
                font=dict(color='#ffffff', size=10)
            ),
            margin=dict(l=50, r=50, t=80, b=50),
            barmode='overlay'
        )
        
        # Update axes
        # Upper subplot (Equity Curve) - add gridlines and styling
        fig.update_yaxes(
            title_text="Cumulative PnL", 
            row=1, col=1,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff'),
            side="left"
        )

        # Add second y-axis for Asset Price (right side) - only for the upper subplot
        fig.update_yaxes(
            title_text="Asset Price",
            secondary_y=True,
            row=1, col=1,
            showgrid=False,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff'),
            side="right"
        )
        

        
        # Lower subplot (Drawdown) - invert y-axis so drawdowns go downward
        fig.update_yaxes(
            title_text="Drawdown (%)", 
            row=2, col=1,
            autorange="reversed",  # Invert y-axis so drawdowns extend downward
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        # Update x-axis (shared) - add gridlines and styling for both subplots
        fig.update_xaxes(
            title_text="", 
            row=1, col=1,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=0.8,
            tickfont=dict(size=9, color='#ffffff')
        )
        
        fig.update_xaxes(
            title_text="Time", 
            row=2, col=1,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        return fig

    @staticmethod
    def winrate_vs_avgpnl(tb: pd.DataFrame) -> go.Figure:
        num_trades = len(tb)
        win_rate = tb["is_win"].mean() * 100 if num_trades > 0 else 0.0
        # Use gross PnL for average PnL calculation
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        avg_pnl = tb[pnl_column].mean() if num_trades > 0 else 0.0
        
        fig = go.Figure(data=[
            go.Bar(x=["Win Rate (%)", "Avg PnL"], y=[win_rate, avg_pnl], marker_color=["#00E396", "#FEB019"])
        ])
        fig.update_layout(template="plotly_dark", height=300, yaxis_title="Value")
        return fig

    @staticmethod
    def trade_frequency_by_time(tb: pd.DataFrame) -> go.Figure:
        """Show number of trades per date with bars split between positive (green) and negative (red) trades"""
        if tb.empty:
            return go.Figure()
        
        # Group trades by date and count positive vs negative trades
        tb_copy = tb.copy()
        tb_copy['date'] = tb_copy['Close_time'].dt.date
        
        # Count positive and negative trades per date
        positive_trades = tb_copy[tb_copy['pnl'] >= 0].groupby('date').size()
        negative_trades = tb_copy[tb_copy['pnl'] < 0].groupby('date').size()
        
        # Get all unique dates and ensure both series have the same index
        all_dates = sorted(tb_copy['date'].unique())
        positive_trades = positive_trades.reindex(all_dates, fill_value=0)
        negative_trades = negative_trades.reindex(all_dates, fill_value=0)
        
        # Create the plot with stacked bars
        fig = go.Figure()

        # Add positive trades (green)
        fig.add_trace(go.Bar(
            x=all_dates, 
            y=positive_trades.values, 
            name="Positive Trades",
            marker_color="#00E396",
            opacity=0.8,
            hovertemplate="<b>Date:</b> %{x}<br><b>Positive Trades:</b> %{y}<extra></extra>"
        ))
        
        # Add negative trades (red) - stacked on top of positive
        fig.add_trace(go.Bar(
            x=all_dates, 
            y=negative_trades.values, 
            name="Negative Trades",
            marker_color="#FF4560",
            opacity=0.8,
            hovertemplate="<b>Date:</b> %{x}<br><b>Negative Trades:</b> %{y}<extra></extra>"
        ))
        
        fig.update_layout(
            template="plotly_dark", 
            height=300, 
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1,
                font=dict(color='#ffffff', size=10)
            ),
            margin=dict(l=50, r=50, t=50, b=50),
            barmode='stack'  # Stack the bars
        )
        
        # Update axes to match equity curve styling
        fig.update_yaxes(
            title_text="Number of Trades",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        fig.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        return fig

    @staticmethod
    def trade_frequency_by_day_for_month(tb: pd.DataFrame, selected_month: str) -> go.Figure:
        """Show daily trade frequency for a specific month"""
        # Ensure month column exists
        if "month" not in tb.columns and "Open_time" in tb.columns:
            tb = tb.copy()
            tb["month"] = tb["Open_time"].dt.month_name()
        
        # Filter data for the selected month
        month_data = tb[tb["month"] == selected_month]
        daily = month_data.groupby("dow").size()
        
        # Order days chronologically
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = daily.reindex([d for d in day_order if d in daily.index])
        
        # Create the plot with daily data
        fig = go.Figure()
        
        # Add daily data bars
        fig.add_trace(go.Bar(
            x=daily.index, 
            y=daily.values, 
            name=f"Trades by Day - {selected_month}",
            marker_color="#00E396",
            opacity=0.8,
            hovertemplate="<b>Day:</b> %{x}<br><b>Trades:</b> %{y}<extra></extra>"
        ))
        
        fig.update_layout(
            template="plotly_dark", 
            height=300, 
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Update axes to match equity curve styling
        fig.update_yaxes(
            title_text="Number of Trades",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        fig.update_xaxes(
            title_text="Day of Week",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        return fig

    @staticmethod
    def trade_frequency_by_hour_for_day(tb: pd.DataFrame, selected_day: str) -> go.Figure:
        """Show hourly trade frequency for a specific day"""
        # Filter data for the selected day
        day_data = tb[tb["dow"] == selected_day]
        hourly = day_data.groupby("hour").size()
        
        # Order hours chronologically (0-23)
        hourly = hourly.reindex(sorted(hourly.index))
        
        # Create the plot with hourly data
        fig = go.Figure()
        
        # Add hourly data bars
        fig.add_trace(go.Bar(
            x=hourly.index, 
            y=hourly.values, 
            name=f"Trades by Hour - {selected_day}",
            marker_color="#008FFB",
            opacity=0.8,
            hovertemplate="<b>Hour:</b> %{x}<br><b>Trades:</b> %{y}<extra></extra>"
        ))

        fig.update_layout(
            template="plotly_dark", 
            height=300, 
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Update axes to match equity curve styling
        fig.update_yaxes(
            title_text="Number of Trades",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        fig.update_xaxes(
            title_text="Hour of Day",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        return fig

    @staticmethod
    def trade_size_vs_pnl(tb: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        
        # Calculate PnL as percentage of trade size
        tb_copy = tb.copy()
        tb_copy['pnl_percentage'] = np.where(
            tb_copy['size_usdt'] != 0, 
            (tb_copy['pnl'] / tb_copy['size_usdt']) * 100, 
            0
        )
        
        # Cap extreme percentage values to reasonable range
        tb_copy['pnl_percentage'] = np.clip(tb_copy['pnl_percentage'], -50, 50)
        
        # Add scatter plot with consistent styling
        fig.add_trace(
            go.Scatter(
                x=tb_copy["size_usdt"], 
                y=tb_copy["pnl_percentage"], 
                mode="markers", 
                name="Trade Size vs PnL",
                marker=dict(
                    color=tb_copy["pnl_percentage"], 
                    colorscale="RdYlGn", 
                    size=6,
                    line=dict(color='rgba(255,255,255,0.2)', width=1)
                ),
                hovertemplate="<b>Size:</b> %{x:.2f} USDT<br><b>Performance:</b> %{y:.2f}%<extra></extra>"
            )
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=300,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Update axes to match equity/drawdown styling
        fig.update_yaxes(
            title_text="Performance (%)",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff'),
            # Adapt y-axis range to actual data with padding
            range=[
                tb_copy['pnl_percentage'].min() - 1,  # Add 1% padding below
                tb_copy['pnl_percentage'].max() + 1   # Add 1% padding above
            ]
        )
        
        fig.update_xaxes(
            title_text="Trade Size (USDT)",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        return fig

    @staticmethod
    def expectation_vs_reality(tb: pd.DataFrame, score_column: Optional[str] = None) -> Optional[go.Figure]:
        if score_column is None:
            for col in ["Confidence", "Score", "Signal"]:
                if col in tb.columns:
                    score_column = col
                    break
        if score_column is None:
            return None
        
        # Use gross PnL for scatter plot
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        fig = go.Figure(data=[
            go.Scatter(x=tb[score_column], y=tb[pnl_column], mode="markers", marker=dict(color=tb[pnl_column], colorscale="RdYlGn", size=8))
        ])
        fig.update_layout(template="plotly_dark", height=300, xaxis_title=f"{score_column}", yaxis_title="Actual PnL")
        return fig

    @staticmethod
    def pnl_histogram(tb: pd.DataFrame) -> go.Figure:
        # Create histogram data manually to match drawdown style
        # Use gross PnL for distribution
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        hist, bin_edges = np.histogram(tb[pnl_column], bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        fig = go.Figure()
        
        # Add histogram bars (like drawdown bars)
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=hist,
                name="PnL Distribution",
                marker=dict(
                    color="#3a3a3a",  # Same dark grey as drawdown bars
                    opacity=0.8,
                    line=dict(color="#3a3a3a", width=0)
                ),
                width=0.001,  # Very thin bars like drawdown curve
                hovertemplate="<b>PnL Range:</b> %{x:.2f}<br><b>Frequency:</b> %{y}<extra></extra>"
            )
        )
        
        # Add points on top of bars (like drawdown points)
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=hist,
                name="Distribution Points",
                mode="markers",
                marker=dict(
                    color="#FF4560",  # Same red as drawdown points
                    size=4,
                    line=dict(color="#FF4560", width=1)
                ),
                showlegend=False,
                hovertemplate="<b>PnL Range:</b> %{x:.2f}<br><b>Frequency:</b> %{y}<extra></extra>"
            )
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=300,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50),
            barmode='overlay'
        )
        
        # Update axes to match drawdown styling
        fig.update_yaxes(
            title_text="Frequency",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.3,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        fig.update_xaxes(
            title_text="PnL",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.3,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        return fig

    @staticmethod
    def pnl_timeline_with_markers(tb: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        
        # Use gross PnL for timeline
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        cum_pnl_column = 'cum_gross_pnl' if 'cum_gross_pnl' in tb.columns else 'cum_pnl'
        
        # Add cumulative PnL line (like equity curve)
        fig.add_trace(
            go.Scatter(
                x=tb["Close_time"], 
                y=tb[cum_pnl_column], 
                mode="lines", 
                name="Cumulative PnL",
                line=dict(
                    color="#8A2BE2",  # Same purple as equity curve
                    width=1
                ),
                hovertemplate="<b>Date:</b> %{x}<br><b>Cumulative PnL:</b> %{y:.2f}<extra></extra>"
            )
        )
        
        # Add trade PnL markers
        fig.add_trace(
            go.Scatter(
                x=tb["Close_time"], 
                y=tb[pnl_column], 
                mode="markers", 
                name="Trade PnL",
                marker=dict(
                    color=tb[pnl_column], 
                    colorscale="RdYlGn", 
                    size=4,
                    line=dict(color='rgba(255,255,255,0.2)', width=1)
                ),
                showlegend=False,
                hovertemplate="<b>Date:</b> %{x}<br><b>Trade PnL:</b> %{y:.2f}<extra></extra>"
            )
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=360,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Update axes to match equity/drawdown styling
        fig.update_yaxes(
            title_text="PnL",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        fig.update_xaxes(
            title_text="Time",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        return fig

    # --------- Page 2: Statistical Insight ---------
    @staticmethod
    def profit_vs_volatility(tb: pd.DataFrame) -> go.Figure:
        md = st.session_state.get('market_data', pd.DataFrame())
        tb_copy = tb.copy()



        if not md.empty:
            md_copy = md.copy()
            
            # VOLATILITY COMPUTATION EXPLANATION:
            # This is PRE-TRADE volatility, computed at the time of trade opening
            # For each trade, we look at the market volatility at the moment the trade was opened
            # This helps understand the market conditions when you decided to enter the trade
            # This is more useful for strategy adjustment as it shows entry conditions
            
            # Step 1: Calculate intra-candle volatility (High-Low spread)
            md_copy['volatility'] = md_copy['high'] - md_copy['low']
            

            
            # Ensure both timestamps are in the same format
            # Convert market timestamp to datetime if needed
            if md_copy['timestamp'].dtype == 'int64' or md_copy['timestamp'].dtype == 'float64':
                md_copy['timestamp'] = pd.to_datetime(md_copy['timestamp'], unit='ms')
            elif md_copy['timestamp'].dtype == 'object':
                md_copy['timestamp'] = pd.to_datetime(md_copy['timestamp'])
            
            # Ensure Open_time is in datetime format
            if tb_copy['Open_time'].dtype == 'object':
                tb_copy['Open_time'] = pd.to_datetime(tb_copy['Open_time'])
            

            
            # Step 2: Resample volatility to 1-minute intervals and take the mean
            # This smooths out the volatility data and makes it more consistent
            vol_resampled = (
                md_copy.set_index('timestamp')
                .resample('1min')['volatility']
                .mean()
                .fillna(0)
            )
            

            
            # Step 3: Assign volatility to each trade based on its open time
            # This gives us the market volatility at the moment each trade was opened
            # This is more useful for strategy adjustment as it shows entry conditions
            tb_copy['volatility'] = tb_copy['Open_time'].apply(lambda x: vol_resampled.asof(x))
            

        else:
            tb_copy['volatility'] = tb_copy['entry_vol']

        # Use logarithmic bucketing for better volatility distribution
        vol_values = tb_copy['volatility'].values
        if len(vol_values) > 0:
            # Filter out zero and negative values for log bucketing
            positive_vol = vol_values[vol_values > 0]
            
            if len(positive_vol) > 0:
                # Use logarithmic bucketing for better distribution
                min_vol = positive_vol.min()
                max_vol = positive_vol.max()
                
                # Create log-spaced buckets
                log_min = np.log10(min_vol)
                log_max = np.log10(max_vol)
                log_buckets = np.logspace(log_min, log_max, 7)  # 6 buckets + 1 endpoint
                
                def assign_vol_bucket(vol):
                    if vol <= 0:
                        return 'Very Low'
                    elif vol <= log_buckets[1]: return 'Very Low'
                    elif vol <= log_buckets[2]: return 'Low'
                    elif vol <= log_buckets[3]: return 'Medium'
                    elif vol <= log_buckets[4]: return 'High'
                    elif vol <= log_buckets[5]: return 'Very High'
                    else: return 'Extreme'
                
                tb_copy['vol_bucket'] = tb_copy['volatility'].apply(assign_vol_bucket)
                vol_labels = {
                    'Very Low': 'Very Low',
                    'Low': 'Low',
                    'Medium': 'Medium',
                    'High': 'High',
                    'Very High': 'Very High',
                    'Extreme': 'Extreme'
                }
            else:
                # Fallback to percentile if no positive values
                q16, q33, q50, q66, q83 = np.percentile(vol_values, [16.67, 33.33, 50, 66.67, 83.33])
                
                def assign_vol_bucket(vol):
                    if vol <= q16: return 'Very Low'
                    elif vol <= q33: return 'Low'
                    elif vol <= q50: return 'Medium'
                    elif vol <= q66: return 'High'
                    elif vol <= q83: return 'Very High'
                    else: return 'Extreme'
                
                tb_copy['vol_bucket'] = tb_copy['volatility'].apply(assign_vol_bucket)
                vol_labels = {
                    'Very Low': 'Very Low',
                    'Low': 'Low',
                    'Medium': 'Medium',
                    'High': 'High',
                    'Very High': 'Very High',
                    'Extreme': 'Extreme'
                }
        else:
            tb_copy['vol_bucket'] = 'No Data'
            vol_labels = {'No Data': 'No Data'}

        # Calculate positive and negative PnL by volatility bucket
        # Use gross PnL for volatility analysis
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb_copy.columns else 'pnl'
        grouped_positive = tb_copy[tb_copy[pnl_column] > 0].groupby('vol_bucket')[pnl_column].sum().reset_index()
        grouped_negative = tb_copy[tb_copy[pnl_column] <= 0].groupby('vol_bucket')[pnl_column].sum().reset_index()
        
        # Apply the labels with boundaries
        grouped_positive['vol_bucket_labeled'] = grouped_positive['vol_bucket'].map(vol_labels)
        grouped_negative['vol_bucket_labeled'] = grouped_negative['vol_bucket'].map(vol_labels)

        # Helper function to create vol_ranges for a group
        def create_vol_ranges(buckets, tb_data):
            vol_ranges = []
            for bucket in buckets:
                if bucket in tb_data['vol_bucket'].values:
                    bucket_data = tb_data[tb_data['vol_bucket'] == bucket]
                    min_vol = bucket_data['volatility'].min()
                    max_vol = bucket_data['volatility'].max()
                    vol_ranges.append(f"{min_vol:.4f}-{max_vol:.4f}")
                else:
                    vol_ranges.append("N/A")
            return vol_ranges
        
        # Add actual volatility values to labels for positive group
        positive_vol_ranges = create_vol_ranges(grouped_positive['vol_bucket'], tb_copy)
        grouped_positive['vol_range'] = positive_vol_ranges
        grouped_positive['x_label'] = grouped_positive['vol_bucket_labeled'].astype(str) + '<br>(' + grouped_positive['vol_range'] + ')'
        
        # Add actual volatility values to labels for negative group
        negative_vol_ranges = create_vol_ranges(grouped_negative['vol_bucket'], tb_copy)
        grouped_negative['vol_range'] = negative_vol_ranges
        grouped_negative['x_label'] = grouped_negative['vol_bucket_labeled'].astype(str) + '<br>(' + grouped_negative['vol_range'] + ')'

        # Create subplots with shared x-axis
        fig = make_subplots(
            rows=2, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.5, 0.5]  # Equal height for both subplots
        )

        # Define colors for each volatility bucket
        bucket_colors = {
            'Very Low': '#00E396',    # Green
            'Low': '#FFB019',         # Yellow
            'Medium': '#FF6B6B',      # Light Red
            'High': '#FF4560',        # Red
            'Very High': '#FF1744',   # Dark Red
            'Extreme': '#D50000'      # Very Dark Red
        }
        
        # Create color arrays for positive and negative plots
        positive_colors = [bucket_colors.get(bucket, '#00E396') for bucket in grouped_positive['vol_bucket']]
        negative_colors = [bucket_colors.get(bucket, '#FF4560') for bucket in grouped_negative['vol_bucket']]

        # Upper subplot: Positive PnL
        fig.add_trace(
            go.Bar(
                x=grouped_positive['x_label'],
                y=grouped_positive[pnl_column],
                name="Positive PnL",
                marker=dict(color=positive_colors, opacity=0.8),
                hovertemplate="<b>Volatility Regime:</b> %{x}<br><b>Positive PnL:</b> %{y:.2f} USDT<extra></extra>",
                showlegend=False
            ),
            row=1, col=1
        )

        # Lower subplot: Negative PnL (inverted)
        fig.add_trace(
            go.Bar(
                x=grouped_negative['x_label'],
                y=abs(grouped_negative[pnl_column]),  # Convert negative values to positive for display
                name="Negative PnL",
                marker=dict(color=negative_colors, opacity=0.8),
                hovertemplate="<b>Volatility Regime:</b> %{x}<br><b>Negative PnL:</b> %{customdata:.2f} USDT<extra></extra>",
                customdata=grouped_negative[pnl_column],  # Keep original negative values for tooltip
                showlegend=False
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            template="plotly_dark",
            height=600,  # Increased height for two subplots
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )

        # Update axes
        fig.update_yaxes(
            title_text="", 
            row=1, col=1,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.8,
            tickfont=dict(size=9, color='#ffffff'),
            range=[0, None]  # Start from 0 for positive values
        )
        
        fig.update_yaxes(
            title_text="", 
            row=2, col=1,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.8,
            tickfont=dict(size=9, color='#ffffff'),
            autorange="reversed"  # Invert the y-axis so negative values extend downward
        )
        
        fig.update_xaxes(
            title_text="Volatility Regime", 
            row=2, col=1,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )

        return fig

    @staticmethod
    def get_trade_counts_by_volatility_regime(tb: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Get trade counts and average performance split by High Volatility and Stable regimes for positive and negative trades.
        Uses the same threshold as regime_split_equity (70th percentile).
        
        Returns:
            Tuple of (positive_trades_df, negative_trades_df, vol_threshold) with columns: regime, count, avg_pnl
        """
        tb_copy = tb.copy()
        
        # Use entry_vol for consistency with regime_split_equity
        # Calculate volatility threshold at 70th percentile
        vol_threshold = tb_copy["entry_vol"].quantile(0.7)
        
        # Split into High Volatility and Stable regimes
        tb_copy['regime'] = tb_copy['entry_vol'].apply(
            lambda x: 'High Volatility Regime' if x >= vol_threshold else 'Stable Regime'
        )
        
        # Use gross PnL if available, otherwise use pnl
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb_copy.columns else 'pnl'
        
        # Count and calculate average PnL for positive trades by regime
        positive_trades = tb_copy[tb_copy[pnl_column] > 0].groupby('regime').agg({
            pnl_column: ['count', 'mean']
        }).reset_index()
        positive_trades.columns = ['regime', 'count', 'avg_pnl']
        
        # Count and calculate average PnL for negative trades by regime
        negative_trades = tb_copy[tb_copy[pnl_column] <= 0].groupby('regime').agg({
            pnl_column: ['count', 'mean']
        }).reset_index()
        negative_trades.columns = ['regime', 'count', 'avg_pnl']
        
        # Ensure both dataframes have both regimes (fill with 0 if missing)
        all_regimes = ['High Volatility Regime', 'Stable Regime']
        
        for regime in all_regimes:
            if regime not in positive_trades['regime'].values:
                positive_trades = pd.concat([
                    positive_trades,
                    pd.DataFrame([{'regime': regime, 'count': 0, 'avg_pnl': 0.0}])
                ], ignore_index=True)
            
            if regime not in negative_trades['regime'].values:
                negative_trades = pd.concat([
                    negative_trades,
                    pd.DataFrame([{'regime': regime, 'count': 0, 'avg_pnl': 0.0}])
                ], ignore_index=True)
        
        # Sort to have High Volatility first, then Stable
        regime_order = ['High Volatility Regime', 'Stable Regime']
        positive_trades['regime'] = pd.Categorical(positive_trades['regime'], categories=regime_order, ordered=True)
        negative_trades['regime'] = pd.Categorical(negative_trades['regime'], categories=regime_order, ordered=True)
        positive_trades = positive_trades.sort_values('regime').reset_index(drop=True)
        negative_trades = negative_trades.sort_values('regime').reset_index(drop=True)
        
        return positive_trades, negative_trades, vol_threshold

    @staticmethod
    def profit_vs_trend(tb: pd.DataFrame) -> go.Figure:
        # TREND COMPUTATION EXPLANATION:
        # This is PRE-TRADE trend, computed at the time of trade opening
        # For each trade, we look at the market trend at the moment the trade was opened
        # This helps understand the market direction when you decided to enter the trade
        # This is more useful for strategy adjustment as it shows entry conditions
        
        tb_copy = tb.copy()
        
        # Use improved bucketing for better trend distribution
        trend_values = tb_copy['entry_trend'].values
        if len(trend_values) > 0:
            # Check if we have enough variation in trend values
            min_trend = trend_values.min()
            max_trend = trend_values.max()
            trend_range = max_trend - min_trend
            
            # If range is too small, use percentile-based bucketing
            if trend_range < 0.001 or np.std(trend_values) < 0.001:
                # Use percentile-based bucketing for small ranges
                q10, q25, q40, q60, q75, q90 = np.percentile(trend_values, [10, 25, 40, 60, 75, 90])
                
                def assign_trend_bucket(trend):
                    if trend <= q10: return 'Very Low'
                    elif trend <= q25: return 'Low'
                    elif trend <= q40: return 'Medium'
                    elif trend <= q60: return 'High'
                    elif trend <= q75: return 'Very High'
                    else: return 'Extreme'
                
                tb_copy['trend_bucket'] = tb_copy['entry_trend'].apply(assign_trend_bucket)
                trend_labels = {
                    'Very Low': 'Very Low',
                    'Low': 'Low',
                    'Medium': 'Medium',
                    'High': 'High',
                    'Very High': 'Very High',
                    'Extreme': 'Extreme'
                }
            else:
                non_zero_trend = trend_values[trend_values != 0]
                
                if len(non_zero_trend) > 0:
                    # Create log-spaced buckets based on absolute values
                    abs_trends = np.abs(non_zero_trend)
                    min_abs = abs_trends.min()
                    max_abs = abs_trends.max()
                    
                    # Ensure we have a reasonable range for log bucketing
                    if max_abs / min_abs > 10:  # Only use log if we have significant range
                        log_min = np.log10(min_abs)
                        log_max = np.log10(max_abs)
                        log_buckets = np.logspace(log_min, log_max, 7)  # 6 buckets + 1 endpoint
                        
                        def assign_trend_bucket(trend):
                            if trend == 0:
                                return 'Neutral'
                            abs_trend = abs(trend)
                            if abs_trend <= log_buckets[1]: 
                                return 'Very Low'
                            elif abs_trend <= log_buckets[2]: 
                                return 'Low'
                            elif abs_trend <= log_buckets[3]: 
                                return 'Medium'
                            elif abs_trend <= log_buckets[4]: 
                                return 'High'
                            elif abs_trend <= log_buckets[5]: 
                                return 'Very High'
                            else: 
                                return 'Extreme'
                    else:
                        # Use linear bucketing for smaller ranges
                        def assign_trend_bucket(trend):
                            if trend == 0:
                                return 'Neutral'
                            abs_trend = abs(trend)
                            if abs_trend <= max_abs * 0.2: 
                                return 'Very Low'
                            elif abs_trend <= max_abs * 0.4: 
                                return 'Low'
                            elif abs_trend <= max_abs * 0.6: 
                                return 'Medium'
                            elif abs_trend <= max_abs * 0.8: 
                                return 'High'
                            elif abs_trend <= max_abs * 0.95: 
                                return 'Very High'
                            else: 
                                return 'Extreme'
                    
                    tb_copy['trend_bucket'] = tb_copy['entry_trend'].apply(assign_trend_bucket)
                    trend_labels = {
                        'Very Low': 'Very Low',
                        'Low': 'Low',
                        'Medium': 'Medium',
                        'High': 'High',
                        'Very High': 'Very High',
                        'Extreme': 'Extreme',
                        'Neutral': 'Neutral'
                    }
                else:
                    # All values are zero
                    tb_copy['trend_bucket'] = 'Neutral'
                    trend_labels = {'Neutral': 'Neutral'}


        # Calculate positive and negative PnL by trend bucket
        # Use gross PnL for trend analysis
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb_copy.columns else 'pnl'
        grouped_positive = tb_copy[tb_copy[pnl_column] > 0].groupby('trend_bucket')[pnl_column].sum().reset_index()
        grouped_negative = tb_copy[tb_copy[pnl_column] <= 0].groupby('trend_bucket')[pnl_column].sum().reset_index()
        
        # Apply the labels with boundaries
        grouped_positive['trend_bucket_labeled'] = grouped_positive['trend_bucket'].map(trend_labels)
        grouped_negative['trend_bucket_labeled'] = grouped_negative['trend_bucket'].map(trend_labels)
        
        # Helper function to create trend_ranges for a group
        def create_trend_ranges(buckets, tb_data):
            trend_ranges = []
            for bucket in buckets:
                if bucket in tb_data['trend_bucket'].values:
                    bucket_data = tb_data[tb_data['trend_bucket'] == bucket]
                    min_trend = bucket_data['entry_trend'].min()
                    max_trend = bucket_data['entry_trend'].max()
                    
                    # Add direction indicators
                    if bucket == 'Neutral':
                        trend_ranges.append("Neutral")
                    elif min_trend >= 0 and max_trend >= 0:
                        trend_ranges.append(f"↗ {min_trend:.3f}-{max_trend:.3f}")
                    elif min_trend <= 0 and max_trend <= 0:
                        trend_ranges.append(f"↘ {min_trend:.3f}-{max_trend:.3f}")
                    else:
                        trend_ranges.append(f"↔ {min_trend:.3f}-{max_trend:.3f}")
                else:
                    trend_ranges.append("N/A")
            return trend_ranges
        
        # Add actual trend values to labels for positive group
        positive_trend_ranges = create_trend_ranges(grouped_positive['trend_bucket'], tb_copy)
        grouped_positive['trend_range'] = positive_trend_ranges
        grouped_positive['x_label'] = grouped_positive['trend_bucket_labeled'].astype(str) + '<br>(' + grouped_positive['trend_range'] + ')'
        
        # Add actual trend values to labels for negative group
        negative_trend_ranges = create_trend_ranges(grouped_negative['trend_bucket'], tb_copy)
        grouped_negative['trend_range'] = negative_trend_ranges
        grouped_negative['x_label'] = grouped_negative['trend_bucket_labeled'].astype(str) + '<br>(' + grouped_negative['trend_range'] + ')'

        # Create subplots with shared x-axis
        fig = make_subplots(
            rows=2, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.5, 0.5]  # Equal height for both subplots
        )

        # Define colors for each trend bucket
        bucket_colors = {
            'Very Low': '#00E396',    # Green
            'Low': '#FFB019',         # Yellow
            'Medium': '#FF6B6B',      # Light Red
            'High': '#FF4560',        # Red
            'Very High': '#FF1744',   # Dark Red
            'Extreme': '#D50000',     # Very Dark Red
            'Neutral': '#8E8E93'      # Gray
        }
        
        # Create color arrays for positive and negative plots
        positive_colors = [bucket_colors.get(bucket, '#00E396') for bucket in grouped_positive['trend_bucket']]
        negative_colors = [bucket_colors.get(bucket, '#FF4560') for bucket in grouped_negative['trend_bucket']]

        # Upper subplot: Positive PnL
        fig.add_trace(
            go.Bar(
                x=grouped_positive['x_label'],
                y=grouped_positive[pnl_column],
                name="Positive PnL",
                marker=dict(color=positive_colors, opacity=0.8),
                hovertemplate="<b>Trend Regime:</b> %{x}<br><b>Positive PnL:</b> %{y:.2f} USDT<extra></extra>",
                showlegend=False
            ),
            row=1, col=1
        )

        # Lower subplot: Negative PnL (inverted)
        fig.add_trace(
            go.Bar(
                x=grouped_negative['x_label'],
                y=abs(grouped_negative[pnl_column]),  # Convert negative values to positive for display
                name="Negative PnL",
                marker=dict(color=negative_colors, opacity=0.8),
                hovertemplate="<b>Trend Regime:</b> %{x}<br><b>Negative PnL:</b> %{customdata:.2f} USDT<extra></extra>",
                customdata=grouped_negative[pnl_column],  # Keep original negative values for tooltip
                showlegend=False
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            template="plotly_dark",
            height=600,  # Increased height for two subplots
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )

        # Update axes
        fig.update_yaxes(
            title_text="", 
            row=1, col=1,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.8,
            tickfont=dict(size=9, color='#ffffff'),
            range=[0, None]  # Start from 0 for positive values
        )
        
        fig.update_yaxes(
            title_text="", 
            row=2, col=1,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.8,
            tickfont=dict(size=9, color='#ffffff'),
            autorange="reversed"  # Invert the y-axis so negative values extend downward
        )
        
        fig.update_xaxes(
            title_text="Trend Regime", 
            row=2, col=1,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )

        return fig

    @staticmethod
    def profit_velocity(tb: pd.DataFrame) -> go.Figure:
        # Calculate mean and median for metric boxes
        tb_copy = tb.copy()
        profit_per_min = tb_copy['profit_per_minute'].dropna()
        if len(profit_per_min) > 0:
            p1, p99 = np.percentile(profit_per_min, [1, 99])
            tb_copy['profit_per_minute_capped'] = np.clip(tb_copy['profit_per_minute'], p1, p99)
        else:
            tb_copy['profit_per_minute_capped'] = tb_copy['profit_per_minute']
        
        mean_profit_per_min = tb_copy['profit_per_minute_capped'].mean()
        median_profit_per_min = tb_copy['profit_per_minute_capped'].median()
        
        # Simple blue histogram
        fig = px.histogram(tb_copy, x="profit_per_minute_capped", nbins=30, template="plotly_dark")
        fig.update_layout(
            height=300, 
            xaxis_title="Return per Minute (%)", 
            yaxis_title="Frequency"
        )
        
        return fig, mean_profit_per_min, median_profit_per_min

    @staticmethod
    def edge_ratio_plot(tb: pd.DataFrame) -> go.Figure:
        # Use gross PnL for coloring
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        fig = go.Figure(data=[
            go.Scatter(x=tb["duration_min"], y=tb["return_frac"], mode="markers", marker=dict(color=tb[pnl_column], colorscale="RdYlGn", size=8))
        ])
        fig.update_layout(template="plotly_dark", height=300, xaxis_title="Duration (min)", yaxis_title="Return Fraction")
        return fig

    @staticmethod
    def ttest_mean_profit(tb: pd.DataFrame) -> Tuple[float, float]:
        # One-sample t-test for mean > 0. Return (t_stat, p_value_one_sided)
        # Use gross PnL for statistical test
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        x = tb[pnl_column].dropna().values
        n = len(x)
        if n < 3:
            return float("nan"), float("nan")
        mean = x.mean()
        std = x.std(ddof=1)
        if std == 0:
            return float("inf") if mean > 0 else float("-inf"), 0.0 if mean > 0 else 1.0
        t_stat = mean / (std / math.sqrt(n))
        # one-sided p-value using survival function of t-dist; approximate with normal if SciPy unavailable
        try:
            from scipy.stats import t

            p_two_sided = 2 * (1 - t.cdf(abs(t_stat), df=n - 1))
        except Exception:
            # normal approx
            from math import erf

            p_two_sided = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / math.sqrt(2))))
        p_one_sided = p_two_sided / 2 if mean > 0 else 1 - (p_two_sided / 2)
        return t_stat, p_one_sided

    @staticmethod
    def ttest_mean_profit_plot(tb: pd.DataFrame) -> go.Figure:
        """Create a visualization for t-test results."""
        t_stat, p_value = Display.ttest_mean_profit(tb)
        
        # Format p-value for display
        p_value_formatted = f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}"
        
        # Create a simple bar chart showing the t-statistic and p-value
        fig = go.Figure()

        # Add t-statistic bar
        fig.add_trace(go.Bar(
            x=['T-Statistic'],
            y=[t_stat],
            name='T-Statistic',
            marker_color='#007aff',
            text=[f'{t_stat:.3f}'],
            textposition='auto'
        ))
        
        # Add p-value bar
        fig.add_trace(go.Bar(
            x=['P-Value'],
            y=[p_value],
            name='P-Value',
            marker_color='#ff9500',
            text=[p_value_formatted],
            textposition='auto'
        ))
        
        # Add significance threshold line
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                     annotation_text="α = 0.05", annotation_position="top right")

        fig.update_layout(
            template="plotly_dark",
            height=300,
            showlegend=True,
            yaxis_title="Value"
        )
        
        return fig

    @staticmethod
    def autocorr_plot(tb: pd.DataFrame, max_lag: int = 20) -> go.Figure:
        # Use gross PnL for autocorrelation analysis
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        series = tb[pnl_column].fillna(0.0)
        lags = list(range(1, max_lag + 1))
        acfs = [series.autocorr(lag=lag) for lag in lags]
        fig = go.Figure(data=[go.Bar(x=lags, y=acfs, marker_color="#008FFB")])
        fig.update_layout(template="plotly_dark", height=300, xaxis_title="Lag (trades)", yaxis_title="ACF")
        return fig

    @staticmethod
    def rolling_sharpe_sortino(tb: pd.DataFrame, window: int = 30) -> go.Figure:
        # ROLLING SHARPE/SORTINO ANALYSIS:
        # Enhanced with threshold bands and color-coded regimes
        # Key insights: risk-adjusted performance trends and regime shifts
        
        # Use gross PnL for rolling analysis
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        x = tb[pnl_column].fillna(0.0)
        roll_mean = x.rolling(window).mean()
        roll_std = x.rolling(window).std(ddof=1)
        sharpe = (roll_mean / (roll_std + 1e-12)) * np.sqrt(window)
        downside = x.where(x < 0, 0.0)
        downside_std = downside.rolling(window).std(ddof=1)
        sortino = (roll_mean / (downside_std + 1e-12)) * np.sqrt(window)

        fig = go.Figure()
        
        # 1. Add threshold bands with background shading
        # Adverse regime (Sharpe < 0) - Red background
        fig.add_hrect(
            y0=-5, y1=0, 
            fillcolor="rgba(255,69,96,0.1)", 
            layer="below", 
            line_width=0,
            annotation_text="Adverse Regime",
            annotation_position="top left",
            annotation=dict(font=dict(color="#FF4560", size=10))
        )
        
        # Excellent regime (Sharpe > 2) - Green background
        fig.add_hrect(
            y0=2, y1=5, 
            fillcolor="rgba(0,227,150,0.1)", 
            layer="below", 
            line_width=0,
            annotation_text="Excellent Regime",
            annotation_position="top right",
            annotation=dict(font=dict(color="#00E396", size=10))
        )
        
        # Break-even line (Sharpe = 0)
        fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="#FF4560", 
            line_width=1
        )
        
        # Minimum acceptable line (Sharpe = 1)
        fig.add_hline(
            y=1, 
            line_dash="dot", 
            line_color="#FFB019", 
            line_width=1
        )
        
        # Excellent line (Sharpe = 2)
        fig.add_hline(
            y=2, 
            line_dash="dot", 
            line_color="#00E396", 
            line_width=1
        )
        
        # 2. Color-coded Sharpe ratio based on performance
        def get_sharpe_color(sharpe_val):
            if sharpe_val > 1.5:
                return "#00E396"  # Green - Strong
            elif sharpe_val > 0:
                return "#FFB019"  # Yellow - Neutral
            else:
                return "#FF4560"  # Red - Adverse
        
        # Create color array for Sharpe line
        sharpe_colors = [get_sharpe_color(val) for val in sharpe]
        
        # 3. Add Sharpe ratio with dynamic colors
        fig.add_trace(go.Scatter(
            x=tb["Close_time"], 
            y=sharpe, 
            name="Rolling Sharpe", 
            line=dict(color="#8A2BE2", width=1),  # Purple - matches other plots
            hovertemplate="<b>Date:</b> %{x}<br><b>Sharpe:</b> %{y:.3f}<extra></extra>"
        ))
        
        # 4. Add Sortino ratio
        fig.add_trace(go.Scatter(
            x=tb["Close_time"], 
            y=sortino, 
            name="Rolling Sortino", 
            line=dict(color="#00D4AA", width=1),  # Teal - complementary to purple
            hovertemplate="<b>Date:</b> %{x}<br><b>Sortino:</b> %{y:.3f}<extra></extra>"
        ))
        
        # 5. Enhanced layout
        fig.update_layout(
            template="plotly_dark", 
            height=400,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1,
                font=dict(color='#ffffff', size=10)
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # 6. Update axes
        fig.update_yaxes(
            title_text="Sharpe/Sortino Ratio",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        fig.update_xaxes(
            title_text="Time",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        return fig

    @staticmethod
    def rolling_win_rate(tb: pd.DataFrame, window: int = 30) -> go.Figure:
        """
        Create rolling win rate plot with background zones and reference lines.
        
        Args:
            tb: Trade book DataFrame
            window: Rolling window size in number of trades
            
        Returns:
            Plotly figure showing rolling win rate over time
        """
        # Calculate rolling win rate
        x = tb["is_win"].fillna(0)
        rolling_wr = x.rolling(window, min_periods=5).mean() * 100  # Convert to percentage
        
        fig = go.Figure()
        
        # 1. Add threshold bands with background shading
        # Poor performance zone (Win Rate < 40%) - Red background
        fig.add_hrect(
            y0=0, y1=40, 
            fillcolor="rgba(255,69,96,0.1)", 
            layer="below", 
            line_width=0,
            annotation_text="Poor Performance",
            annotation_position="top left",
            annotation=dict(font=dict(color="#FF4560", size=10))
        )
        
        # Excellent performance zone (Win Rate > 70%) - Green background
        fig.add_hrect(
            y0=70, y1=100, 
            fillcolor="rgba(0,227,150,0.1)", 
            layer="below", 
            line_width=0,
            annotation_text="Excellent Performance",
            annotation_position="top right",
            annotation=dict(font=dict(color="#00E396", size=10))
        )
        
        # Break-even line (Win Rate = 50%)
        fig.add_hline(
            y=50, 
            line_dash="dash", 
            line_color="#FFB019", 
            line_width=1
        )
        
        # Good performance line (Win Rate = 60%)
        fig.add_hline(
            y=60, 
            line_dash="dot", 
            line_color="#FFB019", 
            line_width=1
        )
        
        # Excellent performance line (Win Rate = 70%)
        fig.add_hline(
            y=70, 
            line_dash="dot", 
            line_color="#00E396", 
            line_width=1
        )
        
        # 2. Add rolling win rate line
        fig.add_trace(go.Scatter(
            x=tb["Close_time"], 
            y=rolling_wr, 
            name="Rolling Win Rate", 
            line=dict(color="#8A2BE2", width=2),  # Purple - matches other plots
            hovertemplate="<b>Date:</b> %{x}<br><b>Win Rate:</b> %{y:.1f}%<extra></extra>"
        ))
        
        # 3. Enhanced layout
        fig.update_layout(
            template="plotly_dark", 
            height=400,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1,
                font=dict(color='#ffffff', size=10)
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # 4. Update axes
        fig.update_yaxes(
            title_text="Win Rate (%)",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff'),
            range=[0, 100]  # Win rate is always between 0 and 100%
        )
        
        fig.update_xaxes(
            title_text="Time",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        return fig

    @staticmethod
    def regime_split_equity(tb: pd.DataFrame) -> Tuple[go.Figure, float, float, float]:
        vol_thresh = tb["entry_vol"].quantile(0.7)
        high = tb[tb["entry_vol"] >= vol_thresh].copy()
        low = tb[tb["entry_vol"] < vol_thresh].copy()
        high["cum"] = high["pnl"].cumsum()
        low["cum"] = low["pnl"].cumsum()
        
        # Calculate final performance for each regime
        high_vol_performance = high["cum"].iloc[-1] if len(high) > 0 else 0.0
        stable_performance = low["cum"].iloc[-1] if len(low) > 0 else 0.0
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=high["Close_time"], y=high["cum"], name="High Vol Regime", line=dict(color="#FF4560", width=1)))
        fig.add_trace(go.Scatter(x=low["Close_time"], y=low["cum"], name="Stable Regime", line=dict(color="#00E396", width=1)))
        fig.update_layout(
            template="plotly_dark", 
            height=360,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1,
                font=dict(color='#ffffff', size=10)
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig, high_vol_performance, stable_performance, vol_thresh

    # --------- Page 3: Optimization & Sensitivity ---------
    @staticmethod
    def kelly_estimator(tb: pd.DataFrame) -> Tuple[float, float, float]:
        # Use gross PnL for Kelly estimation
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        wins = tb[tb[pnl_column] > 0][pnl_column]
        losses = tb[tb[pnl_column] <= 0][pnl_column]
        p = (len(wins) / len(tb)) if len(tb) > 0 else float("nan")
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0  # negative
        b = (avg_win / abs(avg_loss)) if avg_win > 0 and avg_loss < 0 else float("nan")
        kelly = p - (1 - p) / b if b and not np.isnan(b) else float("nan")
        return p, b, kelly

    @staticmethod
    def duration_vs_profit_heatmap(tb: pd.DataFrame) -> go.Figure:
        fig = px.density_heatmap(tb, x="duration_min", y="pnl", nbinsx=40, nbinsy=40, color_continuous_scale="Viridis", template="plotly_dark")
        fig.update_layout(height=380, xaxis_title="Duration (min)", yaxis_title="PnL")
        return fig

    @staticmethod
    def pnl_by_category(tb: pd.DataFrame, category_col: str = "Symbol") -> go.Figure:
        if category_col not in tb.columns:
            category_col = "Symbol" if "Symbol" in tb.columns else None
        if category_col is None:
            return px.bar(template="plotly_dark")
        agg = tb.groupby(category_col)["pnl"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(agg, x=category_col, y="pnl", template="plotly_dark")
        fig.update_layout(height=360)
        return fig

    @staticmethod
    def monte_carlo_equity(tb: pd.DataFrame, n_sims: int = 200, seed: int = 42) -> tuple[go.Figure, float]:
        rng = np.random.default_rng(seed)
        # Use gross PnL for Monte Carlo simulation
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        pnl = tb[pnl_column].dropna().values
        n = len(pnl)
        if n == 0:
            return go.Figure()
        
        # Get initial investment from first trade
        initial_investment = tb["size_usdt"].iloc[0] if len(tb) > 0 else 0
        
        fig = go.Figure()
        
        # Generate individual paths with bankruptcy consideration
        for _ in range(min(n_sims, 400)):
            sample = rng.choice(pnl, size=n, replace=True)
            cum_pnl = np.cumsum(sample)
            # Account for bankruptcy: if cum_pnl + initial_investment < 0, account becomes 0
            # So the minimum possible PnL is -initial_investment
            adjusted_pnl = np.maximum(cum_pnl, -initial_investment)
            fig.add_trace(go.Scatter(y=adjusted_pnl, mode="lines", line=dict(width=1, color="rgba(0,143,251,0.25)"), showlegend=False))
        
        # Generate paths for percentile calculation
        paths = np.vstack([
            np.cumsum(rng.choice(pnl, size=n, replace=True)) for _ in range(min(n_sims, 1000))
        ])
        
        # Apply bankruptcy logic to percentile paths
        for i in range(paths.shape[0]):
            # Cap the PnL at -initial_investment (you can't lose more than you have)
            paths[i] = np.maximum(paths[i], -initial_investment)
        
        # Calculate percentiles
        p1 = np.percentile(paths, 1, axis=0)
        p5 = np.percentile(paths, 5, axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        
        fig.add_trace(go.Scatter(y=p1, mode="lines", line=dict(color="#FF1744", width=2), name="1st pct"))
        fig.add_trace(go.Scatter(y=p5, mode="lines", line=dict(color="#FF4560", width=2), name="5th pct"))
        fig.add_trace(go.Scatter(y=p25, mode="lines", line=dict(color="#FF9800", width=2), name="25th pct"))
        fig.add_trace(go.Scatter(y=p50, mode="lines", line=dict(color="#00E396", width=2), name="Median"))
        
        # Calculate bankruptcy percentage
        bankruptcy_count = 0
        total_simulations = len(paths)
        
        for path in paths:
            # Check if any point in the path hits bankruptcy (PnL = -initial_investment)
            # Since we capped at -initial_investment, bankruptcy means hitting exactly that floor
            if np.any(path == -initial_investment):
                bankruptcy_count += 1
        
        bankruptcy_percentage = (bankruptcy_count / total_simulations) * 100 if total_simulations > 0 else 0
        
        fig.update_layout(template="plotly_dark", height=420)
        return fig, bankruptcy_percentage

    @staticmethod
    def rolling_expectation(tb: pd.DataFrame, window: int = 30) -> go.Figure:
        # Use gross PnL for rolling expectation
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        x = tb[pnl_column].fillna(0.0)
        wins = x.where(x > 0)
        losses = x.where(x <= 0)
        wr = (wins.notna().rolling(window).sum()) / (x.notna().rolling(window).sum())
        avg_win = wins.rolling(window).mean().fillna(0.0)
        avg_loss = losses.rolling(window).mean().fillna(0.0)
        mu = wr * avg_win - (1 - wr) * abs(avg_loss)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tb["Close_time"], y=mu, name="Rolling Expectation", line=dict(color="#00E396")))
        fig.update_layout(template="plotly_dark", height=360)
        return fig

    # --------- Page 4: Survivability & Risk ---------
    @staticmethod
    def max_drawdown_over_time(tb: pd.DataFrame) -> go.Figure:
        # Calculate drawdown based on cumulative PnL + initial investment
        tb_copy = tb.copy()
        
        # Get initial investment from first trade
        initial_investment = tb_copy["size_usdt"].iloc[0] if len(tb_copy) > 0 else 0
        
        # Calculate total account value (cumulative PnL + initial investment)
        tb_copy["total_value"] = tb_copy["cum_pnl"] + initial_investment
        tb_copy["running_max_value"] = tb_copy["total_value"].cummax()
        
        # Calculate drawdown as percentage of running maximum total value
        tb_copy["drawdown_pct"] = np.where(
            (tb_copy["running_max_value"] > 0) & (tb_copy["running_max_value"] > tb_copy["total_value"]),
            (tb_copy["running_max_value"] - tb_copy["total_value"]) / tb_copy["running_max_value"] * 100,
            0
        )
        
        fig = go.Figure()
        
        # Add drawdown bars (same style as equity curve drawdown)
        fig.add_trace(
            go.Bar(
                x=tb_copy["Close_time"], 
                y=tb_copy["drawdown_pct"], 
                name="Drawdown", 
                marker=dict(
                    color="#3a3a3a",  # Medium grey bars (same as equity curve)
                    opacity=0.8,
                    line=dict(color="#3a3a3a", width=0)
                ),
                width=0.8,
                hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>"
            )
        )
        
        # Add drawdown points on top of bars (same style as equity curve)
        fig.add_trace(
            go.Scatter(
                x=tb_copy["Close_time"], 
                y=tb_copy["drawdown_pct"], 
                name="Drawdown Points",
                mode="markers",
                marker=dict(
                    color="#FF4560",  # Red dots (same as equity curve)
                    size=4,
                    line=dict(color="#FF4560", width=1)
                ),
                showlegend=False,
                hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>"
            )
        )
        
        fig.update_layout(
            template="plotly_dark", 
            height=320, 
            yaxis_title="Drawdown (%)",
            showlegend=False,
            barmode='overlay'
        )
        
        # Update axes to match equity curve styling
        fig.update_yaxes(
            autorange="reversed",  # Invert y-axis so drawdowns extend downward
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=0.8,
            title_font=dict(size=10, color='#ffffff'),
            tickfont=dict(size=9, color='#ffffff')
        )
        
        return fig

    @staticmethod
    def rolling_vol_vs_pnl(tb: pd.DataFrame, md: pd.DataFrame, window_bars: int = 60) -> go.Figure:
        pnl_roll = tb.set_index("Close_time")["pnl"].rolling("3D").mean()  # 3-day rolling mean on close timeline
        vol_roll = md.set_index("timestamp")["ret"].rolling(window_bars).std().rename("mkt_vol")
        vol_roll = vol_roll.reindex(pnl_roll.index, method="nearest")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=pnl_roll.index, y=pnl_roll.values, name="Rolling PnL", line=dict(color="#00E396")), secondary_y=False)
        fig.add_trace(go.Scatter(x=vol_roll.index, y=vol_roll.values, name="Rolling Vol", line=dict(color="#FEB019")), secondary_y=True)
        fig.update_layout(template="plotly_dark", height=360)
        fig.update_yaxes(title_text="PnL (roll mean)", secondary_y=False)
        fig.update_yaxes(title_text="Market Vol (roll std)", secondary_y=True)
        return fig

    @staticmethod
    def var_plot(tb: pd.DataFrame, alpha: float = 0.95) -> go.Figure:
        # Use gross PnL for VaR analysis
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        q = np.quantile(tb[pnl_column].dropna(), 1 - alpha)  # VaR at alpha (e.g., 95%)
        
        # Create histogram data manually for bars and points
        pnl_data = tb[pnl_column].dropna()
        hist, bin_edges = np.histogram(pnl_data, bins=60)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        fig = go.Figure()
        
        # Add bars (same style as drawdown plot)
        fig.add_trace(
            go.Bar(
                x=bin_centers, 
                y=hist, 
                name="VaR Distribution", 
                marker=dict(
                    color="#3a3a3a",  # Medium grey bars (same as drawdown plot)
                    opacity=0.8,
                    line=dict(color="#3a3a3a", width=0)
                ),
                width=(bin_edges[1] - bin_edges[0]) * 0.01,
                hovertemplate="<b>PnL:</b> %{x:.2f}<br><b>Frequency:</b> %{y}<extra></extra>"
            )
        )
        
        # Add red points on top of bars (same style as drawdown plot)
        fig.add_trace(
            go.Scatter(
                x=bin_centers, 
                y=hist, 
                name="Distribution Points",
                mode="markers",
                marker=dict(
                    color="#FF4560",  # Red dots (same as drawdown plot)
                    size=4,
                    line=dict(color="#FF4560", width=1)
                ),
                showlegend=False,
                hovertemplate="<b>PnL:</b> %{x:.2f}<br><b>Frequency:</b> %{y}<extra></extra>"
            )
        )
        
        # Add VaR line
        fig.add_vline(
            x=q, 
            line_color="#FF4560", 
            line_dash="dash", 
            line_width=2,
            annotation_text=f"VaR {int(alpha*100)}% = {q:.2f}",
            annotation_position="top right"
        )
        
        fig.update_layout(
            template="plotly_dark", 
            height=320,
            xaxis_title="PnL",
            yaxis_title="Frequency",
            showlegend=False
        )
        return fig

    @staticmethod
    def trade_streaks(tb: pd.DataFrame) -> go.Figure:
        # Use gross PnL for streak analysis
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        outcomes = (tb[pnl_column] > 0).astype(int)
        
        # Calculate all streaks with their outcomes
        streaks_data = []
        current_streak = 0
        current_outcome = None
        
        for i, outcome in enumerate(outcomes):
            if current_outcome is None:
                current_outcome = outcome
                current_streak = 1
            elif outcome == current_outcome:
                current_streak += 1
            else:
                # Streak ended, record it
                if current_streak > 0:
                    streaks_data.append({
                        'length': current_streak,
                        'is_positive': bool(current_outcome)
                    })
                # Start new streak
                current_outcome = outcome
                current_streak = 1
        
        # Don't forget the last streak
        if current_streak > 0:
            streaks_data.append({
                'length': current_streak,
                'is_positive': bool(current_outcome)
            })
        
        if not streaks_data:
            return go.Figure()
        
        # Create DataFrame for easier manipulation
        df_streaks = pd.DataFrame(streaks_data)
        
        # Get unique streak lengths
        unique_lengths = sorted(df_streaks['length'].unique())
        
        # Calculate positive and negative counts for each streak length
        positive_counts = []
        negative_counts = []
        
        for length in unique_lengths:
            length_data = df_streaks[df_streaks['length'] == length]
            positive_count = len(length_data[length_data['is_positive'] == True])
            negative_count = len(length_data[length_data['is_positive'] == False])
            
            positive_counts.append(positive_count)
            negative_counts.append(negative_count)
        
        fig = go.Figure()
        
        # Add positive portion of bars (green)
        fig.add_trace(
            go.Bar(
                x=unique_lengths,
                y=positive_counts,
                name="Positive Trades",
                marker=dict(
                    color="#00E396",  # Green for positive
                    opacity=0.8,
                    line=dict(color="#00E396", width=0)
                ),
                hovertemplate="<b>Streak Length:</b> %{x}<br><b>Positive Trades:</b> %{y}<extra></extra>"
            )
        )
        
        # Add negative portion of bars (red)
        fig.add_trace(
            go.Bar(
                x=unique_lengths,
                y=negative_counts,
                name="Negative Trades",
                marker=dict(
                    color="#FF4560",  # Red for negative
                    opacity=0.8,
                    line=dict(color="#FF4560", width=0)
                ),
                hovertemplate="<b>Streak Length:</b> %{x}<br><b>Negative Trades:</b> %{y}<extra></extra>"
            )
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=320,
            xaxis_title="Streak Length",
            yaxis_title="Number of Trades",
            barmode='stack',  # Stack bars to show split
            showlegend=True,
            legend=dict(
                x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)', borderwidth=1,
                font=dict(color='#ffffff', size=10)
            )
        )
        return fig

    @staticmethod
    def sharpe_vs_skewness(tb: pd.DataFrame) -> go.Figure:
        # Use gross PnL for Sharpe and skewness analysis
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        pnl = tb[pnl_column].dropna()
        if len(pnl) < 30:
            return go.Figure()
        
        sharpe = pnl.mean() / pnl.std() if pnl.std() > 0 else 0
        skewness = pnl.skew()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[skewness], y=[sharpe], mode="markers", marker=dict(size=20, color="#00E396")))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(template="plotly_dark", height=320, xaxis_title="Skewness", yaxis_title="Sharpe Ratio")
        return fig

    @staticmethod
    def worst_case_5pct_curve(tb: pd.DataFrame, n_sims: int = 1000, seed: int = 7) -> go.Figure:
        rng = np.random.default_rng(seed)
        # Use gross PnL for worst case analysis
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        pnl = tb[pnl_column].dropna().values
        n = len(pnl)
        if n == 0:
            return go.Figure()
        
        paths = np.vstack([
            np.cumsum(rng.choice(pnl, size=n, replace=True)) for _ in range(n_sims)
        ])
        p5 = np.percentile(paths, 5, axis=0)
        p95 = np.percentile(paths, 95, axis=0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=p5, mode="lines", line=dict(color="#FF4560", width=2), name="5th percentile"))
        fig.add_trace(go.Scatter(y=p95, mode="lines", line=dict(color="#00E396", width=2), name="95th percentile"))
        fig.update_layout(template="plotly_dark", height=360)
        return fig

    @staticmethod
    def drawdown_recovery_distribution(tb: pd.DataFrame) -> Tuple[go.Figure, dict]:
        # Calculate time to recovery for each drawdown
        # Use gross cumulative PnL for drawdown analysis
        cum_pnl_column = 'cum_gross_pnl' if 'cum_gross_pnl' in tb.columns else 'cum_pnl'
        running_max = tb[cum_pnl_column].cummax()
        drawdown = running_max - tb[cum_pnl_column]
        
        recovery_times_trades = []
        recovery_times_hours = []
        recovery_times_days = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdown):
            if dd > 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                recovery_trades = i - start_idx
                recovery_times_trades.append(recovery_trades)
                
                # Calculate time-based recovery
                if i < len(tb) and start_idx < len(tb):
                    start_time = tb["Close_time"].iloc[start_idx]
                    end_time = tb["Close_time"].iloc[i]
                    time_diff = end_time - start_time
                    recovery_hours = time_diff.total_seconds() / 3600
                    recovery_days = recovery_hours / 24
                    recovery_times_hours.append(recovery_hours)
                    recovery_times_days.append(recovery_days)
        
        if not recovery_times_trades:
            return go.Figure(), {}
        
        # Create histogram
        fig = px.histogram(x=recovery_times_trades, nbins=min(20, len(set(recovery_times_trades))), template="plotly_dark")
        fig.update_layout(height=320, xaxis_title="Recovery Time (trades)", yaxis_title="Frequency")
        
        # Calculate time-based insights
        insights = {}
        if recovery_times_hours:
            insights = {
                "p75_recovery_hours": np.percentile(recovery_times_hours, 75),
                "max_recovery_hours": np.max(recovery_times_hours),
                "total_recoveries": len(recovery_times_trades)
            }
        
        return fig, insights

    @staticmethod
    def kelly_growth_rate_plot(tb: pd.DataFrame) -> go.Figure:
        """
        Create Kelly Criterion growth rate plot showing optimal wagered fraction.
        
        Args:
            tb: Trade book DataFrame
            
        Returns:
            Plotly figure showing growth rate vs wagered fraction
        """
        # Calculate Kelly parameters from trade data
        # Use gross PnL for Kelly analysis
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb.columns else 'pnl'
        wins = tb[tb[pnl_column] > 0][pnl_column]
        losses = tb[tb[pnl_column] <= 0][pnl_column]
        
        if len(tb) == 0 or len(wins) == 0 or len(losses) == 0:
            return go.Figure()
        
        # Kelly parameters
        p = len(wins) / len(tb)  # Probability of winning
        q = 1 - p  # Probability of losing
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        b = avg_win / avg_loss  # Win/loss ratio
        
        # Calculate optimal Kelly fraction
        kelly_optimal = p - q / b if b > 0 else 0
        
        # Generate wagered fractions from 0 to 2*Kelly (to show over-betting)
        f_values = np.linspace(0, min(1.0, 2 * kelly_optimal), 100)
        
        # Calculate growth rate for each wagered fraction
        growth_rates = []
        for f in f_values:
            # Growth rate formula: r = p*log(1 + b*f) + q*log(1 - f)
            if f < 1:  # Can't bet more than 100%
                growth_rate = p * np.log(1 + b * f) + q * np.log(1 - f)
                growth_rates.append(growth_rate * 100)  # Convert to percentage
            else:
                growth_rates.append(float('-inf'))
        
        # Create the plot
        fig = go.Figure()
        
        # Add growth rate curve
        fig.add_trace(go.Scatter(
            x=f_values * 100,  # Convert to percentage
            y=growth_rates,
            mode='lines',
            name='Growth Rate',
            line=dict(color='#007aff', width=3),
            hovertemplate='<b>Wagered:</b> %{x:.1f}%<br><b>Growth Rate:</b> %{y:.2f}%<extra></extra>'
        ))
        
        # Add optimal Kelly point
        if kelly_optimal > 0:
            optimal_growth = p * np.log(1 + b * kelly_optimal) + q * np.log(1 - kelly_optimal)
            fig.add_trace(go.Scatter(
                x=[kelly_optimal * 100],
                y=[optimal_growth * 100],
                mode='markers',
                name='Optimal Kelly',
                marker=dict(
                    symbol='diamond',
                    size=12,
                    color='black',
                    line=dict(color='white', width=2)
                ),
                hovertemplate='<b>Optimal Kelly:</b> %{x:.1f}%<br><b>Max Growth:</b> %{y:.2f}%<extra></extra>'
            ))
        
        # Add reference lines as traces for legend
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode='lines',
            line=dict(dash="dash", color="rgba(255,255,255,0.5)"),
            name='Break-even',
            showlegend=True
        ))
        
        # Add background regions as traces for legend
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode='markers',
            marker=dict(color="rgba(255,59,48,0.1)", size=10),
            name='Negative Growth',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode='markers',
            marker=dict(color="rgba(52,199,89,0.1)", size=10),
            name='Positive Growth',
            showlegend=True
        ))
        
        # Add parameter traces for legend
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode='markers',
            marker=dict(color="white", size=0),
            name=f'p = {p:.2f}',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode='markers',
            marker=dict(color="white", size=0),
            name=f'q = {q:.2f}',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode='markers',
            marker=dict(color="white", size=0),
            name=f'b = {b:.2f}',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode='markers',
            marker=dict(color="white", size=0),
            name=f'Kelly = {kelly_optimal:.2f}',
            showlegend=True
        ))
        
        # Add actual reference line (invisible in legend)
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.5)", 
                     showlegend=False)
        
        # Add background regions (invisible in legend)
        fig.add_hrect(y0=-10, y1=0, fillcolor="rgba(255,59,48,0.1)", 
                     layer="below", line_width=0, showlegend=False)
        fig.add_hrect(y0=0, y1=10, fillcolor="rgba(52,199,89,0.1)", 
                     layer="below", line_width=0, showlegend=False)
        
        # Update layout
        fig.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Wagered Fraction (%)",
            yaxis_title="Growth Rate (%)",
            showlegend=True,
            legend=dict(
                x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)', borderwidth=1,
                font=dict(color='#ffffff', size=10)
            ),
            margin=dict(l=50, r=50, t=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                showgrid=True,
                range=[0, min(100, kelly_optimal * 200)]
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                showgrid=True
            )
        )
        

        
        return fig
