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
import logging
from dotenv import dotenv_values
import pymongo
import os

logger = logging.getLogger(__name__)


class Display:
    """Encapsulates all plotting/analysis functions for the dashboard."""

    def __init__(self):

        config = dotenv_values(os.path.join(os.path.dirname(__file__), '.env'))
        logger.info("Configuration loaded from .env file.")

        # MongoDB connection
        self.mongo_client = pymongo.MongoClient(config["MONGO_URI"], int(config["MONGO_PORT"]), username=config["MONGO_USER"], password=config["MONGO_PASSWORD"])
        self.db = self.mongo_client.get_database(config["MONGO_DB"])
        logger.info("MongoDB client initialized with URI and port.")

    def load_book_of_trade(self, id: str) -> pd.DataFrame:
        """Fetch a trade book document from MongoDB and return it as a DataFrame."""
        try:
            doc = self.db["BookOfTrade"].find_one({"_id": id})
            if not doc:
                logger.warning(f"No BookOfTrade found for id {id}")
                return pd.DataFrame()

            data_list = doc.get("data", [])
            if not data_list:
                logger.warning(f"BookOfTrade {id} contains no data.")
                return pd.DataFrame()

            df = pd.DataFrame(data_list)
            if df.empty:
                logger.warning(f"BookOfTrade {id} created an empty DataFrame.")
                return df

            # Ensure Close_time/Open_time columns exist and are datetime
            for col in ["Open_time", "Close_time"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            # If Close_time is missing but index contains datetime, move it to column
            if "Close_time" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "Close_time"})

            return df

        except Exception as e:
            logger.error(f"Error loading BookOfTrade {id}: {e}", exc_info=True)
            return pd.DataFrame()

    def load_market_data(self, config: dict) -> Tuple[pd.DataFrame, dict]:
        """
        Load and preprocess market OHLCV data from MongoDB for a given symbol.
        """
        strat_datasets = {}
        try:
            pair = config["pair_name"]
            #depth = config.get("required_data_depth", 2000)

            cursor = (
                self.db["olhcv"]
                .find({"symbol": pair})
                .sort("timestamp", pymongo.DESCENDING)
                #.limit(depth)
            )
            olhcv_list = list(cursor)
            if not olhcv_list:
                logger.warning(f"No OHLCV data found for {pair}")
                return pd.DataFrame(), strat_datasets

            df = pd.DataFrame(olhcv_list)

            # Ensure timestamp conversion
            if "timestamp" not in df.columns:
                logger.error(f"Missing 'timestamp' in OHLCV data for {pair}")
                return pd.DataFrame(), strat_datasets

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Ensure numeric conversion
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop invalid rows
            df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

            # Sort ascending
            df = df.sort_values("timestamp")

            # Optional resampling
            period = config.get("required_data_config", {}).get("data_aggregation_period")
            if period:
                df = self.resample_dataset(df.set_index("timestamp"), period).reset_index()
                df.rename(columns={"index": "timestamp"}, inplace=True)

            # Ensure timestamp available as column (not just index)
            if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "timestamp"})

            # Truncate dataset if too large
            max_size = config.get("max_dataset_size", 10000)
            if len(df) > max_size:
                df = df.tail(max_size)
                logger.info(f"Truncated {pair} dataset to {max_size} rows.")

            strat_datasets[pair] = df
            return df, strat_datasets

        except Exception as e:
            logger.error(f"Error loading market data: {e}", exc_info=True)
            return pd.DataFrame(), strat_datasets

    def resample_dataset(self, dataset: pd.DataFrame, period: str) -> pd.DataFrame:
        """Resample dataset to a given time period with standard OHLCV aggregation."""
        try:
            resampled = (
                dataset.resample(period)
                .agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                })
                .dropna()
            )
            return resampled
        except Exception as e:
            logger.error(f"Error resampling dataset: {e}", exc_info=True)
            return dataset

    def load_and_preprocess(self, trade_book_id: str, config: dict):
        """
        Load trade book and market data from MongoDB, preprocess them, 
        and return clean, feature-rich DataFrames ready for analysis.
        """

        # --- 1. Load trade book ---
        trade_book_df = self.load_book_of_trade(trade_book_id)
        if trade_book_df.empty:
            logger.warning(f"No trade book data found for id {trade_book_id}")
            return pd.DataFrame(), pd.DataFrame()

        # --- 2. Load market data ---
        olhcv_df, strat_datasets = self.load_market_data(config)
        market_data_df = strat_datasets.get(config["pair_name"], pd.DataFrame())

        # --- 3. Preprocessing ---
        tb = trade_book_df.copy()
        md = market_data_df.copy()

        # Remove duplicated columns that can appear after resetting indexes / resampling
        tb = tb.loc[:, ~tb.columns.duplicated()]
        md = md.loc[:, ~md.columns.duplicated()]
        # Drop leftover generic "index" column
        if "index" in tb.columns:
            tb = tb.drop(columns=["index"])
        if "index" in md.columns:
            md = md.drop(columns=["index"])

        # Ensure Close_time exists as column
        if "Close_time" not in tb.columns and isinstance(tb.index, pd.DatetimeIndex):
            tb = tb.reset_index().rename(columns={"index": "Close_time"})

        # ---- Handle timestamps ----
        for col in ["Open_time", "Close_time"]:
            if col in tb.columns:
                tb[col] = pd.to_datetime(tb[col], errors="coerce", unit="ms" if np.issubdtype(tb[col].dtype, np.number) else None)

        if not md.empty:
            if "timestamp" in md.columns:
                md["timestamp"] = pd.to_datetime(md["timestamp"], errors="coerce", unit="ms" if np.issubdtype(md["timestamp"].dtype, np.number) else None)
                md = md.sort_values("timestamp").set_index("timestamp")

        # ---- Trade metrics ----
        tb = tb.sort_values("Close_time", ignore_index=True)
        tb["pnl"] = tb.get("Profit & Loss", np.nan)
        tb["size_usdt"] = tb.get("USDT value open", np.nan)

        tb["is_win"] = (
            (tb["Positive"] == 1).astype(int)
            if "Positive" in tb.columns
            else (tb["pnl"] > 0).astype(int)
        )


        # Calculate Duration as time difference between Open_time and Close_time
        if "Open_time" in tb.columns and "Close_time" in tb.columns:
            # Columns are already datetime objects after conversion above (line 226)
            # Calculate duration as timedelta: Close_time - Open_time
            tb["Duration"] = tb["Close_time"] - tb["Open_time"]
            # Handle any NaT values from the subtraction
            tb["duration_sec"] = tb["Duration"].dt.total_seconds().fillna(0)
            tb["duration_min"] = tb["duration_sec"] / 60.0
        else:
            tb["Duration"] = pd.Timedelta(0)
            tb["duration_sec"] = 0.0
            tb["duration_min"] = 0.0
 

        tb["return_frac"] = np.where(tb["size_usdt"] > 0, tb["pnl"] / tb["size_usdt"], np.nan)
        tb["profit_per_minute"] = np.where(
            (tb["duration_min"] > 0) & (tb["size_usdt"] > 0),
            (tb["pnl"] / tb["size_usdt"]) / tb["duration_min"],
            np.nan,
        )

        tb["hour"] = tb["Open_time"].dt.hour
        tb["dow"] = tb["Open_time"].dt.day_name()
        tb["month"] = tb["Open_time"].dt.month_name()

        # ---- Equity curve ----
        tb["cum_pnl"] = tb["pnl"].cumsum()

        # ---- Fees and gross pnl ----
        tb["trade_fee"] = (
            tb.get("Fee", tb.get("Commission", tb.get("Fees", tb["size_usdt"] * 0.001)))
        )
        tb["gross_pnl"] = tb["pnl"] + tb["trade_fee"]
        tb["cum_gross_pnl"] = tb["gross_pnl"].cumsum()

        # ---- Drawdown ----
        if not tb.empty:
            initial_investment = tb["size_usdt"].iloc[0]
            tb["total_value"] = tb["cum_pnl"] + initial_investment
            tb["running_max_value"] = tb["total_value"].cummax()
            tb["drawdown"] = np.where(
                tb["running_max_value"] > tb["total_value"],
                (tb["running_max_value"] - tb["total_value"]) / tb["running_max_value"] * 100,
                0,
            )

        # ---- Market features ----
        if not md.empty and "close" in md.columns:
            # Convert numeric fields
            for col in ["open", "high", "low", "close", "volume"]:
                if col in md.columns:
                    md[col] = pd.to_numeric(md[col], errors="coerce")

            md.dropna(subset=["close"], inplace=True)

            # Compute volatility and trend
            md["ret"] = md["close"].pct_change()
            bars_vol_window = 30
            md["roll_vol"] = md["ret"].rolling(bars_vol_window, min_periods=5).std()
            md["roll_trend"] = md["close"].pct_change(bars_vol_window)

            # ✅ Drop NaT in Open_time before merge
            tb = tb.dropna(subset=["Open_time"]).copy()

            # ✅ Sort both sides
            tb = tb.sort_values("Open_time")
            md = md.sort_index()

            # ✅ Safe merge_asof (align trade open times with recent market data)
            try:
                aligned = pd.merge_asof(
                    tb,
                    md[["roll_vol", "roll_trend"]],
                    left_on="Open_time",
                    right_index=True,
                    direction="backward"
                )
                tb["entry_vol"] = aligned["roll_vol"].values
                tb["entry_trend"] = aligned["roll_trend"].values
            except ValueError as e:
                logger.warning(f"Market feature merge failed: {e}")
                tb["entry_vol"] = np.nan
                tb["entry_trend"] = np.nan
        else:
            tb["entry_vol"] = np.nan
            tb["entry_trend"] = np.nan

        # Ensure timestamps are available as regular columns
        if isinstance(md.index, pd.DatetimeIndex):
            md = md.reset_index().rename(columns={"index": "timestamp"})
        elif "timestamp" not in md.columns:
            md["timestamp"] = pd.NaT

        # ---- Rolling win rate ----
        tb["rolling_win_rate"] = tb["is_win"].rolling(30, min_periods=5).mean() * 100

        # Calculate rolling Kelly fraction
        tb["rolling_kelly_fraction"] = self.calculate_rolling_kelly(tb["pnl"], 10)
        
        # Calculate suggested position size based on Kelly and current account value
        # Use a conservative approach: Kelly * 0.25 (quarter Kelly) for safety
        tb["kelly_position_size"] = np.where(
            tb["rolling_kelly_fraction"].notna() & (tb["rolling_kelly_fraction"] > 0),
            tb["total_value"] * tb["rolling_kelly_fraction"] * 0.25,  # Quarter Kelly
            tb["size_usdt"]  # Fallback to actual size if no Kelly signal
        )

        logger.info(f"Preprocessing complete for trade_book_id={trade_book_id}")

        return tb, md
    
    @staticmethod
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
    def equity_curve_with_backtest(tb_real: pd.DataFrame, md: pd.DataFrame, date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None) -> go.Figure:
        """Create equity curve chart with both real-time and backtest curves, plus market price."""
        tb_real_plot = tb_real.copy()  
        md_plot = md.copy()

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
            logger.debug("cum_gross_pnl column not found, calculating inline...")
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
            # Check for common price column names
            price_column = None
            if "close" in md_plot.columns:
                price_column = "close"
            
            if price_column:
                # Align market data with trade timeline
                start_time = tb_real_plot["Close_time"].min()
                end_time = tb_real_plot["Close_time"].max()
                
                md_filtered = md_plot[(md_plot["timestamp"] >= start_time) & (md_plot["timestamp"] <= end_time)]
                
                if not md_filtered.empty:
                    # Ensure timestamps are properly formatted and aligned
                    md_filtered = md_filtered.copy()
                    md_filtered["formatted_timestamp"] = pd.to_datetime(md_filtered["timestamp"])
                    
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
                    logger.debug("No market data in trade timeline range")
            else:
                logger.debug("No price column found in market data")
        else:
            logger.debug("Market data is empty")

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

        # Filter out NaN values for bucketing
        valid_vol = tb_copy['volatility'].dropna()
        
        if len(valid_vol) == 0:
            # No valid volatility data
            tb_copy['vol_bucket'] = 'No Data'
            vol_labels = {'No Data': 'No Data'}
        else:
            # Use percentile-based bucketing for better distribution
            # This ensures we always get a good spread across buckets
            vol_values = valid_vol.values
            
            # Filter out zero values for percentile calculation
            non_zero_vol = vol_values[vol_values > 0]
            
            if len(non_zero_vol) > 0:
                # Use quantiles to ensure even distribution
                q20, q40, q60, q80, q90 = np.percentile(non_zero_vol, [20, 40, 60, 80, 90])
                
                def assign_vol_bucket(vol):
                    if pd.isna(vol):
                        return 'Neutral'
                    elif vol == 0:
                        return 'Neutral'
                    elif vol <= q20:
                        return 'Very Low'
                    elif vol <= q40:
                        return 'Low'
                    elif vol <= q60:
                        return 'Medium'
                    elif vol <= q80:
                        return 'High'
                    elif vol <= q90:
                        return 'Very High'
                    else:
                        return 'Extreme'
            else:
                # All values are zero
                def assign_vol_bucket(vol):
                    if pd.isna(vol) or vol == 0:
                        return 'Neutral'
                    else:
                        return 'Very Low'
            
            tb_copy['vol_bucket'] = tb_copy['volatility'].apply(assign_vol_bucket)
            vol_labels = {
                'Neutral': 'Neutral',
                'Very Low': 'Very Low',
                'Low': 'Low',
                'Medium': 'Medium',
                'High': 'High',
                'Very High': 'Very High',
                'Extreme': 'Extreme',
                'No Data': 'No Data'
            }

        # Calculate positive and negative PnL by volatility bucket
        # Use gross PnL for volatility analysis
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb_copy.columns else 'pnl'
        grouped_positive = tb_copy[tb_copy[pnl_column] > 0].groupby('vol_bucket')[pnl_column].sum().reset_index()
        grouped_negative = tb_copy[tb_copy[pnl_column] <= 0].groupby('vol_bucket')[pnl_column].sum().reset_index()
        
        # Apply the labels with boundaries
        grouped_positive['vol_bucket_labeled'] = grouped_positive['vol_bucket'].map(vol_labels)
        grouped_negative['vol_bucket_labeled'] = grouped_negative['vol_bucket'].map(vol_labels)

        # Helper function to get volatility range for a bucket
        def get_vol_range(bucket):
            if bucket in tb_copy['vol_bucket'].values:
                bucket_data = tb_copy[tb_copy['vol_bucket'] == bucket]
                min_vol = bucket_data['volatility'].min()
                max_vol = bucket_data['volatility'].max()
                return f"{min_vol:.4f}-{max_vol:.4f}"
            else:
                return "N/A"
        
        # Add actual volatility values to labels for positive group
        grouped_positive['vol_range'] = grouped_positive['vol_bucket'].apply(get_vol_range)
        grouped_positive['x_label'] = grouped_positive['vol_bucket_labeled'].astype(str) + '<br>(' + grouped_positive['vol_range'] + ')'
        
        # Add actual volatility values to labels for negative group
        grouped_negative['vol_range'] = grouped_negative['vol_bucket'].apply(get_vol_range)
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
    def profit_vs_trend(tb: pd.DataFrame) -> go.Figure:
        # TREND COMPUTATION EXPLANATION:
        # This is PRE-TRADE trend, computed at the time of trade opening
        # For each trade, we look at the market trend at the moment the trade was opened
        # This helps understand the market direction when you decided to enter the trade
        # This is more useful for strategy adjustment as it shows entry conditions
        
        tb_copy = tb.copy()
        
        # Filter out NaN values for bucketing
        valid_trend = tb_copy['entry_trend'].dropna()
        
        if len(valid_trend) == 0:
            # No valid trend data
            tb_copy['trend_bucket'] = 'Neutral'
            trend_labels = {'Neutral': 'Neutral'}
        else:
            # Use percentile-based bucketing for better distribution
            # This ensures we always get a good spread across buckets
            trend_values = valid_trend.values
            min_trend = trend_values.min()
            max_trend = trend_values.max()
            
            # Use quantiles to ensure even distribution
            q20, q40, q60, q80, q90 = np.percentile(trend_values, [20, 40, 60, 80, 90])
            
            def assign_trend_bucket(trend):
                if pd.isna(trend):
                    return 'Neutral'
                elif trend == 0:
                    return 'Neutral'
                elif trend <= q20:
                    return 'Very Low'
                elif trend <= q40:
                    return 'Low'
                elif trend <= q60:
                    return 'Medium'
                elif trend <= q80:
                    return 'High'
                elif trend <= q90:
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


        # Calculate positive and negative PnL by trend bucket
        # Use gross PnL for trend analysis
        pnl_column = 'gross_pnl' if 'gross_pnl' in tb_copy.columns else 'pnl'
        grouped_positive = tb_copy[tb_copy[pnl_column] > 0].groupby('trend_bucket')[pnl_column].sum().reset_index()
        grouped_negative = tb_copy[tb_copy[pnl_column] <= 0].groupby('trend_bucket')[pnl_column].sum().reset_index()
        
        # Apply the labels with boundaries
        grouped_positive['trend_bucket_labeled'] = grouped_positive['trend_bucket'].map(trend_labels)
        grouped_negative['trend_bucket_labeled'] = grouped_negative['trend_bucket'].map(trend_labels)
        
        # Helper function to get trend range for a bucket
        def get_trend_range(bucket):
            if bucket in tb_copy['trend_bucket'].values:
                bucket_data = tb_copy[tb_copy['trend_bucket'] == bucket]
                min_trend = bucket_data['entry_trend'].min()
                max_trend = bucket_data['entry_trend'].max()
                
                # Add direction indicators
                if bucket == 'Neutral':
                    return "Neutral"
                elif min_trend >= 0 and max_trend >= 0:
                    return f"↗ {min_trend:.3f}-{max_trend:.3f}"
                elif min_trend <= 0 and max_trend <= 0:
                    return f"↘ {min_trend:.3f}-{max_trend:.3f}"
                else:
                    return f"↔ {min_trend:.3f}-{max_trend:.3f}"
            else:
                return "N/A"
        
        # Add actual trend values to labels for positive group
        grouped_positive['trend_range'] = grouped_positive['trend_bucket'].apply(get_trend_range)
        grouped_positive['x_label'] = grouped_positive['trend_bucket_labeled'].astype(str) + '<br>(' + grouped_positive['trend_range'] + ')'
        
        # Add actual trend values to labels for negative group
        grouped_negative['trend_range'] = grouped_negative['trend_bucket'].apply(get_trend_range)
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
