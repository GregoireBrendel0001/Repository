import ta
import ta.momentum
from .additionnal_indicators import *

import pandas as pd


INDICATOR_FUNCTIONS = {
    "sma" : lambda df, window : ta.trend.sma_indicator(df["close"], window=window),
    "ema" : lambda df, window : ta.trend.ema_indicator(df["close"], window=window),
    "rsi" :lambda df, window :  ta.momentum.rsi(df["close"], window=window),
    "atr" : compute_average_true_range,
    "vtxp" : lambda df, window : ta.trend.vortex_indicator_pos(df['high'], df['low'], df['close'], window=window),
    "vtxn" : lambda df, window : ta.trend.vortex_indicator_neg(df['high'], df['low'], df['close'], window=window),
    "roll_atr" : compute_rolling_atr,
    "vtxspread" : compute_vortex_spread,
    "roll_vol" : compute_rolling_volume,
    "macd" : lambda df, window_slow, window_fast :ta.trend.macd(df["close"], window_slow=window_slow, window_fast=window_fast),
    "bband" : compute_bollinger_bands,
    "stoch_osc" : compute_stochastic_oscillator
}

class IndicatorBuilder:
    def __init__(self, config):
        self.config = config
    
    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        # Validate input data
        if data is None or data.empty:
            raise ValueError("Input data is None or empty")
            
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        for indicator in self.config:
            # Validate indicator configuration
            if not isinstance(indicator, dict):
                raise ValueError(f"Invalid indicator configuration: {indicator}")
                
            if "name" not in indicator:
                raise ValueError(f"Indicator missing 'name' field: {indicator}")
                
            if "column_name" not in indicator:
                raise ValueError(f"Indicator missing 'column_name' field: {indicator}")
                
            if "params" not in indicator:
                raise ValueError(f"Indicator missing 'params' field: {indicator}")
                
            indicator_name = indicator["name"]
            if indicator_name not in INDICATOR_FUNCTIONS:
                raise ValueError(f"Indicator {indicator_name} is not recognized.")
                
            new_column_names = indicator["column_name"]
            indicator_params = indicator["params"]
            
            # Validate parameters based on indicator type
            if indicator_name == "sma" or indicator_name == "ema" or indicator_name == "rsi":
                if "window" not in indicator_params:
                    raise ValueError(f"Indicator {indicator_name} requires 'window' parameter")
                if indicator_params["window"] <= 0:
                    raise ValueError(f"Window parameter must be positive for {indicator_name}")
                    
            try:
                indicator_func = INDICATOR_FUNCTIONS[indicator_name]
                indicator_data = indicator_func(data, **indicator_params)
            except Exception as e:
                raise ValueError(f"Error computing indicator {indicator_name}: {e}")

            if isinstance(indicator_data, pd.DataFrame):
                indicator_data.columns = new_column_names
            elif isinstance(indicator_data, pd.Series):
                indicator_data.name = new_column_names[0]
            else:
                raise ValueError(f"Unexpected return type from indicator {indicator_name}")
                
            data = pd.concat([data, indicator_data], axis=1)
        return data