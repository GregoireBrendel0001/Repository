import pandas as pd
import ta
import ta.trend

def compute_average_true_range(data : pd.DataFrame, window : int = 14) -> pd.Series:
    high_low = data["high"] - data["low"]
    high_close = (data["high"] - data["close"].shift()).abs()
    low_close = (data["low"] - data["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()

def compute_rolling_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    atr = compute_average_true_range(data, window)
    return (atr - atr.shift(window)) / atr.shift(window)

def compute_vortex_spread(data: pd.DataFrame, window=14) -> pd.Series:
    vortex_positive = ta.trend.vortex_indicator_pos(data['high'], data['low'], data['close'], window=window)
    vortex_negative = ta.trend.vortex_indicator_neg(data['high'], data['low'], data['close'], window=window)
    return (vortex_positive - vortex_negative) / ((vortex_positive + vortex_negative) / 2)

def compute_rolling_volume(data: pd.DataFrame, window: int = 14) -> pd.Series:
    return (data['volume'] - data['volume'].shift(window)) / data['volume'].shift(window)

def compute_moving_average_convergence_divergence(data : pd.DataFrame, window_short : int = 12, window_long : int = 26, signal_window : int = 9, price_column : str = "close") -> pd.Series:
    short_ema = ta.trend.ema_indicator(data[price_column], window_short)
    long_ema = ta.trend.ema_indicator(data[price_column], window_long)
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd - signal

def compute_bollinger_bands(data : pd.DataFrame, window : int = 20, price_column : str = "close") -> pd.DataFrame:
    sma = ta.trend.sma_indicator(data[price_column], window, price_column)
    rolling_std = data[price_column].rolling(window=window).std()
    upper_band = sma + (rolling_std * 2)
    lower_band = sma - (rolling_std * 2)
    return pd.DataFrame({
        "sma": sma,
        "upper_band": upper_band,
        "lower_band": lower_band
    })

def compute_stochastic_oscillator(data : pd.DataFrame, window : int = 14, price_column : str = "close") -> pd.Series:
    lowest_low = data[price_column].rolling(window=window).min()
    highest_high = data[price_column].rolling(window=window).max()
    return 100 * ((data[price_column] - lowest_low) / (highest_high - lowest_low))

