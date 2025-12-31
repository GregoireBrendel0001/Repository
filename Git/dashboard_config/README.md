# Strategy Configuration Dashboard

A Streamlit-based dashboard for configuring trading strategies based on the backtest system structure.

## Features

- **Multi-Pair Support**: Configure multiple trading pairs within a single strategy configuration
- **Interactive Configuration Form**: Configure all strategy parameters through a user-friendly interface
- **Multiple Indicators**: Support for 12+ technical indicators including RSI, SMA, EMA, MACD, Bollinger Bands, and more
- **Dynamic Indicator Configuration**: Configure indicator parameters based on the selected indicator type
- **Real-time Preview**: See your configuration as JSON before exporting
- **Validation**: Built-in validation for strategy parameters across all pairs
- **Export**: Download your configuration as a JSON file ready for backtesting
- **Pairs Management**: Add, edit, and delete trading pairs with individual configurations

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run app.py
```

## Usage

1. **Strategy Configuration**: Fill out the form with your strategy parameters
2. **Preview & Export**: Review the generated JSON and download it
3. Use the downloaded JSON file in your backtest system

## Configuration Structure

The dashboard generates JSON configurations that match your backtest system structure, including:

- **Global Strategy Settings**: Class, module, start/end dates applied to all pairs
- **Multi-Pair Support**: Each pair can have individual configurations within the same strategy
- **Trading Pair Settings**: Individual pair symbols, currencies, and allocation weights
- **Risk Management**: Per-pair stop loss and take profit configurations (T1, T2, T3 levels)
- **Technical Indicators**: Multiple indicators per pair with customizable parameters
- **Time Period Settings**: Data aggregation periods for main strategy and stop-loss monitoring

## Supported Indicators

The dashboard supports 12+ technical indicators from your `INDICATOR_FUNCTIONS` dictionary:

- **Trend Indicators**: SMA, EMA, MACD, Bollinger Bands
- **Momentum Indicators**: RSI, Stochastic Oscillator
- **Volatility Indicators**: ATR, Rolling ATR, Vortex Spread
- **Volume Indicators**: Rolling Volume
- **Custom Indicators**: Vortex Positive/Negative

## Example

The dashboard generates configurations compatible with your `strategies.json` format and `MeanReversingStrategy.py` implementation.
