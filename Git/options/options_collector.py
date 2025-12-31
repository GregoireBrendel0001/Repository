import os
from datetime import datetime, timedelta
from typing import List
import pytz
import json

import pymongo
from dotenv import dotenv_values
from binance.client import Client

# ============================================================
# Utility functions (reused / efficient)
# ============================================================

def filter_options_around_spot(option_symbols, spot_price, band=0.20):
    lower = spot_price * (1 - band)
    upper = spot_price * (1 + band)
    return [
        opt for opt in option_symbols
        if lower <= float(opt["strikePrice"]) <= upper
    ]

def get_available_expiries(option_symbols, underlying):
    return sorted({
        opt["expiryDate"]
        for opt in option_symbols
        if opt["underlying"] == underlying
    })

def get_strikes_for_expiry(option_symbols, underlying, expiry):
    return sorted({
        float(opt["strikePrice"])
        for opt in option_symbols
        if opt["underlying"] == underlying
        and opt["expiryDate"] == expiry
    })

# ============================================================
# Core Class: Options Data Extractor
# ============================================================

class OptionsDataExtractor:
    """
    Binance Options data extractor.
    - Order book: every minute
    - Historical prices: daily (D-1 at 00:05)
    """

    def __init__(self, config_path="config.json"):
        # ---- Load config
        config = dotenv_values(os.path.join(os.path.dirname(__file__), ".env"))
        if not config:
            raise RuntimeError("Unable to load .env file")

        with open("config.json", "r") as f:
            self.dictionnary_config = self.load_config(config_path)

        # ---- Binance client
        self.client = Client(
            api_key=config["API_KEY"],
            api_secret=config["SECRET_KEY"]
        )

        # ---- Exchange info
        self.exchange_info = self.client.options_exchange_info()
        self.option_symbols = self.exchange_info["optionSymbols"]

        # ---- MongoDB
        mongo_client = pymongo.MongoClient(
            host=config["MONGO_URI"],
            port=int(config.get("MONGO_PORT", 27017)),
            username=config.get("MONGO_USER"),
            password=config.get("MONGO_PASSWORD"),
        )
        self.db = mongo_client["quants_db_develop"]
        self.options_col = self.db["options"]  # Single collection for all tickers
        self.db["Options"]

    def load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            return json.load(f)
        
    # --------------------------------------------------------
    def is_symbol_available(self, symbol: str) -> bool:
        return any(opt["symbol"] == symbol for opt in self.option_symbols)

    # --------------------------------------------------------
    def build_option_universe(
        self,
        underlying: str,
        expiry: int,
        spot_price: float,
        band: float,
        strike_step: float,
        side: str = None
    ) -> List[str]:
        

        strikes = get_strikes_for_expiry(self.option_symbols, underlying, expiry)
        lower = spot_price * (1 - band)
        upper = spot_price * (1 + band)
        valid_strikes = {s for s in strikes if lower <= s <= upper and abs(s / strike_step - round(s / strike_step)) < 1e-6}

        return [
            opt["symbol"]
            for opt in self.option_symbols
            if opt["underlying"] == underlying
            and opt["expiryDate"] == expiry
            and float(opt["strikePrice"]) in valid_strikes
            and (side is None or opt["side"] == side)
        ]

    # --------------------------------------------------------

    def collect_order_book(self, symbol: str, limit: int = 50):
        if not self.is_symbol_available(symbol):
            return

        book = self.client.options_order_book(symbol=symbol, limit=limit)

        ts = datetime.utcnow().strftime("%Y-%m-%dT%H_%M_%SZ")

        snapshot = {
            "bids": book["bids"],
            "asks": book["asks"]
        }

        self.db["options"].update_one(
            {"symbol": symbol},
            {
                "$setOnInsert": {
                    "symbol": symbol,
                    "created_at": datetime.utcnow()
                },
                "$set": {
                    f"order_book.{ts}": snapshot
                }
            },
            upsert=True
        )
    # --------------------------------------------------------

    def collect_historical_price(self, symbol: str):
        if not self.is_symbol_available(symbol):
            return

        klines = self.client.options_klines(
            symbol=symbol,
            interval="1d",
            limit=1
        )

        if not klines:
            return

        k = klines[0]

        ts = datetime.utcfromtimestamp(k[0] / 1000).strftime("%Y-%m-%dT00_00_00Z")

        ohlc = {
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5])
        }

        self.db["options"].update_one(
            {"symbol": symbol},
            {
                "$setOnInsert": {
                    "symbol": symbol,
                    "created_at": datetime.utcnow()
                },
                "$set": {
                    f"historical_price.{ts}": ohlc
                }
            },
            upsert=True
        )

# ============================================================
# MAIN
# ============================================================

    def main(self):

        # ---- Underlyings to process
        underlyings = self.dictionnary_config.keys()

        for underlying in underlyings:

            # --- Spot price
            spot_price = float(
                self.client.options_index_price(underlying=underlying)["indexPrice"]
            )
            print(f"\nSpot price {underlying}: {spot_price}")

            # --- All available expiries
            expiries = get_available_expiries(self.option_symbols, underlying)
            if not expiries:
                print(f"No expiries found for {underlying}")
                continue

            print(f"Found {len(expiries)} expiries for {underlying}")

            # ======================================================
            # Loop over ALL expiries
            # ======================================================
            for expiry in expiries:

                expiry_date = datetime.utcfromtimestamp(expiry / 1000).date()
                print(f"\nProcessing expiry: {expiry_date}")

                # --- Build option universe for this expiry
                symbols = self.build_option_universe(
                    underlying=underlying,
                    expiry=expiry,
                    spot_price=spot_price,
                    band=self.dictionnary_config[underlying]['band'],
                    strike_step=self.dictionnary_config[underlying]['step']  # adjust if needed
                )

                print(
                    f"Selected {len(symbols)} option symbols "
                    f"for {underlying} | expiry {expiry_date}"
                )

                if not symbols:
                    print("No symbols found for this expiry — skipping.")
                    continue

                # --------------------------------------------------
                # Minute-level: order books
                # --------------------------------------------------
                print("Collecting order book snapshots...")
                for symbol in symbols:
                    try:
                        self.collect_order_book(symbol)
                    except Exception as e:
                        print(f"[ERROR] Order book failed for {symbol}: {e}")

                # --------------------------------------------------
                # Scheduled job: historical prices
                # --------------------------------------------------
                print("Collecting historical prices...")
                for symbol in symbols:
                    try:
                        self.collect_historical_price(symbol)
                    except Exception as e:
                        print(f"[ERROR] Historical prices failed for {symbol}: {e}")

        print("\nFull universe extraction completed successfully.")

