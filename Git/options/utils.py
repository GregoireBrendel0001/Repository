from collections import OrderedDict
from datetime import datetime
import re
from options_collector import OptionsDataExtractor




class OptionsUtils:

    def __init__(self):
        # ---- Exchange info
        self.extractor = OptionsDataExtractor()

    def market_depth_building(self, pair_name: str, timestamp):
        """Calculate market depth for a given option symbol and timestamp"""

        market_depth = {}

        # --- Normalize timestamp to match DB keys (ISO-like)
        if isinstance(timestamp, datetime):
            timestamp_str = timestamp.strftime("%Y-%m-%dT%H_%M_%SZ")
        else:
            timestamp_str = str(timestamp)

        # --- Fetch option document
        doc = self.extractor.db["options"].find_one({"symbol": pair_name})

        if not doc or "order_book" not in doc:
            return {
                "bids": {0.0: 0.0},
                "asks": {0.0: 0.0}
            }

        order_book_dict = doc["order_book"]

        # --- Find closest snapshot if exact timestamp not present
        if timestamp_str not in order_book_dict:
            available_ts = sorted(order_book_dict.keys())
            if not available_ts:
                return {
                    "bids": {0.0: 0.0},
                    "asks": {0.0: 0.0}
                }
            timestamp_str = available_ts[-1]  # fallback to latest

        snapshot = order_book_dict[timestamp_str]

        # --- Build cumulative depth
        for side in ["bids", "asks"]:
            if side not in snapshot:
                market_depth[side] = {0.0: 0.0}
                continue

            # Convert [[price, qty], ...] → {price: qty}
            side_data = {
                float(price): float(qty)
                for price, qty in snapshot[side]
            }

            # Sort
            if side == "bids":
                sorted_prices = OrderedDict(sorted(side_data.items(), reverse=True))
            else:
                sorted_prices = OrderedDict(sorted(side_data.items()))

            cumulative_qty = 0.0
            cumulative_depth = {}

            for price, qty in sorted_prices.items():
                cumulative_qty += qty
                cumulative_depth[price] = cumulative_qty

            market_depth[side] = cumulative_depth

        return market_depth

    def parse_option_symbol(self, symbol: str):
        """
        Parse Binance option symbol: BTC-251219-100000-C
        Returns underlying, expiry (datetime), strike, side
        """
        pattern = r"([A-Z]+)-(\d{6})-(\d+)-(C|P)"
        match = re.match(pattern, symbol)
        if not match:
            return None

        underlying, expiry_str, strike, side = match.groups()
        expiry_dt = datetime.strptime(expiry_str, "%y%m%d")
        return {
            "underlying": underlying,
            "expiry": expiry_dt,
            "strike": float(strike),
            "side": side
        }

    def find_nearest_option_db(self, underlying: str, given_strike: float, given_expiry: datetime, side: str):
        """
        Find nearest option in MongoDB matching side, returning distances and metadata.
        """
        docs = self.extractor.db["options"].find(
            {"symbol": {"$regex": f"^{underlying}-"}}
        )

        best_match = None
        best_score = float("inf")
        now = datetime.utcnow()

        for doc in docs:
            parsed = self.parse_option_symbol(doc["symbol"])
            if not parsed:
                continue

            # Ensure side matches
            if parsed["side"] != side:
                continue

            strike_diff = abs(parsed["strike"] - given_strike)
            expiry_diff_days = abs((parsed["expiry"] - given_expiry).days)

            # Weighted score: expiry difference dominates
            score = expiry_diff_days * 1_000_000 + strike_diff

            if score < best_score:
                best_score = score
                best_match = {
                    "symbol": doc["symbol"],
                    "given_strike": given_strike,
                    "given_expiry": given_expiry,
                    "selected_strike": parsed["strike"],
                    "selected_expiry": parsed["expiry"],
                    "strike_diff": strike_diff,
                    "expiry_diff_days": expiry_diff_days,
                    "days_to_target_expiry": (given_expiry - now).days,
                    "days_to_selected_expiry": (parsed["expiry"] - now).days,
                    "side": side
                }

        return best_match

