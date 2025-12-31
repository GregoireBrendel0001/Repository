import pandas as pd
import os
import traceback
from collections import defaultdict
import logging
import json
import random
import math
import numpy as np

from binance.client import Client
import pymongo
from dotenv import dotenv_values
from datetime import datetime, timedelta
from collections import OrderedDict

from utils.IndicatorBuilder import IndicatorBuilder
from MeanReversingStrategy import MeanReversingStrategy


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PROD_STRATEGIES = {
    "MeanReversingStrategy" : MeanReversingStrategy
}

class MultipleAssetBacktest:

    """
    Optimized backtest class combining the best features from both implementations:
    - Secure API configuration (from folder 2)
    - Flexible indicator calculation (from folder 1 with improvements)
    - Clean code organization (from folder 2)
    - Robust fee calculation (consistent across both)
    """

    def __init__(self, strategies_config_path=os.path.join(os.path.dirname(__file__), 'configs/strategies.json')):
        
        logger.info("Loading strategies from config file...")
        self.strategies_config_path = strategies_config_path
        self.strategies_config = {}
        self.strat_datasets = {}
        self.fiat_strategies_mapping = defaultdict(list)
        self.strat_indicator_builders = {}
        self.strategies = self.load_strategies()
        self.market_data_dictionnary= {}
        logger.info("Strategies loaded successfully.")
        
        config = dotenv_values(os.path.join(os.path.dirname(__file__), '.env'))
        logger.info("Configuration loaded from .env file.")

        # MongoDB connection
        self.mongo_client = pymongo.MongoClient(config["MONGO_URI"], int(config["MONGO_PORT"]), username=config["MONGO_USER"], password=config["MONGO_PASSWORD"])
        self.db = self.mongo_client.get_database(config["MONGO_DB"])
        logger.info("MongoDB client initialized with URI and port.")

        # Initialize variable
        self.account = {}
        self.position = {strategy : {} for strategy in self.strategies.keys()}
        self.borrowing_rate  = 0.01
        self.get_data_and_compute_indicators()
        self.index_list = self.get_timestamps(self.market_data_dictionnary)
        self.intialize_account()

    def load_strategies(self):

        with open(self.strategies_config_path, 'r') as file:
            self.strategies_config = json.load(file)
        
        strategies = {}
        for strategy_name, config in self.strategies_config.items():
            class_to_instanciate = config["class"]
            if class_to_instanciate in PROD_STRATEGIES:
                
                strat_config = config.copy()
                strat_config.pop("class", None)  # Remove class key if it exists

                self.strat_indicator_builders[strategy_name] = IndicatorBuilder(config.get("indicators", []))
                self.fiat_strategies_mapping[config["balance_currency"]].append(strategy_name)
                strategies[strategy_name] = config
                strategies[strategy_name] = PROD_STRATEGIES[class_to_instanciate](**strat_config)
            else:
                logger.warning(f"Warning: Strategy {class_to_instanciate} not found in PROD_STRATEGIES.")

        return strategies
        
    def get_data_and_compute_indicators(self) : 

        # How to manage multiple Pair Backtest

        for strategy, config in self.strategies_config.items():
            try:
                #get ohlcv data from mongodb for the pair

                olhcv_data = self.db["olhcv"].find({"symbol": config["pair_name"],
                                        "timestamp": {"$gte": config["start_date"], "$lte": config["end_date"]}}).sort("timestamp", pymongo.DESCENDING)
                olhcv_list = list(olhcv_data)
                
                if not olhcv_list:
                    logger.warning(f"Warning: No data found for strategy {strategy} with pair {config['pair_name']}")
                    continue
                    
                olhcv_df = pd.DataFrame(olhcv_list)
                
                # Validate required columns exist
                required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
                missing_columns = [col for col in required_columns if col not in olhcv_df.columns]
                if missing_columns:
                    logger.warning(f"Warning: Missing columns {missing_columns} for strategy {strategy}")
                    continue
                    
                olhcv_df = olhcv_df[required_columns]
                
                # Convert timestamp with error handling
                try:
                    olhcv_df["timestamp"] = pd.to_datetime(olhcv_df["timestamp"], format='%Y-%m-%d %H:%M')
                except Exception as e:
                    logger.error(f"Error converting timestamp for strategy {strategy}: {e}")
                    continue
                    
                olhcv_df.set_index("timestamp", inplace=True)
                olhcv_df.sort_index(inplace=True)
                
                # Convert numeric columns with error handling
                try:
                    olhcv_df = olhcv_df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
                except Exception as e:
                    logger.error(f"Error converting numeric columns for strategy {strategy}: {e}")
                    continue

                #resample the data to the required period
                if "data_aggregation_period" in config:
                    try:
                        olhcv_df = self.resample_dataset(olhcv_df, config["data_aggregation_period"])
                    except Exception as e:
                        logger.error(f"Error resampling data for strategy {strategy}: {e}")
                        continue

                #compute indicators
                try:
                    indicator_builder = self.strat_indicator_builders[strategy]
                    olhcv_df = indicator_builder.build(olhcv_df)
                except Exception as e:
                    logger.error(f"Error computing indicators for strategy {strategy}: {e}")
                    continue
                    
                self.strat_datasets[strategy] = olhcv_df
                nested_dict = self.df_to_nested_dict(olhcv_df, config["data_aggregation_period"])
                print(nested_dict)
                self.market_data_dictionnary[config['pair_name']] = nested_dict

                # Load stoploss data if different period
                stoploss_period = config["data_aggregation_period_stoploss"]
                if stoploss_period != config["data_aggregation_period"]:
                    # Resample for stoploss period
                    stoploss_df = olhcv_df.resample(stoploss_period).agg({
                        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
                    })
                    stoploss_nested_dict = self.df_to_nested_dict(stoploss_df, stoploss_period)
                    # Merge into existing data structure
                    self.market_data_dictionnary[config['pair_name']][stoploss_period] = stoploss_nested_dict[stoploss_period]
                
            except Exception as e:
                logger.error(f"Error processing data for strategy {strategy}: {e}")
                logger.error(traceback.format_exc())
                continue

    def resample_dataset(self, dataset, period):
        """
        Resample the dataset to the specified period.
        :param dataset: The dataset to resample.
        :param period: The period to resample to (e.g., '1T' for 1 minute).
        :return: Resampled dataset.
        """
        return dataset.resample(period).agg({"open": "max", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    
    def df_to_nested_dict(self, df, frequency):
        """
        Convert a DataFrame into a nested dictionary:
        dict[frequency][timestamp_str] = row_values
        All timestamps are converted to strings in "%Y-%m-%d %H:%M:%S" format.
        """
        nested_dict = {frequency: {}}

        for ts, row in df.iterrows():
            # Convert timestamp to string
            if hasattr(ts, "to_pydatetime"):
                ts_str = ts.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
            else:
                ts_str = str(ts)

            nested_dict[frequency][ts_str] = row.to_dict()

        return nested_dict

    def get_timestamps(self, market_data):
        """Get sorted unique timestamps from all pairs and their '1min' sub-dictionaries, as strings."""
        all_timestamps = set()

        for pair_data in market_data.values():
            one_min_data = pair_data.get('1min', {})
            for ts in one_min_data.keys():
                # Convert each timestamp to string if needed
                if isinstance(ts, str):
                    all_timestamps.add(ts)
                else:
                    all_timestamps.add(ts.strftime("%Y-%m-%d %H:%M:%S"))

        return sorted(all_timestamps)

    def intialize_account(self):

        wallet_balance = 100 # Should be replace in the future
        start = self.index_list[0]

        for strategy, config in self.strategies_config.items():

            """Update account state for current timestamp"""
            if config['pair_name'] not in self.account.keys():
                self.account[config['pair_name']] = {}
            if start not in self.account[config['pair_name']].keys():
                self.account[config['pair_name']][start] = {}

            self.account[config['pair_name']][start]['Fiat_balance'] = config['allocated_weight'] * wallet_balance
            self.account[config['pair_name']][start]['Asset_balance'] = 0
            self.account[config['pair_name']][start]['Account_balance'] = config['allocated_weight'] * wallet_balance
            self.account[config['pair_name']][start]['Maximum'] = config['allocated_weight'] * wallet_balance
            self.account[config['pair_name']][start]['Drawdown'] = config['allocated_weight'] * wallet_balance

    def update_account(self):

        """Update account state for current timestamp"""

        if self.current_timestamp not in self.account[self.current_config['pair_name']].keys():
            self.account[self.current_config['pair_name']][self.current_timestamp] = {}

        self.account[self.current_config['pair_name']][self.current_timestamp]['Fiat_balance'] = self.account[self.current_config['pair_name']][self.current_pre_timestamp]['Fiat_balance'] 
        self.account[self.current_config['pair_name']][self.current_timestamp]['Asset_balance'] = self.exposition()
        self.account[self.current_config['pair_name']][self.current_timestamp]['Account_balance'] = self.account[self.current_config['pair_name']][self.current_timestamp]['Fiat_balance'] + self.account[self.current_config['pair_name']][self.current_timestamp]['Asset_balance']
        self.account[self.current_config['pair_name']][self.current_timestamp]['Maximum'] = self.maximum()
        self.account[self.current_config['pair_name']][self.current_timestamp]['Drawdown'] = self.drawdown()

    def update_account_with_order(self, order_value, order_fee, side):
        """Update account balance after order execution"""

        balance = self.account[self.current_config['pair_name']][self.current_timestamp]['Fiat_balance']
        
        if side == "BUY":
            # Deduct order value and fees from fiat balance
            self.account[self.current_config['pair_name']][self.current_timestamp]['Fiat_balance'] = balance - order_value - order_fee
        elif side == "SELL":
            # Add order value minus fees to fiat balance
            self.account[self.current_config['pair_name']][self.current_timestamp]['Fiat_balance'] = balance + order_value - order_fee

    def exposition(self):
        """Calculate current exposition for a symbol"""
        if self.current_strategy in self.position and self.position[self.current_strategy]:
            exposition = 0
            for random_number, pos in self.position[self.current_strategy].items():
                exposition += pos.get('Pending', 0) 
            
            # Fix the market data access path
            frequency = self.current_config["data_aggregation_period_stoploss"]
            current_price = self.market_data_dictionnary[self.current_config['pair_name']][frequency][self.current_timestamp]['close']
            exposition *= current_price
        else:
            exposition = 0
        return exposition

    def maximum(self):
        """Track the highest account value reached"""
        current_balance = self.account[self.current_config['pair_name']][self.current_timestamp]['Account_balance']
        previous_maximum = self.account[self.current_config['pair_name']][self.current_pre_timestamp]['Maximum']
        
        if current_balance > previous_maximum:
            return current_balance
        else:
            return previous_maximum

    def drawdown(self):
        """Calculate drawdown for risk management"""
        current_balance = self.account[self.current_config['pair_name']][self.current_timestamp]['Account_balance']
        current_maximum = self.account[self.current_config['pair_name']][self.current_timestamp]['Maximum']
        
        drawdown = current_maximum - current_balance
        return max(0, drawdown)  # Drawdown should never be negative

    def generate_id(self, symbol, side, random_number=None):
        """Generate unique order IDs"""
        if side not in ["BUY", "SELL"]:
            raise ValueError("Invalid 'side' value. Only 'BUY' or 'SELL' are accepted.")

        if side == 'BUY':
            random_number = random.randint(1000000000, 9999999999)
            order_id = f"{symbol}-{side}-{random_number:010}"
        elif side == "SELL":
            order_id = f"{symbol}-{side}-{random_number}"
        
        return random_number, order_id

    def execute_order(self, side, order_type, fee_rate, random_number=None):
        """Execute trading order with proper fee calculation"""

        random_number, order_id = self.generate_id(self.current_config["pair_name"], side, random_number)
        
        # Build market depth for current timestamp
        self.market_depth_building()
        
        # Get order price from market data
        frequency = self.current_config["data_aggregation_period_stoploss"]
        order_price = self.market_data_dictionnary[self.current_config["pair_name"]][frequency][self.current_timestamp]['open']
        
        # Calculate order size and costs
        order_size, order_value, borrow_amount, borrowing_cost, liquidity_cost = self.order_size(
            random_number, side, order_type
        )
        
        # Calculate fee on original order value
        order_fee = fee_rate * order_value
        
        # Update account and positions
        self.update_account_with_order(order_value, order_fee, side)
        self.update_position(
            random_number, order_size, side, order_price, order_type, borrow_amount
        )
        
        # Create transaction record
        self.create_transaction_row(
            self.current_timestamp, self.current_config["pair_name"], side, order_id, 
            order_type, order_price, order_size, order_value, order_fee, 
            borrow_amount, borrowing_cost, liquidity_cost
        )

        return 

    def create_transaction_row(self, index, symbol, side, order_id, order_type, order_price, order_size, order_value, order_fee, debt, borrowing_cost, liquidity_cost):
        """Create standardized transaction record"""
        transaction = {
            'date': index,
            'symbol': symbol,
            'order_id': order_id,
            'order_type': order_type,
            'price': order_price,
            'size': order_size,
            'value': order_value,
            'fee': order_fee,
            'position': side,
            'borrow': debt,
            'borrowing cost': borrowing_cost,
            'liquidity_cost': liquidity_cost
        }
        # Ensure book_of_order exists and is a DataFrame
        if not hasattr(self, 'book_of_order') or self.book_of_order is None:
            self.book_of_order = pd.DataFrame(columns=transaction.keys())

        # Append the new row
        self.book_of_order.loc[len(self.book_of_order)] = transaction
        
    def order_size(self, random_number, side, order_type):
        """Calculate optimal order size with borrowing cost consideration"""
        
        borrowing_cost = 0.0
        borrow_amount = 0.0
        order_size = 0.0
        order_value = 0.0
        liquidity_cost = 0.0

        # Get current price from market data
        try:
            frequency = self.current_config["data_aggregation_period"]
            price = self.market_data_dictionnary[self.current_config["pair_name"]][frequency][self.current_timestamp]['open']
        except Exception:
            raise ValueError(f"No price data for {self.current_strategy} at {self.current_timestamp}")


        # BUY case
        if side == "BUY":
            cash_balance = self.account[self.current_config['pair_name']][self.current_timestamp]["Fiat_balance"]
            price, liquidity_cost = self.calculate_weighted_price(cash_balance, "asks")
            leverage = self.current_config.get("leverage", 1.0)
            borrow_amount = cash_balance * (leverage - 1)
            order_size = (cash_balance + borrow_amount) / price
            order_value = cash_balance + borrow_amount

        # SELL case
        elif side == "SELL":
            if random_number not in self.positions:
                raise ValueError(f"Position {random_number} not found for strategy {self.current_strategy}")
                
            pos = self.position[self.current_strategy][random_number]
            price, liquidity_cost = self.calculate_weighted_price(pos['Asset_value'], "bids")

            # Compute holding duration
            open_time = datetime.strptime(pos["Open_Time"], "%Y-%m-%d %H:%M:%S")
            current_time = datetime.strptime(self.current_timestamp, "%Y-%m-%d %H:%M:%S")
            duration = current_time - open_time
            duration_in_days = max(1, math.ceil(duration.total_seconds() / (60 * 60 * 24)))
            rate = (self.borrowing_rate / 365) * duration_in_days


            if order_type in ["NORMAL", "STOP", 'MARGIN']:
                borrowing_cost = pos["Debt"] * rate
                order_size = pos["Pending"]
                order_value = order_size * price

            else:
                order_weight = self.current_config['take_profit'][order_type]['Order_weight']
                borrowing_cost = (pos["Asset_value"] * (self.current_config.get("leverage", 1.0) - 1)) * order_weight * rate
                borrow_amount = pos["Debt"] * (1 - order_weight)
                order_size = pos["Pending"] * order_weight
                order_value = order_size * price

        return order_size, order_value, borrow_amount, borrowing_cost, liquidity_cost

    def market_depth_building(self):
        """Calculate market depth for current symbol and timestamp"""
        self.market_depth = {}

        # Ensure timestamp is formatted as string to match DB format "YYYY-MM-DD HH:MM"
        if isinstance(self.current_pre_timestamp, datetime):
            timestamp_str = self.current_pre_timestamp.strftime("%Y-%m-%d %H:%M")
        else:
            # If it's already a string, normalize it
            try:
                # Try parsing with seconds, then reformat without seconds
                timestamp_obj = datetime.strptime(self.current_pre_timestamp, "%Y-%m-%d %H:%M:%S")
                timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                try:
                    # If it’s already in the correct format
                    datetime.strptime(self.current_pre_timestamp, "%Y-%m-%d %H:%M")
                    timestamp_str = self.current_pre_timestamp
                except ValueError:
                    # If the format is completely different, fall back to simple str()
                    timestamp_str = str(self.current_pre_timestamp)

        # Query order book data from MongoDB
        market_order_book_raw_data = self.db["order_book"].find(
            {
                "symbol": self.current_config["pair_name"],
                "timestamp": timestamp_str
            }
        )

        market_order_book_list = list(market_order_book_raw_data)

        if not market_order_book_list:  # Must be replaced by default market depth
            logger.warning(
                f"Warning: No order book data found for pair {self.current_config['pair_name']} "
                f"at {timestamp_str}"
            )
            # Create default market depth if no data
            self.market_depth = {
                "bids": {0.0: 0.0},
                "asks": {0.0: 0.0}
            }
            return

        # Take the first snapshot (latest in the query range)
        order_book = market_order_book_list[0]

        for side in ["bids", "asks"]:
            if side not in order_book:
                continue

            sub_data = order_book[side]

            # Convert lists of [price, qty] → dict of {price: qty}
            if isinstance(sub_data, list):
                sub_data = {float(price): float(quantity) for price, quantity in sub_data}
            else:
                sub_data = {float(price): float(quantity) for price, quantity in sub_data.items()}

            # Sort prices: bids descending, asks ascending
            if side == "bids":
                sorted_prices = OrderedDict(sorted(sub_data.items(), reverse=True))
            else:
                sorted_prices = OrderedDict(sorted(sub_data.items()))

            cumulative_quantity = 0.0
            cumulative_data = {}

            for price, quantity in sorted_prices.items():
                cumulative_quantity += round(quantity, 0)
                cumulative_data[price] = cumulative_quantity

            self.market_depth[side] = cumulative_data

    def calculate_weighted_price(self, quantity_needed, side):
        """
        Calculate weighted price for a given quantity and side.
        
        Args:
            quantity_needed (float): Quantity to be filled
            side (str): 'bids' or 'asks'
            
        Returns:
            float: Cost percentage relative to best price, or None if insufficient liquidity.
        """
        if side not in self.market_depth:
            return self.default_market_price(side), 0

        levels = self.market_depth[side]

        # Best price = highest bid or lowest ask
        best_price = max(levels.keys()) if side == "bids" else min(levels.keys())

        # Sort prices: bids descending, asks ascending
        sorted_levels = sorted(levels.items(), reverse=(side == "bids"))

        total_quantity = 0.0
        total_cost = 0.0

        for price, available_quantity in sorted_levels:
            if total_quantity + available_quantity >= quantity_needed:
                quantity_to_fill = quantity_needed - total_quantity
                total_cost += price * quantity_to_fill
                weighted_price = total_cost / quantity_needed
                liquidity_cost = (weighted_price - best_price) * quantity_to_fill
                return weighted_price, liquidity_cost

            total_quantity += available_quantity
            total_cost += price * available_quantity

        # Not enough liquidity to fill the order
        return self.default_market_price(side), liquidity_cost
    
    def default_market_price(self, side):
        """If there is no market depth, take a default liquidity cost of 10 bps"""

        if side == "BUY":
            market_price = self.market_data_dictionnary[self.current_config['pair_name']][self.current_config["data_aggregation_period_stoploss"]][self.current_timestamp]['close'] * 1.0005
            return market_price
        if side == "SELL":
            market_price = self.market_data_dictionnary[self.current_config['pair_name']][self.current_config["data_aggregation_period_stoploss"]][self.current_timestamp]['close'] * 0.9995
            return market_price

    def update_position(self, random_number, order_size, side, order_price, order_type, debt_cash_flow):
        """Update open positions tracking"""
        # Ensure strategy exists in positions
        if self.current_strategy not in self.position:
            self.position[self.current_strategy] = {}

        if side == "BUY":
            self.position[self.current_strategy][random_number] = {}
            self.position[self.current_strategy][random_number]["Pending"] = order_size
            self.position[self.current_strategy][random_number]["Stoploss"] = (order_price * self.current_config["stop_loss"])
            leverage = self.current_config.get("leverage", 1.0)
            self.position[self.current_strategy][random_number]["Margin"] = (order_price * (1 - (1 / leverage)))
            self.position[self.current_strategy][random_number]["Asset_value"] = (order_price * order_size)
            self.position[self.current_strategy][random_number]["Open_Time"] = self.current_timestamp
            self.position[self.current_strategy][random_number]["Debt"] = debt_cash_flow

        elif side == "SELL":
            if random_number not in self.position[self.current_strategy]:
                logger.warning(f"Position {random_number} not found for SELL order")
                return self.position[self.current_strategy]
                
            self.position[self.current_strategy][random_number]["Pending"] -= order_size
            adaptive_stop_loss = self.current_config.get("adaptive_stop_loss", self.current_config["stop_loss"])
            self.position[self.current_strategy][random_number]["Stoploss"] = (order_price * adaptive_stop_loss)
            self.position[self.current_strategy][random_number]["Debt"] = debt_cash_flow

        # Update take profit levels
        take_profit_config = self.current_config.get("take_profit", {})
        if isinstance(take_profit_config, dict):
            takeprofit_list = list(take_profit_config.keys())
            
            for takeprofit in takeprofit_list:
                self.position[self.current_strategy][random_number][takeprofit] = (order_price * take_profit_config[takeprofit]['Level'])

                if order_type == takeprofit:
                    self.position[self.current_strategy][random_number][takeprofit] = None

                    # Close position if pending is very small (with proper floating point comparison)
                    if self.position[self.current_strategy][random_number]["Pending"] < 0.0001:
                        del self.position[self.current_strategy][random_number]

        # Close position if needed
        if order_type in ["STOP", "NORMAL", "MARGIN"] and side == "SELL":
            if random_number in self.position[self.current_strategy]:
                del self.position[self.current_strategy][random_number]

        return self.position[self.current_strategy]

    def stoploss_trigger(self):

        for random_number in self.positions:

            if random_number not in self.position[self.current_strategy]:
                continue  

            stoploss_price = self.position[self.current_strategy][random_number]['Stoploss']
            current_price = self.market_data_dictionnary[self.current_config['pair_name']]['1min'][self.current_pre_timestamp]['close']

            # Check if stoploss is triggered
            if stoploss_price > current_price:
                self.execute_order(
                    side='SELL',
                    order_type='STOP',
                    fee_rate=0.000,  # 0.1% fee
                    random_number=random_number
                )

                logger.info(f"Stoploss rich for position {random_number}")

    def takeprofit_trigger(self): # In process

        for random_number in self.position[self.current_strategy]:
            # Ensure account has balance before checking stoploss
            if self.account[self.current_config['pair_name']][self.current_timestamp]['Asset_balance'] <= 0:
                continue  

            if random_number not in self.position[self.current_strategy]:
                continue 

            for key, value in self.current_config["take_profit"].items():
                current_price = self.market_data_dictionnary[self.current_config['pair_name']]['1min'][self.current_pre_timestamp]['close']
                takeprofit_level = value['Level'] * current_price

                # Check if stoploss is triggered
                if takeprofit_level < current_price:
                    self.execute_order(
                        side='SELL',
                        order_type= key,
                        fee_rate=0.000,  # 0.1% fee
                        random_number=random_number
                    )
                    logger.info(f"Take profit order executed for position {random_number}")

    def margin_trigger(self):

        for random_number in self.position[self.current_strategy]:
            # Ensure account has balance before checking stoploss
            if self.account[self.current_config['pair_name']][self.current_timestamp]['Asset_balance'] <= 0:
                continue  

            if random_number not in self.position[self.current_strategy]:
                continue  

            margin_price = self.position[self.current_strategy][random_number]['Margin']
            current_low = self.market_data_dictionnary[self.current_config['pair_name']]['1min'][self.current_pre_timestamp]['low']

            # Check if stoploss is triggered
            if margin_price > current_low:
                self.execute_order(
                    side='SELL',
                    order_type='MARGIN',
                    fee_rate=0.000,  # 0.1% fee,
                    random_number=random_number
                )

                logger.info(f"Liquidation process for position {random_number}")

    def valid_candle(self, dt, frequency: str):
        """
        Check if a datetime `dt` is a valid Binance candle start for the given frequency (UTC-based).
        Returns:
            (is_valid: bool, current_candle_str: str, previous_candle_str: str)
        Timestamps are returned as strings in "%Y-%m-%d %H:%M" format.
        """
        # Normalize input
        if isinstance(dt, str):
            try:
                # Try parsing with seconds
                dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                dt = datetime.strptime(dt, "%Y-%m-%d %H:%M")

        # Initialize variables
        valid = False
        prev_dt = None

        # Compute validity and previous timestamp
        if frequency.endswith("min"):
            interval = int(frequency[:-3])
            valid = dt.minute % interval == 0 and dt.second == 0 and dt.microsecond == 0
            prev_dt = dt - timedelta(minutes=interval)

        elif frequency.endswith("hour"):
            interval = int(frequency[:-4])
            valid = dt.hour % interval == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0
            prev_dt = dt - timedelta(hours=interval)

        elif frequency == "1day":
            valid = dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0
            prev_dt = dt - timedelta(days=1)

        elif frequency == "1week":
            valid = dt.weekday() == 0 and dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0
            prev_dt = dt - timedelta(weeks=1)

        elif frequency == "1month":
            valid = dt.day == 1 and dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0
            # Handle previous month safely
            prev_month = (dt.month - 1) if dt.month > 1 else 12
            prev_year = dt.year if dt.month > 1 else dt.year - 1
            prev_dt = dt.replace(year=prev_year, month=prev_month, day=1)

        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        first_index =  datetime.strptime(self.index_list[0], "%Y-%m-%d %H:%M:%S") # Compar prev timestamp to first index
        #first_index =  datetime.strptime(self.index_list[0], "%Y-%m-%d %H:%M:%S")
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        prev_str = prev_dt.strftime("%Y-%m-%d %H:%M:%S")

        if prev_dt < first_index:
            return valid, dt_str, self.index_list[0] 

        return valid, dt_str, prev_str

    def process_strategy(self, strategy):
        """
        Process the strategy by fetching data and computing indicators.
        :param strategy_name: The name of the strategy.
        :param strategy: The strategy instance.
        """
        #logger.info(f"Processing strategy: {strategy_name}")
        dataset = self.market_data_dictionnary[self.current_config['pair_name']][self.current_config['data_aggregation_period']]

        if strategy.buy_condition(data=dataset, timestamp = self.current_pre_timestamp) and self.account[self.current_config['pair_name']][self.current_pre_timestamp]['Fiat_balance'] > 10:
            return 'BUY'

        elif strategy.sell_condition(data=dataset, timestamp = self.current_pre_timestamp) and len(self.position.get(self.current_strategy, [])) > 0:  # Only sell if a position exists
            return 'SELL'

        else:
            return 'HOLD'

    def build_book_of_trade(self):
        global_trading_report = []

        for strategy, config in self.strategies_config.items():
            pair_name = config["pair_name"]

            local_order_book = self.book_of_order[self.book_of_order["symbol"] == pair_name].copy()
            local_order_book["order_number"] = local_order_book["order_id"].str.rsplit("-", n=1).str[-1]

            buy_orders = local_order_book[local_order_book["position"] == "BUY"]
            sell_orders = local_order_book[local_order_book["position"] == "SELL"]

            grouped_buy = (
                buy_orders.groupby("order_number")
                .apply(
                    lambda g: pd.Series({
                        "date": g["date"].max(),
                        "symbol": g["symbol"].iloc[0],
                        "price": g["price"].max(),
                        "size": g["size"].sum(),
                        "value": g["value"].sum(),
                        "fee": g["fee"].sum(),
                        "position": "BUY",
                        "borrow": g["borrow"].max(),
                        "borrowing cost": g["borrowing cost"].sum(),
                        "liquidity_cost": g["liquidity_cost"].sum()
                    })
                )
            )

            grouped_sell = (
                sell_orders.groupby("order_number")
                .apply(
                    lambda g: pd.Series({
                        "date": g["date"].max(),
                        "symbol": g["symbol"].iloc[0],
                        "price": g["price"].max(),
                        "size": g["size"].sum(),
                        "value": g["value"].sum(),
                        "fee": g["fee"].sum(),
                        "position": "SELL",
                        "borrow": g["borrow"].max(),
                        "borrowing cost": g["borrowing cost"].sum(),
                        "liquidity_cost": g["liquidity_cost"].sum()
                    })
                )
            )


            merged = grouped_buy.join(grouped_sell, lsuffix="_buy", rsuffix="_sell", how="inner")

            if merged.empty:
                continue

            merged["buy_date"] = pd.to_datetime(merged["date_buy"], errors="coerce", utc=False)
            merged["sell_date"] = pd.to_datetime(merged["date_sell"], errors="coerce", utc=False)
            merged["Duration"] = merged["sell_date"] - merged["buy_date"]
            merged["Absolute Performance %"] = ((merged["price_sell"] - merged["price_buy"]) / merged["price_buy"]) * 100
            merged["Levered Performance %"] = (
                (merged["value_sell"] - merged["value_buy"] - merged["borrow_buy"]) / merged["value_buy"]
            ) * 100
            merged["Profit & Loss"] = merged["value_sell"] - merged["value_buy"] - merged["borrow_buy"]
            merged["Fees"] = merged["fee_sell"] + merged["fee_buy"]
            merged["Positive"] = (merged["Levered Performance %"] > 0).astype(int)
            merged["Liquidity cost total"] = merged["liquidity_cost_sell"] + merged["liquidity_cost_buy"]


            trading_report = merged.rename(columns={
                "date_buy": "Open_time",
                "date_sell": "Close_time",
                "symbol_buy": "Symbol",
                "price_buy": "Open",
                "price_sell": "Close",
                "value_buy": "USDT value open",
                "value_sell": "USDT value close",
                "borrowing cost_buy": "Borrowing cost",
                "liquidity_cost_buy": "Liquidity cost buy",
                "liquidity_cost_sell": "Liquidity cost sell",
            })[
                [
                    "Open_time", "Close_time", "Duration", "Symbol",
                    "Open", "Close",
                    "USDT value open", "USDT value close",
                    "Fees", "Profit & Loss",
                    "Absolute Performance %", "Levered Performance %",
                    "Positive", "Borrowing cost",
                    "Liquidity cost total", "Liquidity cost buy", "Liquidity cost sell"
                ]
            ]

            global_trading_report.append(trading_report)

        if global_trading_report:
            self.book_of_trade = pd.concat(global_trading_report, ignore_index=True)

    def load_book_of_trade(self):
        """
        Store the DataFrame `book_of_trade` into MongoDB under collection 'book_of_trade_test',
        with a unique document _id = YYYYMMDDHHMMSS_SYMBOL.
        """

        def to_mongo_safe(obj):
            """Recursively convert any object to Python-native types for MongoDB."""
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return obj.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(obj, pd.Timedelta):
                return obj.total_seconds()
            elif isinstance(obj, dict):
                return {k: to_mongo_safe(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, set)):
                return [to_mongo_safe(v) for v in obj]
            elif obj != obj:  # NaN check
                return None
            else:
                return obj

        # --- 1. Unique ID ---
        now_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        symbol = self.current_config["pair_name"]
        unique_id = f"{now_str}_{symbol}"

        # --- 2. Prepare DataFrame ---
        df_to_save = self.book_of_trade.copy()
        df_to_save.reset_index(drop=True, inplace=True)

        # --- 3. Convert all cells to Mongo-safe ---
        df_safe = df_to_save.applymap(to_mongo_safe)

        # --- 4. Convert to list of dicts ---
        records = df_safe.to_dict(orient="records")

        # --- 5. Insert into MongoDB ---
        collection = self.db["BookOfTrade"]
        document = {
            "_id": unique_id,
            "symbol": symbol,
            "created_at": datetime.utcnow(),
            "nb_trades": len(df_safe),
            "data": records
        }

        try:
            collection.insert_one(document)
            print(f"✅ book_of_trade successfully saved under ID: {unique_id}")
            return unique_id
        except pymongo.errors.DuplicateKeyError:
            print(f"⚠️ Document with ID {unique_id} already exists — skipping insert.")
            return unique_id
        except Exception as e:
            print(f"❌ Error while saving book_of_trade: {e}")
            return None


    def backtest(self):
        """Run backtest for all strategies"""

        index_list = self.get_timestamps(self.market_data_dictionnary)

        # Initialize first timestamp
        self.current_timestamp = index_list[0]
        self.current_pre_timestamp = index_list[0]

        for timestamp in index_list[1:]:

            self.current_pre_timestamp = self.current_timestamp
            self.current_timestamp = timestamp
    
            for strategy, config in self.strategies_config.items():

                self.current_strategy = strategy
                self.current_config = config  # a dict with configs

                # Get timestamps
                frequency = self.current_config["data_aggregation_period"]
                complet_candle, self.current_timestamp, self.current_pre_timestamp = self.valid_candle(self.current_timestamp, frequency)

                # Initialize account for first timestamp
                self.update_account()
                
                for strategy_name, strategy in self.strategies.items():

                    self.positions = [k for k in self.position.get(self.current_strategy, {})]
                    
                    if complet_candle == True:

                        action = self.process_strategy(strategy)

                        if action  == 'BUY':

                            self.execute_order(
                                side=action, 
                                order_type='NORMAL', 
                                fee_rate=0.001  # 0.1% fee
                            )
                        
                        if action  == 'SELL':

                            for random_number in self.positions:
                                self.execute_order(
                                    side=action,
                                    order_type='NORMAL',
                                    fee_rate=0.001,
                                    random_number=random_number
                                )

                    self.stoploss_trigger() # Update to knew structure
                    self.margin_trigger() # Update to knew structure
                    self.takeprofit_trigger() # Update to knew structure


        logger.info(f"Backtest completed")
