import os
import traceback
from collections import defaultdict
import logging

from binance.client import Client
import pymongo
from dotenv import dotenv_values
from datetime import datetime

import json
import pandas as pd
from utils.IndicatorBuilder import IndicatorBuilder
from utils.trading_class import trading_class
from utils.report_class import report_class

from MeanReversingStrategy import MeanReversingStrategy
from MomentumStrategy import MomentumStrategy

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
    "MeanReversingStrategy" : MeanReversingStrategy,
    "MomentumStrategy" : MomentumStrategy
}

TICK_PERIOD = 60  # Default tick period in seconds

class TradingWorker : 
    def __init__(self, strategies_config_path=os.path.join(os.path.dirname(__file__), 'configs/strategies.json')):
        logger.info("Loading strategies from config file...")
        self.strategies_config_path = strategies_config_path
        self.strategies_config = {}
        self.strat_datasets = {}
        self.fiat_strategies_mapping = defaultdict(list)
        self.strat_indicator_builders = {}
        self.strategies = self.load_strategies()
        self.max_dataset_size = 1000  # Maximum number of rows to keep in memory
        logger.info("Strategies loaded successfully.")
        
        config = dotenv_values(os.path.join(os.path.dirname(__file__), '.env'))
        logger.info("Configuration loaded from .env file.")
        #binance API credentials
        api_key = config["API_KEY"]  
        api_secret = config["SECRET_KEY"] 
        self.binance_client = Client(api_key=api_key, api_secret=api_secret)
        logger.info("Binance client initialized with API key and secret.")

        # MongoDB connection
        self.mongo_client = pymongo.MongoClient(config["MONGO_URI"], int(config["MONGO_PORT"]), username=config["MONGO_USER"], password=config["MONGO_PASSWORD"])
        self.db = self.mongo_client.get_database("quants_db_fin")
        logger.info("MongoDB client initialized with URI and port.")
     

        self.trader = trading_class(self.mongo_client, self.binance_client, db_name="quants_db_fin")
        self.report = report_class(self.mongo_client, self.binance_client, db_name="quants_db_fin")

        self.strategy_next_action = dict([(strategy, "BUY") for strategy in self.strategies.keys()])
        self.tick_counters = dict([(strategy, 0) for strategy in self.strategies.keys()])
        self.stoploss_tick_counters = dict([(strategy, 0) for strategy in self.strategies.keys()])
        self.buy_values = dict([(strategy, -1) for strategy in self.strategies.keys()])
        self.random_numbers = dict([(strategy, -1) for strategy in self.strategies.keys()])
        
    def load_strategies(self):

        with open(self.strategies_config_path, 'r') as file:
            strategies_config = json.load(file)
            self.strategies_config = strategies_config
        
        strategies = {}
        for strategy_name, config in strategies_config.items():
            class_to_instanciate = config["class"]
            if class_to_instanciate in PROD_STRATEGIES:
                
                strat_config = config.copy()
                strat_config.pop("class", None)  # Remove class key if it exists

                self.strat_indicator_builders[strategy_name] = IndicatorBuilder(config.get("indicators", []))
                self.fiat_strategies_mapping[config["balance_currency"]].append(strategy_name)
                strategies[strategy_name] = PROD_STRATEGIES[class_to_instanciate](**strat_config)
            else:
                logger.warning(f"Warning: Strategy {class_to_instanciate} not found in PROD_STRATEGIES.")
        
        return strategies

    def get_data_and_compute_indicators(self) : 
        for strategy, config in self.strategies_config.items():
            try:
                #get ohlcv data from mongodb for the pair
                olhcv_data = self.db["olhcv"].find({"symbol" : config["pair_name"]}).sort("timestamp", pymongo.DESCENDING).limit(config["required_data_depth"])
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
                    olhcv_df["timestamp"] = pd.to_datetime(olhcv_df["timestamp"], format='%d/%m/%Y %H:%M')
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
                if "data_aggregation_period" in config["required_data_config"]:
                    try:
                        olhcv_df = self.resample_dataset(olhcv_df, config["required_data_config"]["data_aggregation_period"])
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
                    
                # Clean up old data to prevent memory accumulation
                if len(olhcv_df) > self.max_dataset_size:
                    olhcv_df = olhcv_df.tail(self.max_dataset_size)
                    logger.info(f"Truncated dataset for strategy {strategy} to {self.max_dataset_size} rows")
                    
                self.strat_datasets[strategy] = olhcv_df
                
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
    
    def process_strategy(self, strategy_name, strategy):
        """
        Process the strategy by fetching data and computing indicators.
        :param strategy_name: The name of the strategy.
        :param strategy: The strategy instance.
        """
        logger.info(f"Processing strategy: {strategy_name}")
        dataset = self.strat_datasets[strategy_name]
        next_action = self.strategy_next_action[strategy_name]

        if next_action == "BUY":
            if strategy.buy_condition(data=dataset):
                logger.info(f"Buy condition met for {strategy_name}. Executing buy order.")
                return 'BUY'
            else:
                logger.info(f"Buy condition not met for {strategy_name}.")
                return 'HOLD'
        elif next_action == "SELL":
            if strategy.sell_condition(data=dataset):
                logger.info(f"Sell condition met for {strategy_name}. Executing sell order.")
                return 'SELL'
            else:
                logger.info(f"Sell condition not met for {strategy_name}.")
                return 'HOLD'

    def check_stop_loss_and_take_profit(self, strategy_name, buy_value):
        """
        Check if the stop loss condition is met for the given strategy.
        :param strategy_name: The name of the strategy.
        :param current_price: The current price of the asset.
        """
        strategy = self.strategies[strategy_name]
        current_price = self.strat_datasets[strategy_name].iloc[-1]['close']

        if self.strategy_next_action[strategy_name] == "SELL":
            self.stoploss_tick_counters[strategy_name] += 1
            if self.stoploss_tick_counters[strategy_name] >= strategy.stoploss_tick_period:
                if strategy.stoploss_condition(buy_value, current_price) or strategy.take_profit_condition(buy_value, current_price):
                    logger.info(f"Stop loss/Take profit condition met for {strategy_name}. Executing stop loss order.")
                    return True
                self.stoploss_tick_counters[strategy_name] = 0
        return False

    def tick(self) :
        logger.info("Starting trading worker tick...")
        try:
            self.get_data_and_compute_indicators()
        except Exception as e:
            logger.error(f"Error in get_data_and_compute_indicators: {e}")
            logger.error(traceback.format_exc())
            return
            
        for strategy_name, strategy in self.strategies.items():
            
            
            if self.check_stop_loss_and_take_profit(strategy_name, self.buy_values[strategy_name]):
                try : 
                    logger.info(f"Stop loss / Take Profit triggered for {strategy_name}. Executing sell order. with buy value {self.buy_values[strategy_name]}")
                    order, random_number, order_id = self.trader.create_order_spot(
                        symbol=strategy.spot_currency,
                        fiat=strategy.balance_currency,
                        side='SELL',
                        order_type='MARKET',
                        random_number=self.random_numbers[strategy_name],
                        quantity_weight=1.0,  # Assuming we want to sell all holdings
                    )
                    self.strategy_next_action[strategy_name] = "BUY"
                    self.trader.delete_position_to_book_trade(random_number)
                    continue
            
                except Exception as e:
                    logger.error(f"Error executing stop loss order for {strategy_name}: {e}")
                    logger.error(traceback.format_exc())
                    continue
            
            # Increment the tick counter for the strategy
            self.tick_counters[strategy_name] += 1
            if self.tick_counters[strategy_name] >= strategy.tick_period:
                logger.info(f"Processing strategy {strategy_name} at tick {self.tick_counters[strategy_name]}")
                self.tick_counters[strategy_name] = 0
                try:
                    action = self.process_strategy(strategy_name, strategy)
                except Exception as e:
                    logger.error(f"Error processing strategy {strategy_name}: {e}")
                    logger.error(traceback.format_exc())
                    continue
                logger.info(f"Action for strategy {strategy_name}: {action}")
                if action == 'BUY':
                    try : 
                        # Calculate quantity weight with validation
                        available_strategies = self.fiat_strategies_mapping.get(strategy.balance_currency, [])
                        if not available_strategies:
                            logger.warning(f"Warning: No strategies found for currency {strategy.balance_currency}")
                            continue
                            
                        total_weight = sum([
                            self.strategies[s_name].allocated_weight 
                            for s_name in available_strategies 
                            if self.strategy_next_action[s_name] == "BUY"
                        ])
                        
                        if total_weight <= 0:
                            logger.warning(f"Warning: Total weight is 0 for currency {strategy.balance_currency}")
                            continue
                            
                        q_weight = strategy.allocated_weight / total_weight
                        
                        order, random_number, order_id = self.trader.create_order_spot(
                            symbol=strategy.spot_currency,
                            fiat=strategy.balance_currency,
                            side='BUY',
                            order_type='MARKET',
                            quantity_weight=q_weight,
                        )
                        logger.info(f"Buy order executed for {strategy_name}.")
                        self.strategy_next_action[strategy_name] = "SELL"
                        self.buy_values[strategy_name] = self.strat_datasets[strategy_name].iloc[-1]['close']
                        self.random_numbers[strategy_name] = random_number
                    except Exception as e:
                        logger.error(f"Error executing buy order for {strategy_name}: {e}")
                        logger.error(traceback.format_exc())
                elif action == 'SELL':
                    try:
                        self.trader.create_order_spot(
                            symbol=strategy.spot_currency,
                            fiat=strategy.balance_currency,
                            side='SELL',
                            order_type='MARKET',
                            random_number=self.random_numbers[strategy_name], # UPDATES 31/12/2025
                            quantity_weight=1.0,  # Assuming we want to sell all holdings
                        )
                        logger.info(f"Sell order executed for {strategy_name}.")
                        self.report.add_trade_to_trade_report() # UPDATES 31/12/2025
                        self.strategy_next_action[strategy_name] = "BUY"
                    except Exception as e:
                        logger.error(f"Error executing sell order for {strategy_name}: {e}")
                        logger.error(traceback.format_exc())
                
        logger.info("Tick processing completed.")
                
if __name__ == "__main__":
    worker = TradingWorker()
    while True:
        worker.tick()
        # Sleep for a while to simulate the tick period
        import time
        time.sleep(TICK_PERIOD)  # Adjust the sleep time as needed