

import pymongo
from  dotenv import dotenv_values
import os
import re

# Load environment variables from .env file
config = dotenv_values(os.path.join(os.path.dirname(__file__), '.env'))
print("Configuration loaded from .env file.")

# MongoDB connection
mongo_client = pymongo.MongoClient(config["MONGO_URI"], int(config["MONGO_PORT"]), username=config["MONGO_USER"], password=config["MONGO_PASSWORD"])
db = mongo_client.get_database("quants_db")
print("MongoDB client initialized with URI and port.")


class report_class:


    def __init__(self, mongo_client, binance_client, db_name="quants_db_fin"):

        """
        Est ce qu'on est obligé de réinitialiser la config, et binance client dans la class trading class ???
        """
        
        self.client = binance_client
        self.mongo_client = mongo_client
        self.db_name = db_name
        self.db = mongo_client.get_database(db_name)

        print(f"MongoDB client initialized with database {db_name}.")
        if 'OrderBook' not in self.db.list_collection_names():
            self.db.create_collection('OrderBook')
            print(f"Collection OrderBook created in MongoDB.")    

        if 'OrderHistory' not in self.db.list_collection_names():
            self.db.create_collection('OrderHistory')
            print(f"Collection OrderHistory created in MongoDB.")

        if 'BookOfTrade' not in self.db.list_collection_names():
            self.db.create_collection('BookOfTrade')
            print("Collection 'BookOfTrade' created.")


    def check_trade_analyzed(self, random_id):
        """
        Verify if the trade has already been analyzed
        :param random_id: trade identifier
        """
        return self.db["BookOfTrade"].find_one({"trade_id": random_id}) is not None


    def extract_all_random_ids(self):
        orders = self.db["OrderHistory"].find({}, {"order_id": 1})
        return list({
            oid['order_id'].split("-")[-1]
            for oid in orders
            if 'order_id' in oid and len(oid['order_id'].split("-")) == 3
        })


    def get_order_pair(self, random_id):
        """
        Retrieve the buy and sell order of the trade based on the random number
        :param random_id: trade identifier
        """
        regex = re.compile(f"-{random_id:010}$")
        query = {"order_id": {"$regex": regex}}

        orders = list(self.db["OrderHistory"].find(query))
        buy = next((o for o in orders if o.get("side") == "BUY"), None)
        sell = next((o for o in orders if o.get("side") == "SELL"), None)

        return buy, sell


    def performance(self, open, close):
        """
        Calculate performance of the trade
        :param open: open level
        :param close: close level
        """
        return (close - open)/open


    def add_trade_to_trade_report(self):
        for random_id in self.extract_all_random_ids():
            if self.check_trade_analyzed(random_id):
                continue

            buy_order, sell_order = self.get_order_pair(random_id)
            if not buy_order or not sell_order:
                print(f"Trade {random_id} incomplete, skipping.")
                continue

            try:
                report = {
                    'trade_id': random_id,
                    'symbol': buy_order['symbol'],
                    'open_time': buy_order['date'],
                    'close_time': sell_order['date'],
                    'duration': sell_order['date'] - buy_order['date'],
                    'open': buy_order['execution_price'],
                    'close': sell_order['execution_price'],
                    'usdt_value_open': buy_order['settled_amount_executed'],
                    'usdt_value_close': sell_order['settled_amount_executed'],
                    'fees': buy_order['commission'] + sell_order['commission'],
                    'profit_loss': sell_order['settled_amount_executed'] - buy_order['settled_amount_executed'],
                    'performance_pct': self.performance(buy_order['execution_price'], sell_order['execution_price']),
                    # 'strategy': buy_order['stratgy']
                }

                self.db['BookOfTrade'].insert_one(report)
                print(f"Trade {random_id} saved.")

            except KeyError as e:
                print(f"Missing key in trade {random_id}: {e}")

