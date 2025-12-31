
from binance.client import Client
import math
import pymongo
from  dotenv import dotenv_values
import os
import random
from datetime import datetime
import pandas as pd


class trading_class:

    def __init__(self):

        """
        Est ce qu'on est obligé de réinitialiser la config, et binance client dans la class trading class ??? NON
        """
        
        # Load config, and get api_key, api_secret
        config = dotenv_values(os.path.join(os.path.dirname(__file__), '.env'))
        print("Configuration loaded from .env file.")
        self.api_key = config["API_KEY"]
        self.api_secret = config["SECRET_KEY"]
        self.client = Client(api_key=self.api_key, api_secret=self.api_secret)
        print("Binance client initialized with API key and secret.")

        # Load connection with mongo database
        mongo_client = pymongo.MongoClient(config["MONGO_URI"], int(config["MONGO_PORT"]), username=config["MONGO_USER"], password=config["MONGO_PASSWORD"])
        self.db = mongo_client.get_database("quants_db_fin")
        print("MongoDB client initialized with URI and port.")

        # Could be improve (necessary ?)
        if 'OrderBook' not in self.db.list_collection_names():
            self.db.create_collection('OrderBook')
            print(f"Collection OrderBook created in MongoDB.")    

        if 'OrderHistory' not in self.db.list_collection_names():
            self.db.create_collection('OrderHistory')
            print(f"Collection OrderHistory created in MongoDB.")


    def generate_id(self, symbol, side, random_number):
        """
        Build ad unique id for each order executed.
        :param symbol: The symbol traded.
        :param side: The side of the order.
        :param random_number: For sell order the random number of the buy order.
        """
        if side == 'BUY':
            random_number = random.randint(1000000000, 9999999999) # Générer un nouvel ID pour l'opération
            order_id = f"{symbol}-{side}-{random_number:010}"
            return random_number, order_id

        elif side == "SELL":
            order_id = f"{symbol}-{side}-{random_number}" # Utiliser same id as open order
            return random_number, order_id


    def get_asset_balance(self, symbol):
        """
        Get the balance of the spot account for a specific symbol.
        :param symbol: The symbol traded.
        """
        balance_asset = self.client.get_asset_balance(asset=symbol) # Binance request for the symbol

        if balance_asset and 'free' in balance_asset:
            asset = float(balance_asset['free'])
        else:
            asset = 0.0

        return asset


    def get_spot_balance(self, symbol, fiat):
        """
        Get the balance of the spot account for both symbol and fiat.
        :param symbol: The symbol traded.
        :param fiat: The fiat symbol.
        """
        balance_crypto = self.get_asset_balance(symbol)
        balance_fiat = self.get_asset_balance(fiat)
        return balance_crypto, balance_fiat


    def get_margin_balance(self, symbol, fiat):
        """
        Get the balance of the margin account for a specific pair.
        :param symbol: The symbol traded.
        :param fiat: The fiat symbol.
        """
        margin_account_info = self.client.get_isolated_margin_account(symbols= symbol + fiat) # Request

        balance_crypto = 0.0
        balance_fiat = 0.0
        
        for asset in margin_account_info['assets']:
            if asset['baseAsset']['asset'] == symbol:
                balance_crypto = float(asset['baseAsset']['free'])
                balance_fiat = float(asset['quoteAsset']['free'])

        return balance_crypto, balance_fiat
    

    def create_repay_margin_loan(self, symbol, fiat, side, leverage):
        """
        Create or repay a margin loan.
        :param symbol: The symbol traded.
        :param fiat: The fiat symbol.
        :param side: Buy or Sell.
        :param leverage: Level of leverage wanted (*1). Could be improved, get directly from config
        """
        type_transaction = 'BORROW' if side == 'BUY' else 'REPAY'
        balance_crypto, balance_fiat = self.get_margin_balance(symbol, fiat)
        quantity = str(round(balance_fiat * (leverage - 1), 2))

        # Borrow & Repay
        if type_transaction == 'BORROW':
            self.client.create_margin_loan(asset=fiat, symbol=symbol + fiat, amount=quantity, isIsolated='TRUE')
        else:
            self.client.repay_margin_loan(asset=fiat, symbol=symbol + fiat, amount=quantity, isIsolated='TRUE')

        return

        
    def create_order_margin(self,  symbol, fiat, side, order_type, leverage, random_number, account='Margin'):
        """
        Create a margin order.
        :param symbol: The symbol traded.
        :param fiat: The fiat symbol.
        :param side: Buy or Sell.
        :param order_type: GTC; LIMIT; ...
        :param leverage: Level of leverage wanted (*1). Could be improved, get directly from config
        :param random_number: For sell order the random number of the buy order.
        :param account: Account used.
        """
        if side == 'BUY':
            self.create_repay_margin_loan(symbol, fiat, side, leverage) # Borrow money for leverage order, logic could be improved
        random_number, order_id = self.generate_id(symbol, side, random_number) # Generate order ID for tracking purpose
        balance_crypto, balance_fiat = self.get_margin_balance(symbol, fiat) # Request available balance
        price = float(self.client.get_symbol_ticker(symbol= symbol + fiat)['price']) # Request market price
        order_amount = balance_fiat if side == 'BUY' else math.floor(balance_crypto * price) # Calculate order amount in fiat amount

        # Placer l'ordre
        order = self.client.create_margin_order(
                                symbol = symbol + fiat,
                                side = side,
                                type = order_type,
                                quoteOrderQty = order_amount,
                                newClientOrderId=order_id,
                                isIsolated='TRUE')
        


        self.update_book_trade(account, symbol, side, random_number, order_amount, price) # Update order book (stoploss, ...)
        if side == 'SELL' and leverage > 1:
            self.create_repay_margin_loan(symbol, fiat, side, leverage) # Repay Borrow account, logic could be improved
        self.add_order_to_order_history(random_number, symbol, order_type, side, price, order_amount)
        
        return order
    

    def create_order_spot(self, symbol, fiat, side, order_type, random_number, account = 'Spot'):
        """
        Create a spot order.
        :param symbol: The symbol traded.
        :param fiat: The fiat symbol.
        :param side: Buy or Sell.
        :param order_type: GTC; LIMIT; ...
        :param random_number: For sell order the random number of the buy order.
        :param account: Account used.
        """
        random_number, order_id = self.generate_id(symbol, side, random_number) # Generate order ID for tracking purpose
        balance_crypto, balance_fiat = self.get_spot_balance(symbol, fiat) # Request available balance
        price = float(self.client.get_symbol_ticker(symbol= symbol + fiat)['price']) # Request market price
        order_amount = balance_fiat if side == 'BUY' else math.floor(balance_crypto * price) # Calculate order amount in fiat amount

        # Placer l'ordre
        order = self.client.create_order(
                                symbol = symbol + fiat,
                                side = side,
                                type = order_type,
                                quoteOrderQty = order_amount,
                                newClientOrderId=order_id
                                )
        

        self.update_book_trade(account, symbol, side, random_number, order_amount, price) # Update order book (stoploss, ...)
        self.add_order_to_order_history(random_number, symbol, side, order_type, price, order_amount)

        return order


    def fees_optimization(self, fiat_symbol, amount_usdt, limit):

        """
        Optimize fees by paying them with BNB, for transaction executed on binance fees are reduce of 25% for transactions paid in BNB

        """

        if amount_usdt < 10:
            print("Amount order to small ")
            return

        else :
            bnb_available, fiat_available = self.get_spot_balance("BNB", fiat_symbol)
            actualPrice = float(self.client.get_symbol_ticker(symbol= "BNB" + fiat_symbol)['price'])
            limit = round(limit / actualPrice, 4)
            quantity = round(amount_usdt / actualPrice, 3)

            if bnb_available < limit and fiat_available > 10:

                    # PLACE MARKET ORDER
                    OrderMarket = self.client.create_order(
                        symbol="BNBUSDT",
                        side="BUY",
                        type='MARKET',
                        quantity=quantity)
                    
                    print("Buy BNB to pay fees : ", OrderMarket)


    def add_position_to_book_trade(self, account, symbol, side, random_number, settled_amount, execution_price):
        """
        Add a position to the order book.
        :param account: Account used.
        :param side: Bur or Sell.
        :param symbol: The symbol traded.
        :param random_number: For sell order the random number of the buy order.
        :param settled_amount: Size of the order in fiat.
        :param execution_price: Market price at the moment of the execution
        """
        transactions = {
            "order_id": random_number,
            "symbol": symbol,
            "side": side,
            "quantity": settled_amount,
            "price": execution_price,
            #"stoploss": stoploss,
            #"takeprofit": takeprofit,
            #"loan": loan,
            #"account": margin
            #"startegy": strategy
        }

        self.db['OrderBook'].insert_one(transactions)
        print(f"Order {random_number} added to OrderBook.")


    def delete_position_to_book_trade(self, random_number):
        """
        Delete a position from the order book.
        :param random_number: For sell order the random number of the buy order.
        """
        result = self.db['OrderBook'].delete_one({"order_id": random_number})
        if result.deleted_count:
            print(f"Order {random_number} deleted from OrderBook.")
        else:
            print(f"Order {random_number} not found.")


    def update_book_trade(self, account, symbol, side, random_number, settled_amount, execution_price):    
        """
        Update the status of the order book.
        :param account: Account used.
        :param side: Bur or Sell.
        :param symbol: The symbol traded.
        :param random_number: For sell order the random number of the buy order.
        :param settled_amount: Size of the order in fiat.
        :param execution_price: Market price at the moment of the execution
        """
        if side == 'BUY':
            self.add_position_to_book_trade(account, symbol, side, random_number, settled_amount, execution_price)
            return
        elif side == 'SELL':
            self.delete_position_to_book_trade(random_number)
            return


    def add_order_to_order_history(self, order_id, symbol, type, side, execution_price, settled_amount):  
        """
        Add a transaction to the order history.
        :param order_id: Order ID for tracking.
        :param symbol: The symbol traded.
        :param type: Order type.
        :param side: Buy or Sell.
        :param execution_price: Market price at the moment of the execution.
        :param settled_amount: Size of the order in fiat.
        """
        transaction = {   
        'order_id': order_id,
        'date': datetime.now().timestamp(),
        'symbol': symbol,
        'order_type': type,
        'side' : side,
        'size': settled_amount/execution_price,
        'arrival price': execution_price,
        'settled amount': settled_amount,
        # 'strategy': strategy
        }

        self.db['OrderHistory'].insert_one(transaction)
        print(f"Order {order_id} added to OrderHistory.")


    def get_trades_real_time_details(self, traded_symbols, start, end):
        """
        Obtain trades details, this function aim to tick daily to retrieve execution data (execution price,
        quantity_executed, settled_amount_executed, commission, commission_asset, execution_time)
        :param traded_symbols: symbol traded during the day.
        :param start: start date to retrieve data.
        :param end: end date to retrieve data.
        """
        for symbol in traded_symbols:
            trades_info = self.client.get_my_trades(symbol=symbol, startTime=start, endTime=end)
            for trade in trades_info:
                order_id = trade['orderId']
                side = 'BUY' if trade['isBuyer'] else 'SELL'
                update_fields = {
                    'execution_price': float(trade['price']),
                    'quantity_executed': float(trade['qty']),
                    'settled_amount_executed': float(trade['quoteQty']),
                    'commission': float(trade['commission']),
                    'commission_asset': trade['commissionAsset'],
                    'execution_time': trade['time']
                }

                result = self.db['OrderHistory'].update_one(
                    {'order_id': order_id, 'side': side},
                    {'$set': update_fields}
                )

                status = "updated" if result.modified_count > 0 else "not found or already updated"
                print(f"Order {order_id} [{side}] on {symbol} {status}.")


    @staticmethod
    def compare_market_data(dataset_1: pd.DataFrame, dataset_2: pd.DataFrame) -> float: # Function test
        """
        Compare two DataFrames row-by-row based on 'timestamp' and return match percentage.
        :param dataset_1: dataframe to compar.
        :param dataset_2: dataframe to compar.
        """
        try:
            dataset_1 = dataset_1.set_index('timestamp')
            dataset_2 = dataset_2.set_index('timestamp')
            dataset_1, dataset_2 = dataset_1.align(dataset_2, join='inner')
            if dataset_1.empty:
                print("No matching timestamps found.")
                return 0.0

            match_mask = (dataset_1 == dataset_2).all(axis=1)
            # Handle both pandas Series and numpy arrays
            if hasattr(match_mask, 'sum'):
                match_count = match_mask.sum()
            else:
                match_count = sum(match_mask) if isinstance(match_mask, (list, tuple)) else int(match_mask)
            
            total_count = len(match_mask) if hasattr(match_mask, '__len__') else 1
            match_percent = (match_count / total_count) * 100
            print(f"Matched: {match_count}/{total_count} rows ({match_percent:.2f}%)")
            return match_percent
        except Exception as e:
            print(f"Error comparing datasets: {e}")
            return 0.0



