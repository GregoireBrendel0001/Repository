

class GenericStrategy:
    """
    Abstract base class for all strategies.
    This class should be inherited by any specific strategy implementation.
    """

    def __init__(
            self,
            pair_name : str,
            allocated_weight : float,
            spot_currency : str,
            balance_currency : str,
            required_data_config : dict,
            required_data_depth : int,
            indicators : list,  
            tick_period : int,
            stoploss_tick_period : int,
            stop_loss : float,
            take_profit : float,
            **kwargs
        ):
        """
        Initialize the strategy.
        :param pair_name: The trading pair name (e.g., 'BTC/USD').
        :param allocated_weight: The weight allocated to this strategy.
        :param required_data_config: Configuration for the required data.
        :param required_data_depth: The depth of the required data.
        :param tick_period: The period for ticks in seconds.
        :param stop_loss: The stop loss percentage.
        :param take_profit: The take profit percentage.
        :param kwargs: Additional keyword arguments for the strategy.
        """
        self.pair_name = pair_name
        self.allocated_weight = allocated_weight
        self.required_data_config = required_data_config
        self.required_data_depth = required_data_depth
        self.indicators = indicators
        self.tick_period = tick_period
        self.stoploss_tick_period = stoploss_tick_period
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.spot_currency = spot_currency
        self.balance_currency = balance_currency
        self.kwargs = kwargs


    def buy_condition(self, **kwargs):
        """
        Determine if the strategy's buy condition is met.
        This method should be overridden by subclasses.

        :param args: Positional arguments for the condition check.
        :param kwargs: Keyword arguments for the condition check.
        :return: True if the buy condition is met, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def sell_condition(self, orders_history, **kwargs):
        """
        Determine if the strategy's sell condition is met.
        This method should be overridden by subclasses.

        :param args: Positional arguments for the condition check.
        :param kwargs: Keyword arguments for the condition check.
        :return: True if the sell condition is met, False otherwise.
        """
        

        raise NotImplementedError("Subclasses must implement this method.")
    
    def stoploss_condition(self, buy_value, current_value,**kwargs):
        """
        Determine if the strategy's stop loss condition is met.
        This method should be overridden by subclasses.

        :param args: Positional arguments for the condition check.
        :param kwargs: Keyword arguments for the condition check.
        :return: True if the stop loss condition is met, False otherwise.
        """
        if (buy_value - current_value) / buy_value >= self.stop_loss:
            return True
        return False
    
    def take_profit_condition(self, buy_value, current_value, **kwargs):
        """
        Determine if the strategy's take profit condition is met.
        This method should be overridden by subclasses.

        :param args: Positional arguments for the condition check.
        :param kwargs: Keyword arguments for the condition check.
        :return: True if the take profit condition is met, False otherwise.
        """
        if (current_value - buy_value) / buy_value >= self.take_profit:
            return True
        return False