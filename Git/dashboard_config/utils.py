import streamlit as st
import json
import pandas as pd
from datetime import datetime as dt


# Available indicators with their parameter configurations
INDICATOR_FUNCTIONS = {
    "sma": {"name": "Simple Moving Average", "params": ["window"]},
    "ema": {"name": "Exponential Moving Average", "params": ["window"]},
    "rsi": {"name": "Relative Strength Index", "params": ["window"]},
    "atr": {"name": "Average True Range", "params": ["window"]},
    "vtxp": {"name": "Vortex Indicator Positive", "params": ["window"]},
    "vtxn": {"name": "Vortex Indicator Negative", "params": ["window"]},
    "roll_atr": {"name": "Rolling Average True Range", "params": ["window"]},
    "vtxspread": {"name": "Vortex Spread", "params": ["window"]},
    "roll_vol": {"name": "Rolling Volume", "params": ["window"]},
    "macd": {"name": "MACD", "params": ["window_slow", "window_fast"]},
    "bband": {"name": "Bollinger Bands", "params": ["window", "std_dev"]},
    "stoch_osc": {"name": "Stochastic Oscillator", "params": ["window", "smooth_k", "smooth_d"]}
}


def initialize_session_state():
    if 'pairs_config' not in st.session_state:
        st.session_state['pairs_config'] = {}
    if 'current_pair_index' not in st.session_state:
        st.session_state['current_pair_index'] = 0


def get_default_pair_config():
    return {
        "pair_name": "",
        "balance_currency": "USDT",
        "spot_currency": "",
        "allocated_weight": 0.33,
        "data_aggregation_period": "3min",
        "data_aggregation_period_stoploss": "1min",
        "stop_loss": 0.975,
        "adaptative_stop_loos": 0.25,
        "take_profit": {
            "T1": {"Level": 1.25, "Order_weight": 0.25},
            "T2": {"Level": 1.5, "Order_weight": 0.25},
            "T3": {"Level": 1.75, "Order_weight": 0.5}
        },
        "leverage": 1,
        "extra_filter": False,
        "indicators": []
    }


def render_configuration_form():
    st.header("Global Strategy Configuration")

    col1, col2 = st.columns(2)
    with col1:
        strategy_class = st.selectbox(
            "Strategy Class",
            options=["MeanReversingStrategy", "MomentumStrategy", "BreakoutStrategy"],
            help="The strategy implementation class"
        )
    with col2:
        strategy_module = st.text_input(
            "Strategy Module",
            value="Meanreversing_strategy",
            help="Python module containing the strategy class"
        )

    col3, col4 = st.columns(2)
    with col3:
        start_date = st.date_input(
            "Start Date",
            value=dt(2025, 1, 1),
            help="Strategy start date"
        )
    with col4:
        end_date = st.date_input(
            "End Date",
            value=dt(2025, 10, 1),
            help="Strategy end date"
        )

    if 'global_config' not in st.session_state:
        st.session_state['global_config'] = {
            'class': strategy_class,
            'module': strategy_module,
            'start_date': f"{start_date} 00:00",
            'end_date': f"{end_date} 00:00"
        }
    else:
        st.session_state['global_config'].update({
            'class': strategy_class,
            'module': strategy_module,
            'start_date': f"{start_date} 00:00",
            'end_date': f"{end_date} 00:00"
        })

    st.header("Trading Pairs Configuration")

    pairs_list = list(st.session_state['pairs_config'].keys()) if st.session_state['pairs_config'] else []

    col_select, col_input = st.columns([1, 1])
    with col_select:
        selected_pair_to_edit = st.selectbox(
            "Select Existing Pair",
            options=pairs_list,
            help="Choose an existing trading pair configuration to edit",
            key="pair_to_edit_select"
        )
    with col_input:
        new_pair_input = st.text_input(
            "Or Enter New Pair Name",
            placeholder="e.g., BTCUSDT",
            key="pair_to_edit_input",
            help="Enter a new pair name if not listed above"
        )

    col_add, col_del = st.columns([1, 1])
    with col_add:
        if st.button("Add", type="primary", help="Add a new trading pair", use_container_width=True):
            new_pair_name = new_pair_input
            st.session_state['pairs_config'][new_pair_name] = get_default_pair_config()
            st.rerun()
    with col_del:
        if st.button("Delete", type="primary", help="Delete the selected pair", use_container_width=True):
            if selected_pair_to_edit:
                del st.session_state['pairs_config'][selected_pair_to_edit]
                st.rerun()

    if pairs_list and selected_pair_to_edit:
        render_pair_config_form(selected_pair_to_edit)
    else:
        st.info("No pairs configured yet. Click 'Add New Pair' to start.")

    if pairs_list:
        if st.button("Save All Configurations", type="primary"):
            global_config = st.session_state['global_config']
            pairs_config = st.session_state['pairs_config']

            final_config = {}
            for pair_name, pair_config in pairs_config.items():
                final_config[pair_name] = {
                    **global_config,
                    **pair_config
                }
            st.session_state['final_strategy_config'] = final_config
            st.success("All configurations saved! Go to Preview & Export to view and download.")


def render_pair_config_form(pair_name):
    if pair_name in st.session_state['pairs_config']:
        config = st.session_state['pairs_config'][pair_name].copy()
    else:
        config = get_default_pair_config()

    st.subheader(f"Configuration for: {pair_name}")

    with st.form(f"pair_form_{pair_name}"):
        col1, col2 = st.columns(2)
        with col1:
            pair_symbol = st.text_input(
                "Trading Pair Symbol",
                value=config["pair_name"],
                help="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)",
                key=f"pair_name_{pair_name}"
            )
            balance_currency = st.selectbox(
                "Balance Currency",
                options=["USDT", "BTC", "ETH", "BNB"],
                index=["USDT", "BTC", "ETH", "BNB"].index(config["balance_currency"]) if config["balance_currency"] in ["USDT", "BTC", "ETH", "BNB"] else 0,
                help="Base currency for the strategy",
                key=f"balance_currency_{pair_name}"
            )
            spot_currency = st.text_input(
                "Spot Currency",
                value=config["spot_currency"],
                help="The cryptocurrency being traded",
                key=f"spot_currency_{pair_name}"
            )
            allocated_weight = st.slider(
                "Allocated Weight",
                min_value=0.01,
                max_value=1.0,
                value=config["allocated_weight"],
                step=0.01,
                help="Portfolio allocation weight for this strategy",
                key=f"allocated_weight_{pair_name}"
            )
            leverage = st.number_input(
                "Leverage",
                min_value=1,
                max_value=125,
                value=config["leverage"],
                help="Trading leverage",
                key=f"leverage_{pair_name}"
            )
        with col2:
            data_agg_options = ["1min", "3min", "5min", "15min", "30min", "1h", "4h", "1d"]
            data_aggregation_period = st.selectbox(
                "Data Aggregation Period",
                options=data_agg_options,
                index=data_agg_options.index(config["data_aggregation_period"]) if config["data_aggregation_period"] in data_agg_options else 1,
                help="Primary data aggregation timeframe",
                key=f"data_aggregation_{pair_name}"
            )
            stoploss_agg_options = ["1min", "3min", "5min", "15min", "30min"]
            data_aggregation_period_stoploss = st.selectbox(
                "Stop Loss Data Aggregation Period",
                options=stoploss_agg_options,
                index=stoploss_agg_options.index(config["data_aggregation_period_stoploss"]) if config["data_aggregation_period_stoploss"] in stoploss_agg_options else 0,
                help="Data aggregation for stop loss",
                key=f"data_aggregation_stoploss_{pair_name}"
            )
            stop_loss = st.number_input(
                "Stop Loss (%)",
                min_value=0.8,
                max_value=0.99,
                value=config["stop_loss"],
                step=0.001,
                format="%.3f",
                help="Stop loss percentage (0.975 = 97.5%)",
                key=f"stop_loss_{pair_name}"
            )
            adaptative_stop_loss = st.number_input(
                "Adaptive Stop Loss",
                min_value=0.01,
                max_value=1.0,
                value=config["adaptative_stop_loos"],
                step=0.01,
                help="Adaptive stop loss parameter",
                key=f"adaptative_stop_loss_{pair_name}"
            )
            extra_filter = st.checkbox(
                "Extra Filter",
                value=config["extra_filter"],
                help="Enable additional filtering conditions",
                key=f"extra_filter_{pair_name}"
            )

        st.subheader("Take Profit Configuration")
        col_tp1, col_tp2, col_tp3 = st.columns(3)
        with col_tp1:
            st.write("**T1 Configuration**")
            t1_level = st.number_input(
                "T1 Level",
                min_value=1.0,
                max_value=3.0,
                value=config["take_profit"]["T1"]["Level"],
                step=0.05,
                help="Take profit level 1",
                key=f"t1_level_{pair_name}"
            )
            t1_weight = st.number_input(
                "T1 Order Weight",
                min_value=0.01,
                max_value=1.0,
                value=config["take_profit"]["T1"]["Order_weight"],
                step=0.01,
                help="Portion to sell at T1",
                key=f"t1_weight_{pair_name}"
            )
        with col_tp2:
            st.write("**T2 Configuration**")
            t2_level = st.number_input(
                "T2 Level",
                min_value=1.0,
                max_value=3.0,
                value=config["take_profit"]["T2"]["Level"],
                step=0.05,
                help="Take profit level 2",
                key=f"t2_level_{pair_name}"
            )
            t2_weight = st.number_input(
                "T2 Order Weight",
                min_value=0.01,
                max_value=1.0,
                value=config["take_profit"]["T2"]["Order_weight"],
                step=0.01,
                help="Portion to sell at T2",
                key=f"t2_weight_{pair_name}"
            )
        with col_tp3:
            st.write("**T3 Configuration**")
            t3_level = st.number_input(
                "T3 Level",
                min_value=1.0,
                max_value=3.0,
                value=config["take_profit"]["T3"]["Level"],
                step=0.05,
                help="Take profit level 3",
                key=f"t3_level_{pair_name}"
            )
            t3_weight = st.number_input(
                "T3 Order Weight",
                min_value=0.01,
                max_value=1.0,
                value=config["take_profit"]["T3"]["Order_weight"],
                step=0.01,
                help="Portion to sell at T3",
                key=f"t3_weight_{pair_name}"
            )

        if st.form_submit_button(f"Save {pair_name} Configuration", type="primary"):
            updated_config = {
                "pair_name": pair_symbol,
                "balance_currency": balance_currency,
                "spot_currency": spot_currency,
                "allocated_weight": allocated_weight,
                "data_aggregation_period": data_aggregation_period,
                "data_aggregation_period_stoploss": data_aggregation_period_stoploss,
                "stop_loss": stop_loss,
                "adaptative_stop_loos": adaptative_stop_loss,
                "take_profit": {
                    "T1": {"Level": t1_level, "Order_weight": t1_weight},
                    "T2": {"Level": t2_level, "Order_weight": t2_weight},
                    "T3": {"Level": t3_level, "Order_weight": t3_weight}
                },
                "leverage": leverage,
                "extra_filter": extra_filter
            }
            if f'indicators_{pair_name}' in st.session_state:
                updated_config["indicators"] = st.session_state[f'indicators_{pair_name}']
            else:
                updated_config["indicators"] = config.get("indicators", [])
            st.session_state['pairs_config'][pair_name] = updated_config
            st.success(f"Configuration for {pair_name} saved!")

    render_indicators_config(pair_name, config.get("indicators", []))


def render_indicators_config(pair_name, current_indicators):
    st.subheader("Indicators Configuration")
    if f'indicators_{pair_name}' not in st.session_state:
        st.session_state[f'indicators_{pair_name}'] = current_indicators.copy()
    indicators = st.session_state[f'indicators_{pair_name}']
    if indicators:
        indicators_data = []
        for i, indicator in enumerate(indicators):
            indicator_type = indicator.get('name', 'N/A')
            display_name = INDICATOR_FUNCTIONS.get(indicator_type, {}).get('name', indicator_type.upper())
            column_names = ', '.join(indicator.get('column_name', ['N/A']))
            params = indicator.get('params', {})
            params_str = ', '.join([f"{k}: {v}" for k, v in params.items()]) if params else 'None'
            indicators_data.append({
                'Indicator': display_name,
                'Type': indicator_type,
                'Column Name(s)': column_names,
                'Parameters': params_str,
                'Actions': i
            })
        if indicators_data:
            col_indicator, col_type, col_column, col_params, col_delete = st.columns([2, 1, 1.5, 2, 0.8])
            with col_indicator:
                st.write("**Indicator**")
            with col_type:
                st.write("**Type**")
            with col_column:
                st.write("**Column Name(s)**")
            with col_params:
                st.write("**Parameters**")
            with col_delete:
                st.write("**Delete**")
            st.markdown("---")
            for i, indicator_data in enumerate(indicators_data):
                col_indicator, col_type, col_column, col_params, col_delete = st.columns([2, 1, 1.5, 2, 0.8])
                with col_indicator:
                    st.write(indicator_data['Indicator'])
                with col_type:
                    st.write(indicator_data['Type'])
                with col_column:
                    st.write(indicator_data['Column Name(s)'])
                with col_params:
                    st.write(indicator_data['Parameters'])
                with col_delete:
                    if st.button("Delete", key=f"del_ind_{pair_name}_{i}", help=f"Delete {indicator_data['Indicator']}"):
                        indicators.pop(i)
                        st.session_state[f'indicators_{pair_name}'] = indicators
                        st.rerun()
    else:
        st.info("No indicators configured yet. Add your first indicator below.")

    st.markdown("---")
    st.write("**Add New Indicator**")
    with st.expander("Add New Indicator", expanded=False):
        col_ind_type, col_add = st.columns([3, 1])
        with col_ind_type:
            selected_indicator = st.selectbox(
                "Select Indicator Type",
                options=list(INDICATOR_FUNCTIONS.keys()),
                format_func=lambda x: f"{INDICATOR_FUNCTIONS[x]['name']} ({x})",
                help="Choose the indicator to add",
                key=f"new_indicator_type_{pair_name}"
            )
            indicator_info = INDICATOR_FUNCTIONS[selected_indicator]
            st.caption(f"**{indicator_info['name']}** - Parameters: {', '.join(indicator_info['params'])}")
        with col_add:
            if st.button("Configure", key=f"add_ind_{pair_name}", type="primary"):
                st.session_state[f'show_ind_params_{pair_name}'] = selected_indicator
                st.rerun()
    if f'show_ind_params_{pair_name}' in st.session_state:
        render_indicator_params(pair_name, st.session_state[f'show_ind_params_{pair_name}'])


def render_indicator_params(pair_name, indicator_type):
    indicator_info = INDICATOR_FUNCTIONS[indicator_type]
    params = indicator_info["params"]
    with st.container():
        st.markdown("---")
        st.subheader(f"Configure {indicator_info['name']}")
        if params:
            st.write("**Parameters:**")
            param_values = {}
            param_cols = st.columns(min(len(params), 3))
            for i, param in enumerate(params):
                col_idx = i % 3
                with param_cols[col_idx]:
                    if param == "window":
                        param_values[param] = st.number_input(
                            param.title(),
                            min_value=2,
                            max_value=200,
                            value=12,
                            key=f"param_{param}_{pair_name}_{indicator_type}",
                            help="Number of periods for calculation"
                        )
                    elif param == "window_slow":
                        param_values[param] = st.number_input(
                            param.replace("_", " ").title(),
                            min_value=10,
                            max_value=50,
                            value=26,
                            key=f"param_{param}_{pair_name}_{indicator_type}",
                            help="Slow moving average period"
                        )
                    elif param == "window_fast":
                        param_values[param] = st.number_input(
                            param.replace("_", " ").title(),
                            min_value=5,
                            max_value=20,
                            value=12,
                            key=f"param_{param}_{pair_name}_{indicator_type}",
                            help="Fast moving average period"
                        )
                    elif param == "std_dev":
                        param_values[param] = st.number_input(
                            param.replace("_", " ").title(),
                            min_value=1.0,
                            max_value=5.0,
                            value=2.0,
                            step=0.1,
                            key=f"param_{param}_{pair_name}_{indicator_type}",
                            help="Standard deviation multiplier"
                        )
                    elif param == "smooth_k":
                        param_values[param] = st.number_input(
                            param.replace("_", " ").title(),
                            min_value=1,
                            max_value=10,
                            value=3,
                            key=f"param_{param}_{pair_name}_{indicator_type}",
                            help="K period smoothing"
                        )
                    elif param == "smooth_d":
                        param_values[param] = st.number_input(
                            param.replace("_", " ").title(),
                            min_value=1,
                            max_value=10,
                            value=3,
                            key=f"param_{param}_{pair_name}_{indicator_type}",
                            help="D period smoothing"
                        )
                    else:
                        param_values[param] = st.number_input(
                            param.replace("_", " ").title(),
                            min_value=1,
                            max_value=100,
                            value=14,
                            key=f"param_{param}_{pair_name}_{indicator_type}",
                            help=f"{param.replace('_', ' ').title()} parameter"
                        )
        else:
            param_values = {}
            st.info("No parameters required for this indicator.")
        st.write("**Column Configuration:**")
        default_name = f"{indicator_type.upper()}{12 if 'window' in params else ''}"
        column_name = st.text_input(
            "Column Name",
            value=default_name,
            help="Column name for this indicator in the dataset",
            key=f"column_name_{pair_name}_{indicator_type}"
        )
        col_confirm, col_cancel, col_spacer = st.columns([1, 1, 2])
        with col_confirm:
            if st.button("Confirm Add", key=f"confirm_add_{pair_name}_{indicator_type}", type="primary"):
                new_indicator = {
                    "name": indicator_type,
                    "column_name": [column_name],
                    "params": param_values
                }
                if f'indicators_{pair_name}' not in st.session_state:
                    st.session_state[f'indicators_{pair_name}'] = []
                st.session_state[f'indicators_{pair_name}'].append(new_indicator)
                if f'show_ind_params_{pair_name}' in st.session_state:
                    del st.session_state[f'show_ind_params_{pair_name}']
                st.success(f"Added {indicator_info['name']} indicator!")
                st.rerun()
        with col_cancel:
            if st.button("Cancel", key=f"cancel_{pair_name}_{indicator_type}"):
                if f'show_ind_params_{pair_name}' in st.session_state:
                    del st.session_state[f'show_ind_params_{pair_name}']
                st.rerun()


def get_configured_indicators():
    """Get list of column names from configured indicators in Strategy Configuration"""
    configured_indicators = {}
    
    # Check all pairs for configured indicators
    if 'pairs_config' in st.session_state:
        for pair_name, pair_config in st.session_state['pairs_config'].items():
            indicators = pair_config.get('indicators', [])
            for indicator in indicators:
                indicator_name = indicator.get('name', '').upper()
                column_names = indicator.get('column_name', [])
                if indicator_name and column_names:
                    for col_name in column_names:
                        configured_indicators[col_name] = indicator_name
    
    # Also check individual indicator session states
    for key in st.session_state.keys():
        if key.startswith('indicators_') and st.session_state[key]:
            for indicator in st.session_state[key]:
                indicator_name = indicator.get('name', '').upper()
                column_names = indicator.get('column_name', [])
                if indicator_name and column_names:
                    for col_name in column_names:
                        configured_indicators[col_name] = indicator_name
    
    return configured_indicators


def render_conditions_config():
    """Render buy/sell conditions configuration"""
    st.header("Buy & Sell Conditions Configuration")
    
    # Initialize conditions in session state
    if 'buy_conditions' not in st.session_state:
        st.session_state['buy_conditions'] = []
    if 'sell_conditions' not in st.session_state:
        st.session_state['sell_conditions'] = []
    
    # Get configured indicators from Strategy Configuration
    configured_indicators = get_configured_indicators()
    
    if not configured_indicators:
        st.warning("⚠️ No indicators configured yet. Please configure indicators in the Strategy Configuration page first.")
        return
    
    # Extract column names for the dropdown
    available_columns = list(configured_indicators.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Buy Conditions")
        render_condition_section("buy", st.session_state['buy_conditions'], available_columns, configured_indicators)
    
    with col2:
        st.subheader("Sell Conditions")
        render_condition_section("sell", st.session_state['sell_conditions'], available_columns, configured_indicators)
    
    # Show available indicators info
    st.info(f"📊 **Available Column Names:** {', '.join(available_columns)}")
    
    # Save conditions button
    if st.button("Save All Conditions", type="primary"):
        st.session_state['final_conditions'] = {
            'buy_conditions': st.session_state['buy_conditions'],
            'sell_conditions': st.session_state['sell_conditions']
        }
        
        # Generate Python strategy file
        strategy_code = generate_strategy_file(st.session_state['buy_conditions'], st.session_state['sell_conditions'])
        
        # Save to file
        filename = "GeneratedStrategy.py"
        with open(filename, 'w') as f:
            f.write(strategy_code)
        
        st.success(f"All conditions saved! Strategy file generated: {filename}")
        st.download_button(
            label="Download Strategy File",
            data=strategy_code,
            file_name=filename,
            mime="text/x-python"
        )


def render_condition_section(condition_type, conditions, available_columns, configured_indicators):
    """Render a section for buy or sell conditions"""
    
    # Display existing conditions
    if conditions:
        for i, condition in enumerate(conditions):
            with st.expander(f"Condition {i+1}: {condition.get('description', 'No description')}", expanded=False):
                col_desc, col_del = st.columns([3, 1])
                with col_desc:
                    st.write(f"**Column:** {condition.get('column_name', 'N/A')}")
                    st.write(f"**Operator:** {condition.get('operator', 'N/A')}")
                    st.write(f"**Value:** {condition.get('value', 'N/A')}")
                    if condition.get('additional_conditions'):
                        st.write(f"**Additional:** {condition['additional_conditions']}")
                with col_del:
                    if st.button("Delete", key=f"del_{condition_type}_{i}"):
                        conditions.pop(i)
                        st.rerun()
    else:
        st.info(f"No {condition_type} conditions configured yet.")
    
    # Add new condition
    st.markdown("---")
    st.write(f"**Add New {condition_type.title()} Condition**")
    
    with st.form(f"add_{condition_type}_condition"):
        description = st.text_input(
            "Condition Description",
            placeholder=f"e.g., RSI oversold with no recent extreme",
            key=f"desc_{condition_type}"
        )
        
        column_name = st.selectbox(
            "Column Name",
            options=available_columns,
            help="Select from configured indicator column names",
            key=f"column_{condition_type}"
        )
        
        operator = st.selectbox(
            "Operator",
            options=["<", ">", "<=", ">=", "==", "!="],
            key=f"operator_{condition_type}"
        )
        
        value = st.number_input(
            "Value",
            value=50.0,
            step=0.1,
            key=f"value_{condition_type}"
        )
        
        # Additional conditions for complex logic
        additional_conditions = st.text_area(
            "Additional Conditions (Python code)",
            placeholder="e.g., and not any(rsi < 15 for rsi in prev_rsis)",
            help="Optional: Add complex logic using Python syntax",
            key=f"additional_{condition_type}"
        )
        
        if st.form_submit_button(f"Add {condition_type.title()} Condition"):
            new_condition = {
                "description": description,
                "column_name": column_name,
                "indicator_type": configured_indicators[column_name],
                "operator": operator,
                "value": value,
                "additional_conditions": additional_conditions if additional_conditions else None
            }
            conditions.append(new_condition)
            st.rerun()


def generate_strategy_file(buy_conditions, sell_conditions):
    """Generate a Python strategy file based on the configured conditions"""
    
    # Generate buy condition logic
    buy_logic = []
    for i, condition in enumerate(buy_conditions):
        column_name = condition.get('column_name', 'RSI12')
        operator = condition.get('operator', '<')
        value = condition.get('value', 22)
        additional = condition.get('additional_conditions', '')
        
        if i == 0:
            buy_logic.append(f"        # Fetch current {column_name}")
            buy_logic.append(f"        current_{column_name.lower()} = data.get(current_key, {{}}).get(\"{column_name}\")")
            buy_logic.append(f"        if current_{column_name.lower()} is None:")
            buy_logic.append(f"            return False")
            buy_logic.append(f"")
            buy_logic.append(f"        # {condition.get('description', 'Buy condition')}")
            buy_logic.append(f"        condition_result = current_{column_name.lower()} {operator} {value}")
            if additional:
                buy_logic.append(f"        {additional}")
        else:
            buy_logic.append(f"        # Additional condition: {condition.get('description', '')}")
            buy_logic.append(f"        condition_result = condition_result and current_{column_name.lower()} {operator} {value}")
            if additional:
                buy_logic.append(f"        {additional}")
    
    # Generate sell condition logic
    sell_logic = []
    for i, condition in enumerate(sell_conditions):
        column_name = condition.get('column_name', 'RSI12')
        operator = condition.get('operator', '>')
        value = condition.get('value', 66)
        additional = condition.get('additional_conditions', '')
        
        if i == 0:
            sell_logic.append(f"        # Fetch current {column_name}")
            sell_logic.append(f"        current_{column_name.lower()} = data.get(current_key, {{}}).get(\"{column_name}\")")
            sell_logic.append(f"        if current_{column_name.lower()} is None:")
            sell_logic.append(f"            return False")
            sell_logic.append(f"")
            sell_logic.append(f"        # {condition.get('description', 'Sell condition')}")
            sell_logic.append(f"        return current_{column_name.lower()} {operator} {value}")
            if additional:
                sell_logic.append(f"        {additional}")
        else:
            sell_logic.append(f"        # Additional condition: {condition.get('description', '')}")
            sell_logic.append(f"        return current_{column_name.lower()} {operator} {value}")
            if additional:
                sell_logic.append(f"        {additional}")
    
    # Generate the complete strategy file
    strategy_template = f'''from AbstractStrategy import GenericStrategy

import pandas as pd
from datetime import datetime
import bisect

class GeneratedStrategy(GenericStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_filter = self.kwargs.get('extra_filter', False)
        self._cache = {{}}  # Cache per dataset id

    def _prepare_cache(self, data: dict):
        """
        Prepare and cache sorted timestamps for faster lookup.
        """
        data_id = id(data)
        if data_id in self._cache:
            return self._cache[data_id]

        parsed_timestamps = []
        for ts in data.keys():
            if isinstance(ts, str):
                for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
                    try:
                        ts_dt = datetime.strptime(ts, fmt)
                        parsed_timestamps.append((ts_dt, ts))
                        break
                    except ValueError:
                        continue
            else:
                parsed_timestamps.append((ts, ts))

        parsed_timestamps.sort(key=lambda x: x[0])
        ts_only = [t[0] for t in parsed_timestamps]
        ts_map = dict(parsed_timestamps)

        self._cache[data_id] = (ts_only, ts_map)
        return ts_only, ts_map

    def buy_condition(self, data: dict, timestamp):
        ts_only, ts_map = self._prepare_cache(data)

        # Normalize current timestamp once
        if isinstance(timestamp, str):
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
                try:
                    timestamp_dt = datetime.strptime(timestamp, fmt)
                    break
                except ValueError:
                    continue
        else:
            timestamp_dt = timestamp

        # Binary search for the current candle
        idx = bisect.bisect_right(ts_only, timestamp_dt) - 1
        if idx < 0:
            return False

        current_dt = ts_only[idx]
        current_key = ts_map[current_dt]

{chr(10).join(buy_logic)}

        return condition_result

    def sell_condition(self, data: dict, timestamp):
        ts_only, ts_map = self._prepare_cache(data)

        # Normalize timestamp
        if isinstance(timestamp, str):
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
                try:
                    timestamp_dt = datetime.strptime(timestamp, fmt)
                    break
                except ValueError:
                    continue
        else:
            timestamp_dt = timestamp

        # Binary search for current candle
        import bisect
        idx = bisect.bisect_right(ts_only, timestamp_dt) - 1
        if idx < 0:
            return False

        current_dt = ts_only[idx]
        current_key = ts_map[current_dt]

{chr(10).join(sell_logic)}
'''
    
    return strategy_template


def render_preview_export():
    if 'final_strategy_config' not in st.session_state:
        st.warning("Please generate a configuration first in the Strategy Configuration tab.")
        return
    config = st.session_state['final_strategy_config']
    st.header("Strategy Configuration Preview")
    st.subheader("JSON Configuration")
    json_str = json.dumps(config, indent=2)
    st.code(json_str, language='json')
    
    # Display Buy & Sell Conditions Python Script
    if 'final_conditions' in st.session_state:
        st.subheader("Buy & Sell Conditions Python Script")
        buy_conditions = st.session_state['final_conditions'].get('buy_conditions', [])
        sell_conditions = st.session_state['final_conditions'].get('sell_conditions', [])
        
        if buy_conditions or sell_conditions:
            strategy_code = generate_strategy_file(buy_conditions, sell_conditions)
            st.code(strategy_code, language='python')
            
            # Download button for the strategy file
            st.download_button(
                label="Download Strategy Python File",
                data=strategy_code,
                file_name="GeneratedStrategy.py",
                mime="text/x-python"
            )
        else:
            st.info("No buy/sell conditions configured yet. Go to Buy & Sell Conditions page to configure them.")
    else:
        st.info("No buy/sell conditions configured yet. Go to Buy & Sell Conditions page to configure them.")
    
    st.subheader("Export Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Download JSON", type="primary"):
            json_str = json.dumps(config, indent=2)
            first_pair_name = list(config.keys())[0] if config else "strategy"
            filename = f"strategies_config_{first_pair_name}.json"
            st.download_button(
                label="Download strategies.json",
                data=json_str,
                file_name=filename,
                mime="application/json"
            )
    with col2:
        if st.button("Validate Configuration"):
            validation_results = []
            try:
                for pair_name, pair_config in config.items():
                    if pair_config.get('allocated_weight', 0) > 1.0:
                        validation_results.append(f"{pair_name}: Allocated weight cannot be greater than 1.0")
                    elif pair_config.get('stop_loss', 1) > 0.999 or pair_config.get('stop_loss', 1) < 0.5:
                        validation_results.append(f"{pair_name}: Stop loss should be between 0.5 and 0.999")
                    required_fields = ['pair_name', 'balance_currency', 'spot_currency']
                    for field in required_fields:
                        if not pair_config.get(field):
                            validation_results.append(f"{pair_name}: Missing required field '{field}'")
                if validation_results:
                    st.error("Validation errors found:")
                    for error in validation_results:
                        st.error(error)
                else:
                    st.success("All configurations are valid!")
            except Exception as e:
                st.error(f"Validation error: {str(e)}")
    with col3:
        if st.button("Reset All Configurations"):
            keys_to_remove = [
                'final_strategy_config', 'pairs_config', 'global_config',
                'current_pair_index'
            ]
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            for key in list(st.session_state.keys()):
                if key.startswith('indicators_') or key.startswith('show_ind_params_'):
                    del st.session_state[key]
            st.rerun()
    if config:
        st.subheader("Pairs Configuration Summary")
        pairs_data = []
        for pair_name, pair_config in config.items():
            pairs_data.append({
                'Pair Name': pair_name,
                'Trading Pair': pair_config.get('pair_name', 'N/A'),
                'Balance Currency': pair_config.get('balance_currency', 'N/A'),
                'Spot Currency': pair_config.get('spot_currency', 'N/A'),
                'Allocated Weight': f"{pair_config.get('allocated_weight', 0):.1%}",
                'Leverage': f"{pair_config.get('leverage', 1)}x",
                'Stop Loss': f"{(1-pair_config.get('stop_loss', 0.975))*100:.1f}%",
                'Indicators': len(pair_config.get('indicators', []))
            })
        df = pd.DataFrame(pairs_data)
        st.dataframe(df, use_container_width=True)


