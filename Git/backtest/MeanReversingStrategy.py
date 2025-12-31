
from AbstractStrategy import GenericStrategy

import pandas as pd
from datetime import datetime
import bisect

class MeanReversingStrategy(GenericStrategy):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_filter = self.kwargs.get('extra_filter', False)
        self._cache = {}  # Cache per dataset id

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

        # Fetch current RSI
        current_rsi = data.get(current_key, {}).get("RSI12")
        if current_rsi is None:
            return False

        # Get last 15 RSI values (already sorted)
        start_idx = max(0, idx - 20)
        prev_rsis = [
            data[ts_map[t]].get("RSI12")
            for t in ts_only[start_idx:idx]
            if data.get(ts_map[t], {}).get("RSI12") is not None
        ]

        # Decision rule
        return current_rsi < 22 and not any(rsi < 15 for rsi in prev_rsis)

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
        current_rsi = data.get(current_key, {}).get("RSI12")

        return current_rsi is not None and current_rsi > 66



