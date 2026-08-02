import pandas as pd
import numpy as np
from collections import deque
import time

class VolatilityEngine:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.volatilities = {}
        
    def update(self, key, current_time, db_update_df):
        """Calculate the EWMA volatility for a key based on update frequency."""
        # Count updates in the last hour for this key
        one_hour_ago = current_time - pd.Timedelta(hours=1)
        recent_updates = db_update_df[
            (db_update_df['cache_key'] == key) & 
            (db_update_df['timestamp'] >= one_hour_ago)
        ]
        updates_per_min = len(recent_updates) / 60.0  # Normalized
        
        # Cap it at 1.0
        current_rate = min(updates_per_min * 10, 1.0)
        
        # EWMA Formula: Vol(t) = alpha * current + (1-alpha) * Vol(t-1)
        if key not in self.volatilities:
            self.volatilities[key] = current_rate
        else:
            self.volatilities[key] = (self.alpha * current_rate) + ((1 - self.alpha) * self.volatilities[key])
        
        # Ensure it stays between 0 and 1
        return max(0.0, min(1.0, self.volatilities[key]))

class HotKeyDetector:
    def __init__(self, threshold=1000, window_seconds=60):
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.requests = {}  # key -> deque of timestamps
        
    def record_request(self, key):
        current_time = time.time()
        if key not in self.requests:
            self.requests[key] = deque(maxlen=10000)  # Max size to prevent memory blow
        
        # Remove old timestamps
        cutoff = current_time - self.window_seconds
        while self.requests[key] and self.requests[key][0] < cutoff:
            self.requests[key].popleft()
        
        # Add new request
        self.requests[key].append(current_time)
        
        # Check threshold
        count = len(self.requests[key])
        return count >= self.threshold