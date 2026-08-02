import numpy as np

class TTLFusionEngine:
    def __init__(self, min_ttl=10, max_ttl=3600):
        """
        min_ttl: Minimum TTL in seconds (e.g., 10s)
        max_ttl: Maximum TTL in seconds (e.g., 3600s = 1 hour)
        """
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
    
    def calculate_ttl(self, popularity_score, volatility_score, is_hot):
        """
        popularity_score: float 0.0 to 1.0 (from LSTM/ARIMA)
        volatility_score: float 0.0 to 1.0 (from EWMA)
        is_hot: boolean (True if this key is experiencing a traffic storm)
        """
        # Override for Hot-Keys
        if is_hot:
            return self.max_ttl
        
        # Base TTL: Popularity determines the baseline
        base_ttl = self.max_ttl * popularity_score
        
        # Volatility Penalty: Higher volatility reduces TTL
        # Formula: 1 / (1 + Vol*10) => Vol=0 -> 1.0, Vol=0.5 -> 0.16
        penalty = 1 / (1 + (volatility_score * 10))
        
        final_ttl = base_ttl * penalty
        
        # Clamp between min and max
        return max(self.min_ttl, min(self.max_ttl, final_ttl))