import pandas as pd
import numpy as np
from datetime import datetime

def aggregate_access_logs(raw_df):
    """Group logs into 5-minute buckets and count requests per key."""
    df = raw_df.copy()
    # 1. Floor the timestamp to 5-minute intervals
    df['bucket'] = df['timestamp'].dt.floor('5min')
    
    # 2. Group by bucket and cache_key
    grouped = df.groupby(['bucket', 'cache_key']).size().reset_index(name='request_count')
    
    # 3. Fill missing buckets (for times with 0 requests)
    # Create a full time range
    all_buckets = pd.date_range(start=df['bucket'].min(), end=df['bucket'].max(), freq='5min')
    all_keys = df['cache_key'].unique()
    full_index = pd.MultiIndex.from_product([all_buckets, all_keys], names=['bucket', 'cache_key'])
    
    # Reindex and fill NaN with 0
    full_df = grouped.set_index(['bucket', 'cache_key']).reindex(full_index, fill_value=0).reset_index()
    return full_df

def create_features(df):
    """Add Cyclical (Sine/Cosine), Lag, and Rolling features."""
    df = df.copy()
    df['hour'] = df['bucket'].dt.hour
    df['day_of_week'] = df['bucket'].dt.dayofweek
    
    # Cyclical Encoding (The Circle Trick)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Sort to create lags
    df = df.sort_values(['cache_key', 'bucket'])
    
    # Lag Features (Short-term memory)
    df['t_minus_1'] = df.groupby('cache_key')['request_count'].shift(1)  # 5 mins ago
    df['t_minus_6'] = df.groupby('cache_key')['request_count'].shift(6)  # 30 mins ago
    df['t_minus_12'] = df.groupby('cache_key')['request_count'].shift(12) # 1 hour ago
    
    # Rolling Statistics
    df['rolling_mean_6'] = df.groupby('cache_key')['request_count'].rolling(6).mean().reset_index(level=0, drop=True)
    df['rolling_std_6'] = df.groupby('cache_key')['request_count'].rolling(6).std().reset_index(level=0, drop=True)
    
    # Fill initial NaN rows with 0 (since there is no history for first few rows)
    df = df.fillna(0)
    return df

if __name__ == "__main__":
    # Load raw data
    raw = pd.read_csv('data/raw_logs.csv', parse_dates=['timestamp'])
    
    # Aggregate
    print("Aggregating logs into 5-min buckets...")
    aggregated = aggregate_access_logs(raw)
    
    # Feature Engineering
    print("Creating features (Cyclical, Lags, Rolling)...")
    featured = create_features(aggregated)
    
    # Save
    featured.to_csv('data/processed_features.csv', index=False)
    print(f"Saved {len(featured)} rows of processed features.")