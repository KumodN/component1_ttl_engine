import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_access_logs(days=30, keys=10):
    """Generates synthetic API request logs with weekly patterns."""
    start_date = datetime.now() - timedelta(days=days)
    timestamps = []
    key_names = []
    
    # Simulate different keys with different popularities
    for i in range(keys):
        # Create a base popularity (some keys are just more popular)
        base_pop = np.random.uniform(0.5, 1.5)
        for hour in range(24 * days):
            current_time = start_date + timedelta(hours=hour)
            
            # Simulate daily pattern: peaks at 9 AM and 2 PM
            daily_factor = 1 + 0.8 * np.exp(-((hour % 24 - 9) ** 2) / 10) + 0.6 * np.exp(-((hour % 24 - 14) ** 2) / 8)
            
            # Simulate weekly pattern: higher on weekdays
            weekday = current_time.weekday()
            weekly_factor = 1.5 if weekday < 5 else 0.4  # High Mon-Fri, low weekend
            
            # Calculate final request count (Poisson distribution)
            rate = base_pop * daily_factor * weekly_factor
            count = np.random.poisson(rate * 3)  # Requests per hour
            
            if count > 0:
                # Distribute the hour's requests evenly across the 60 minutes
                for _ in range(count):
                    minute_offset = np.random.randint(0, 60)
                    exact_time = current_time + timedelta(minutes=minute_offset)
                    timestamps.append(exact_time)
                    key_names.append(f"user:{i+1}:data")
    
    df = pd.DataFrame({'timestamp': timestamps, 'cache_key': key_names})
    df = df.sort_values('timestamp')
    return df

def generate_db_updates(days=30, keys=10):
    """Simulates database write logs (for volatility calculation)."""
    start_date = datetime.now() - timedelta(days=days)
    timestamps = []
    key_names = []
    
    for i in range(keys):
        # Some keys update frequently (volatile), some don't
        update_freq = np.random.choice([0.01, 0.05, 0.1, 0.5])  # Probability per hour
        for hour in range(24 * days):
            current_time = start_date + timedelta(hours=hour)
            if np.random.random() < update_freq:
                timestamps.append(current_time)
                key_names.append(f"user:{i+1}:data")
    
    df = pd.DataFrame({'timestamp': timestamps, 'cache_key': key_names})
    return df

if __name__ == "__main__":
    print("Generating synthetic access logs...")
    access_df = generate_access_logs(days=30, keys=10)
    access_df.to_csv('data/raw_logs.csv', index=False)
    
    print("Generating synthetic DB update logs...")
    db_df = generate_db_updates(days=30, keys=10)
    db_df.to_csv('data/db_updates.csv', index=False)
    
    print(f"Generated {len(access_df)} access logs and {len(db_df)} updates.")