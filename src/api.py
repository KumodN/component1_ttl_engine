from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import time
from datetime import datetime

# Import your custom modules
from volatility import VolatilityEngine, HotKeyDetector
from fusion_engine import TTLFusionEngine

app = Flask(__name__)

# --- Load Models & Metadata ---
print("Loading models and metadata...")
metadata = pd.read_csv('models/model_metadata.txt')
models = {}
scalers = {}
for _, row in metadata.iterrows():
    key = row['key']
    selected = row['selected_model']
    safe_key = key.replace(':', '_').replace('/', '_').replace('\\', '_')
    if selected == "LSTM":
        models[key] = load_model(f'models/lstm_{safe_key}.keras')
        scalers[key] = joblib.load(f'models/scaler_{safe_key}.pkl')
    else:
        # For ARIMA, we would load the saved ARIMA model here.
        # For this MVP, we simulate with a placeholder.
        models[key] = "ARIMA_Placeholder"

# --- Load Feature Data ---
# In production, this would be live. For now, we load the historical processed data.
feature_df = pd.read_csv('data/processed_features.csv', parse_dates=['bucket'])

# --- Initialize Engines ---
volatility_engine = VolatilityEngine(alpha=0.3)
hotkey_detector = HotKeyDetector(threshold=1000)
ttl_engine = TTLFusionEngine(min_ttl=10, max_ttl=3600)

# --- Helper Functions ---
def get_features_for_key(key, current_time):
    """
    Simulates getting the last 60 buckets (5 hours) of data for a key.
    In a real system, this fetches from Redis/InfluxDB.
    """
    # Filter for key and get last 60 rows
    data = feature_df[feature_df['cache_key'] == key].sort_values('bucket')
    if len(data) < 60:
        return None
    
    # Get last 60 records (historical data up to now)
    # In production, we would simulate real-time feeding. 
    # For this MVP, we just take the last batch.
    return data.iloc[-60:]

def predict_popularity(key, current_time):
    """
    Runs the selected model (LSTM or ARIMA) to predict future request count.
    Returns a normalized popularity score (0 to 1).
    """
    if key not in models:
        return 0.5  # Default if unknown key
    
    selected = metadata[metadata['key'] == key]['selected_model'].values[0]
    
    if selected == "LSTM":
        # Prepare features
        recent_data = get_features_for_key(key, current_time)
        if recent_data is None:
            return 0.5
        
        # Extract features used in training (must match training order)
        features = ['request_count', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        input_data = recent_data[features].values
        
        # Scale
        scaler = scalers[key]
        scaled_input = scaler.transform(input_data)
        
        # Reshape for LSTM (1 sample, 60 timesteps, 5 features)
        lstm_input = scaled_input.reshape(1, 60, 5)
        
        # Predict
        pred_scaled = models[key].predict(lstm_input, verbose=0)
        
        # Invert scaling (only for request_count)
        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = pred_scaled[0, 0]
        pred_count = scaler.inverse_transform(dummy)[0, 0]
        
        # Normalize to 0-1 (assuming max request count is 100 for this dataset)
        max_count = 150  # This can be tuned or derived from historical data
        return min(1.0, pred_count / max_count)
    
    else:  # ARIMA placeholder
        return 0.7

@app.route('/api/v1/predict/<cache_key>', methods=['POST'])
def predict_ttl(cache_key):
    # Get current time and simulate system load (optional)
    current_time = datetime.now()
    
    # 1. Get Volatility Score
    db_updates = pd.read_csv('data/db_updates.csv', parse_dates=['timestamp'])
    volatility_score = volatility_engine.update(cache_key, current_time, db_updates)
    
    # 2. Get Popularity Score
    popularity_score = predict_popularity(cache_key, current_time)
    
    # 3. Check if Hot Key
    # Simulate recording a request (in production, this is called by the sidecar)
    is_hot = hotkey_detector.record_request(cache_key)
    
    # 4. Calculate TTL
    suggested_ttl = ttl_engine.calculate_ttl(popularity_score, volatility_score, is_hot)
    
    # 5. Generate Confidence (simple logic: based on how much data we have)
    # If we have less than 100 records for this key, confidence is low
    key_data = feature_df[feature_df['cache_key'] == cache_key]
    if len(key_data) < 100:
        confidence = 0.3
    else:
        confidence = 0.9
    
    # 6. Response JSON
    response = {
        "cache_key": cache_key,
        "popularity": round(popularity_score, 3),
        "volatility": round(volatility_score, 3),
        "is_hot": is_hot,
        "suggested_ttl": int(suggested_ttl),
        "confidence": confidence,
        "model_version": "v1.0.0-MVP"
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)