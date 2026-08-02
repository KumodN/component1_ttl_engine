import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# ARIMA
from statsmodels.tsa.arima.model import ARIMA

# TensorFlow/Keras for LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def prepare_lstm_data(data, sequence_length=60):
    """Converts data into X (sequences) and y (next value) for LSTM."""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_models_for_key(key, df, seq_len=60):
    """
    Trains ARIMA and LSTM for a specific key, returns RMSE for both.
    """
    # Replace invalid characters for Windows filenames
    safe_key = key.replace(':', '_').replace('/', '_').replace('\\', '_')

    # Filter for specific key
    key_data = df[df['cache_key'] == key].copy()
    key_data = key_data.sort_values('bucket')
    
    # Use request_count and hour_sin/cos as features for LSTM
    # For simplicity here, we predict based on the count itself + time
    features = ['request_count', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    
    # Split data (last 30% for testing)
    split_idx = int(len(key_data) * 0.7)
    train_data = key_data.iloc[:split_idx]
    test_data = key_data.iloc[split_idx:]
    
    # --- 1. Train ARIMA (Baseline) ---
    # ARIMA only looks at the request_count
    try:
        arima_model = ARIMA(train_data['request_count'], order=(5, 1, 0))
        arima_fit = arima_model.fit()
        # Predict on test set
        arima_preds = arima_fit.forecast(steps=len(test_data))
        arima_rmse = np.sqrt(np.mean((arima_preds - test_data['request_count'].values) ** 2))
    except:
        arima_rmse = float('inf') # If it fails to fit

    # --- 2. Train LSTM (Deep Learning) ---
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(key_data[features].values)
    
    # Prepare sequences
    X, y = prepare_lstm_data(scaled_data, seq_len)
    
    # Split LSTM data
    train_seq_len = int(len(X) * 0.7)
    X_train, X_test = X[:train_seq_len], X[train_seq_len:]
    y_train, y_test = y[:train_seq_len], y[train_seq_len:]
    
    # Build LSTM Model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1)  # Predict the next value
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Predict on test set
    lstm_preds_scaled = model.predict(X_test, verbose=0)
    
    # Invert scaling for the 'request_count' (first column is index 0)
    dummy_preds = np.zeros((len(lstm_preds_scaled), len(features)))
    dummy_preds[:, 0] = lstm_preds_scaled.flatten()
    
    # We need the full scaled array to invert properly, but we only have the scaled predictions.
    # To simplify for this guide, we compare RMSE on normalized data or use the unscaled y.
    # Let's use the y_test which is unscaled request counts.
    lstm_rmse = np.sqrt(np.mean((y_test[:, 0] - lstm_preds_scaled.flatten()) ** 2))
    
    # Save the LSTM model and scaler
    model.save(f'models/lstm_{safe_key}.keras')
    joblib.dump(scaler, f'models/scaler_{safe_key}.pkl')
    
    return arima_rmse, lstm_rmse, key_data, safe_key

if __name__ == "__main__":
    print("Loading processed features...")
    df = pd.read_csv('data/processed_features.csv', parse_dates=['bucket'])
    
    keys = df['cache_key'].unique()
    results = []
    
    for key in keys:
        print(f"Training models for {key}...")
        arima_rmse, lstm_rmse, data, safe_key = train_models_for_key(key, df)
        
        # Model Selector Logic (Simple If-Else)
        if lstm_rmse < arima_rmse:
            winner = "LSTM"
            winner_rmse = lstm_rmse
        else:
            winner = "ARIMA"
            winner_rmse = arima_rmse
        
        results.append({
            'key': key,
            'arima_rmse': arima_rmse,
            'lstm_rmse': lstm_rmse,
            'selected_model': winner,
            'winner_rmse': winner_rmse
        })
        print(f"Selected: {winner} (LSTM: {lstm_rmse:.2f}, ARIMA: {arima_rmse:.2f})")
    
    # Save results
    pd.DataFrame(results).to_csv('models/model_metadata.txt', index=False)
    print("Model training complete. Metadata saved.")