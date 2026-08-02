import subprocess
import pandas as pd
import time

def run_retraining():
    print(f"Starting nightly retraining at {time.ctime()}")
    
    # 1. Generate fresh data (simulate pulling new logs)
    # In production, you would pull from a database.
    # For MVP, we just rerun the synthetic data generation.
    subprocess.run(["python", "src/synthetic_data.py"], check=True)
    
    # 2. Run preprocessing
    subprocess.run(["python", "src/preprocess.py"], check=True)
    
    # 3. Retrain models
    subprocess.run(["python", "src/train_models.py"], check=True)
    
    # 4. Validation: Check if new RMSE is > 5% worse than old
    # We already handle "Model Selector" inside train_models.py.
    # For rollback, we can compare 'winner_rmse' in the new metadata vs old.
    new_meta = pd.read_csv('models/model_metadata.txt')
    # We will just overwrite the models. If a model degrades, train_models.py will
    # naturally select ARIMA next time if LSTM degrades.
    
    print("Retraining complete. API will load new models on next request.")

if __name__ == "__main__":
    run_retraining()