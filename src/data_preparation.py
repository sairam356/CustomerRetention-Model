import pandas as pd
import numpy as np
import os

def generate_data():
    print("Generating synthetic customer churn data...")
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'tenure': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    raw_path = os.path.join('data', 'raw', 'customer_data.csv')
    df.to_csv(raw_path, index=False)
    print(f"Raw data saved to {raw_path}")
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    # One-hot encoding for contract_type
    df = pd.get_dummies(df, columns=['contract_type'])
    
    processed_path = os.path.join('data', 'processed', 'processed_data.csv')
    df.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")

if __name__ == "__main__":
    df = generate_data()
    preprocess_data(df)
