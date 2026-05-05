import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import joblib

def train():
    # Load data
    data_path = os.path.join('data', 'processed', 'processed_data.csv')
    if not os.path.exists(data_path):
        print("Data not found. Run data_preparation.py first.")
        return
        
    df = pd.read_csv(data_path)
    X = df.drop(['customer_id', 'churn'], axis=1)
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameters
    n_estimators = 100
    max_depth = 5
    
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.4f}")
    
    # Save model locally
    model_dir = os.path.join("model", "artifacts")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    
    joblib.dump(model, model_path)
    print(f"Model saved locally to {model_path}")

if __name__ == "__main__":
    train()
