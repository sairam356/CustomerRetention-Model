from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Customer Retention Prediction API")

# Define the input data model
class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_type_Month_to_month: bool
    contract_type_One_year: bool
    contract_type_Two_year: bool

model = None

def load_model():
    global model
    try:
        local_model_path = os.path.join("model", "artifacts", "model.joblib")
        if os.path.exists(local_model_path):
            model = joblib.load(local_model_path)
            print(f"Loaded model from {local_model_path}")
        else:
            print(f"No model found at {local_model_path}. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.on_event("startup")
def startup_event():
    load_model()

@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Map Pydantic fields to the exact feature names seen during fit
    input_data = {
        "tenure": data.tenure,
        "monthly_charges": data.monthly_charges,
        "total_charges": data.total_charges,
        "contract_type_Month-to-month": data.contract_type_Month_to_month,
        "contract_type_One year": data.contract_type_One_year,
        "contract_type_Two year": data.contract_type_Two_year
    }
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    return {
        "retention_risk_prediction": int(prediction[0]), # 1 means at risk
        "retention_risk_probability": float(probability[0][1])
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Customer Retention Prediction API"}
