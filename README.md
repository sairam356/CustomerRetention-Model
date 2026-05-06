# 📊 Customer Retention Prediction Model

![MLOps Workflow](https://img.shields.io/badge/MLOps-End--to--End-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.136.1-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.8.0-orange)

An end-to-end MLOps project designed to predict customer retention risk. By identifying high-risk customers, businesses can proactively apply retention strategies to reduce churn and maximize customer lifetime value.

---

## 🏗️ Architecture & Workflow

Based on the industry-standard MLOps lifecycle:
1.  **Dataset Preparation**: Synthetic data generation and feature engineering (One-hot encoding).
2.  **Model Training**: Training a Random Forest Classifier with Scikit-Learn.
3.  **Artifact Management**: Versioned model storage in `model/artifacts/` using `joblib`.
4.  **Model Serving**: Real-time inference via a **FastAPI** REST endpoint.
5.  **Deployment**: Production-ready **Docker** container and **Kubernetes** manifests (Deployment, Service, HPA).

---

## 📂 Project Structure

```text
├── data/
│   ├── raw/                # Original synthetic customer data
│   └── processed/          # Cleaned data ready for training
├── src/
│   ├── data_preparation.py # ETL script for data cleaning
│   ├── train.py            # Model training & evaluation logic
│   └── app.py              # FastAPI application for serving
├── model/
│   └── artifacts/          # Serialized model.joblib
├── k8s/
│   └── deployment.yaml     # Kubernetes orchestration manifests
├── Dockerfile              # Containerization instructions
└── requirements.txt        # Python dependencies
```

---

## 🚀 Getting Started

### 1. Setup Environment
```bash
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
# Step 1: Prepare the data
python src/data_preparation.py

# Step 2: Train the model
python src/train.py
```

### 3. Start the API
```bash
uvicorn src.app:app --host 127.0.0.1 --port 8006
```

---

## 🧪 API Usage

### **Predict Retention Risk**
**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "tenure": 12,
  "monthly_charges": 70.5,
  "total_charges": 840.0,
  "contract_type_Month_to_month": true,
  "contract_type_One_year": false,
  "contract_type_Two_year": false
}
```

**Success Response:**
```json
{
  "retention_risk_prediction": 0,
  "retention_risk_probability": 0.233
}
```

---

## 📈 Features Used
*   **Tenure**: Duration of customer relationship.
*   **Monthly Charges**: Current monthly billing amount.
*   **Contract Type**: Month-to-month, One-year, or Two-year.

---
*Created by [sairam356](https://github.com/sairam356)*
*Contact: +91 9778967890*
