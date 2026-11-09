import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

import torch
import torch.nn as nn

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# -------------------------------------------------------------
# PyTorch Deep MLP Architecture Definition
# -------------------------------------------------------------
class DeepMLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.25):
        super(DeepMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.network(x)

# Global variables for model and artifacts
model = None
artifacts = None
feature_cols = None
scaler = None

def load_system():
    global model, artifacts, feature_cols, scaler
    artifacts_path = os.path.join("models", "artifacts.pkl")
    model_path = os.path.join("models", "credinet_deep_mlp.pth")
    
    if os.path.exists(artifacts_path) and os.path.exists(model_path):
        artifacts = joblib.load(artifacts_path)
        feature_cols = artifacts["feature_cols"]
        scaler = artifacts["scaler"]
        input_dim = artifacts["input_dim"]
        
        model = DeepMLP(input_dim, dropout_rate=0.20)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print("[SUCCESS] CrediNet PyTorch Deep MLP Model & Scaler loaded successfully!")
    else:
        print("[WARNING] Model weights or artifacts not found yet. Please run train_models.py first.")

# Try initial load if files exist
load_system()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, artifacts, feature_cols, scaler
    if model is None or artifacts is None:
        load_system()
        if model is None:
            return jsonify({"error": "PyTorch Model not initialized. Please ensure model is trained."}), 500

    try:
        data = request.get_json(force=True)

        # Extract values
        gender = 1 if data.get("Gender") == "Male" else 0
        married = 1 if data.get("Married") == "Yes" else 0
        dependents = int(data.get("Dependents", 0))
        education = 1 if data.get("Education") == "Graduate" else 0
        self_employed = 1 if data.get("Self_Employed") == "Yes" else 0
        app_income = float(data.get("ApplicantIncome", 0))
        coapp_income = float(data.get("CoapplicantIncome", 0))
        loan_amount = float(data.get("LoanAmount", 0))
        term = float(data.get("Loan_Amount_Term", 360))
        credit_history = float(data.get("Credit_History", 1.0))
        prop_area = data.get("Property_Area", "Urban")

        # Computed features
        total_income = app_income + coapp_income
        safe_term = term if term > 0 else 360
        emi = round((loan_amount * 1000) / safe_term, 2)
        balance_income = total_income - (emi * 1.5)
        dti_ratio = round((emi / total_income * 100), 1) if total_income > 0 else 100.0

        # Construct dataframe row matching feature_cols exactly
        row_dict = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": app_income,
            "CoapplicantIncome": coapp_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": term,
            "Credit_History": credit_history,
            "Total_Income": total_income,
            "EMI": emi,
            "Balance_Income": balance_income,
            "Property_Area_Rural": 1 if prop_area == "Rural" else 0,
            "Property_Area_Semiurban": 1 if prop_area == "Semiurban" else 0,
            "Property_Area_Urban": 1 if prop_area == "Urban" else 0,
        }

        # Build ordered DataFrame
        df_input = pd.DataFrame([row_dict])[feature_cols]

        # Standard Scale
        scaled_input = scaler.transform(df_input)
        tensor_input = torch.tensor(scaled_input, dtype=torch.float32)

        # PyTorch Inference
        model.eval()
        with torch.no_grad():
            logits = model(tensor_input)
            probability = float(torch.sigmoid(logits).item())

        is_approved = probability >= 0.50
        status = "APPROVED" if is_approved else "REJECTED"

        if probability >= 0.75:
            risk_tier = "LOW"
        elif probability >= 0.50:
            risk_tier = "MODERATE"
        else:
            risk_tier = "HIGH"

        # Explainability Decision Factors
        factors = []
        if credit_history == 1.0:
            factors.append("Clean credit history (Verified CIB Record) strongly elevates approval confidence.")
        else:
            factors.append("CIB Record Alert: Unverified or past default history under this NID heavily penalizes the risk profile.")

        if dti_ratio <= 35:
            factors.append(f"Debt-to-Income (DTI) ratio is healthy at {dti_ratio}%, well below standard 40% cap.")
        elif dti_ratio <= 50:
            factors.append(f"DTI ratio is moderate ({dti_ratio}%); sustainable with household income.")
        else:
            factors.append(f"High debt servicing burden: Monthly EMI consumes {dti_ratio}% of monthly income.")

        if coapp_income > 0:
            factors.append(f"Additional co-applicant income (৳ {int(coapp_income):,}) improves loan repayment security.")

        if education == 1:
            factors.append("Graduate educational qualification positively influences long-term earning stability.")

        return jsonify({
            "loan_status": status,
            "approval_probability": round(probability, 4),
            "risk_tier": risk_tier,
            "financial_summary": {
                "total_income": int(total_income),
                "estimated_emi": int(emi),
                "dti_ratio": dti_ratio,
                "loan_amount_dollars": int(loan_amount * 1000)
            },
            "decision_factors": factors
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/metrics')
def get_metrics():
    metrics_path = os.path.join("models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"error": "Metrics not found. Run training script first."}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*60)
    print("  CrediNet: AI-Powered Commercial Banking & Credit Risk Portal")
    print("  Student: Md. Mahfuz | ID: 0432310005101057 | CSE 422")
    print(f"  URL: http://0.0.0.0:{port}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
