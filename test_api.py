import json
import urllib.request

payload = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 8500,
    "CoapplicantIncome": 3200,
    "LoanAmount": 120,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
}

req = urllib.request.Request(
    "http://127.0.0.1:5000/predict",
    data=json.dumps(payload).encode('utf-8'),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        res_data = json.loads(resp.read().decode())
        print("API Response for Prime Candidate:")
        print(json.dumps(res_data, indent=2))
except Exception as e:
    print("Error calling test API:", e)
