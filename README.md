# CrediNet: Automated Loan Approval and Credit Risk Assessment Using Multi-Layer Perceptrons

**Course Title:** Neural Networks and Deep Learning (CSE 422)  
**Student Name:** Md. Mahfuz  
**Student ID:** 0432310005101057  

---

## 🏦 Project Overview
CrediNet is an automated, AI-driven commercial loan eligibility and credit risk evaluation platform. Powered by **PyTorch Deep Multi-Layer Perceptrons (MLP)**, it assesses financial profiles, debt burdens, and demographic indicators to evaluate default risk and provide instant loan decisions (`APPROVED` or `REJECTED`) with granular confidence scores.

To satisfy the requirements of CSE 422, the system benchmarks a **Baseline Shallow Neural Network** against a regularized **Deep MLP Architecture** featuring **Batch Normalization**, **Dropout**, and **Early Stopping** to mitigate overfitting.

---

## 🚀 Key Features

1. **Commercial Banking Web Portal (Flask + Tailwind CSS):**
   - High-fidelity FinTech UI styled like modern commercial bank portals (Stripe, Mercury, Revolut).
   - One-Click **Preset Buttons** for live viva demonstrations:
     - 🟢 **Preset A (Prime Applicant):** High income, graduate, clean credit history &rarr; Approved with Low Risk.
     - 🔴 **Preset B (High Risk Defaulter):** Past credit default, high loan request &rarr; Rejected with High Risk.
     - 🟡 **Preset C (Moderate Risk):** Average income with co-applicant &rarr; Evaluated conditionally.
   - Dynamic Financial Metrics calculation:
     - Total Household Income
     - Monthly EMI estimate
     - Debt-to-Income (DTI) ratio
   - Explainable Neural Decision factors.

2. **CSE 422 Lab Defense Analytics Tab:**
   - Real-time comparison table between **Shallow MLP** and **Deep MLP**.
   - Performance metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
   - Interactive high-res plots:
     - Training vs. Validation Loss curves (verifying convergence and early stopping).
     - Confusion Matrix heatmap.
     - ROC Curve comparison.

3. **PyTorch Deep Architecture Details:**
   - **Input Layer:** 16 processed features (including one-hot encoded locations and engineered financial features).
   - **Hidden Layer 1:** `Linear(16, 64) -> BatchNorm1d(64) -> ReLU -> Dropout(0.20)`
   - **Hidden Layer 2:** `Linear(64, 32) -> BatchNorm1d(32) -> ReLU -> Dropout(0.20)`
   - **Hidden Layer 3:** `Linear(32, 16) -> ReLU`
   - **Output Layer:** `Linear(16, 1) -> BCEWithLogitsLoss()`
   - **Optimizer:** `Adam(lr=0.001, weight_decay=1e-4)`

---

## 📁 Project Structure

```
d:\6th Semester\NN lab\
│
├── data/
│   └── loan_data.csv            # Standard Loan Prediction Dataset (614 records)
│
├── models/
│   ├── credinet_deep_mlp.pth    # PyTorch Deep MLP trained weights
│   ├── credinet_shallow_mlp.pth # PyTorch Shallow MLP baseline weights
│   ├── artifacts.pkl            # Preprocessing metadata & StandardScaler
│   └── metrics.json             # Test set evaluation metrics
│
├── static/
│   └── plots/
│       ├── loss_curve.png       # Training vs Validation loss curves
│       ├── confusion_matrix.png # Deep MLP confusion matrix
│       └── roc_curve.png        # ROC curves (Shallow vs Deep)
│
├── templates/
│   └── index.html               # Modern FinTech Web Portal UI
│
├── venv/                        # Virtual environment (on D: drive)
├── app.py                       # Flask Web Application & Inference Engine
├── train_models.py              # PyTorch training, evaluation & plot generator
├── download_dataset.py          # Script to fetch raw dataset
├── run_app.bat                  # One-click Windows runner
└── README.md                    # Project Documentation
```

---

## 💻 How to Run the Project

### Option A: Using the One-Click Batch Script
Simply double-click `run_app.bat` inside the folder. It will:
1. Verify the trained PyTorch model (and train if not yet done).
2. Launch the Flask web server.
3. Open your browser at `http://127.0.0.1:5000`.

### Option B: Using the Command Line
```powershell
# 1. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 2. Train the models and generate plots
python train_models.py

# 3. Start the Web Portal
python app.py
```
Open **http://127.0.0.1:5000** in Chrome/Edge.

---

## 🎓 Viva & Lab Presentation Tips for Md. Mahfuz

When presenting to the teacher:
1. **Explain the Architecture:** Point to the **Deep MLP Architecture** tab to show why you added `BatchNorm1d` (to stabilize mini-batch gradient updates) and `Dropout(0.2)` (to eliminate reliance on single features, effectively acting as an ensemble).
2. **Explain Early Stopping:** Explain that training monitors validation loss with a patience parameter (e.g., 20-30 epochs), halting training before the model memorizes noise.
3. **Demonstrate Live Presets:** Use the Preset buttons on the portal to show the model making real-time, logical banking predictions in less than 10 milliseconds.
