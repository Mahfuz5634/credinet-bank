# CrediNet: Automated Loan Approval & Credit Risk Assessment

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20MLP-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-CSS-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-85.37%25-brightgreen.svg)]()
[![F1-Score](https://img.shields.io/badge/F1--Score-90.11%25-green.svg)]()

> **Course:** CSE 422: Neural Networks and Deep Learning  
> **Author:** Md. Mahfuz  
> **Student ID:** `0432310005101057`  

---

## 🏦 Project Overview

**CrediNet** is an AI-powered commercial banking loan origination and risk evaluation system. Traditional loan approval processes in commercial banking often suffer from human error, slow manual file verification, and subjective bias. 

CrediNet solves this by training a **PyTorch Deep Multi-Layer Perceptron (Deep MLP)** on real-world loan datasets to accurately predict credit default risks in less than **10 milliseconds**. The model is wrapped in a clean, modern commercial banking portal (styled with Tailwind CSS) that provides real-time risk tiers, Debt-to-Income (DTI) health indicators, and transparent explainability factors for credit officers.

---

## 📊 Model Performance & Benchmark

We benchmarked our **Deep MLP** (with Batch Normalization, Dropout, and Early Stopping) against a baseline **Shallow Neural Network**:

| Performance Metric | Baseline Shallow MLP | CrediNet Deep MLP | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 82.11% | **85.37%** | **+3.26%** 🚀 |
| **Precision** | 82.80% | **83.67%** | **+0.87%** |
| **Recall** | 95.29% | **97.65%** | **+2.36%** |
| **F1-Score** | 88.61% | **90.11%** | **+1.50%** 🎯 |
| **ROC-AUC** | 76.53% | **83.00%** | **+6.47%** |

---

## 🧠 Neural Network Architecture

```
                 [ 16 Scaled Input Features ]
                             │
                             ▼
  ┌──────────────────────────────────────────────────────┐
  │  Layer 1: Linear(16, 64)                             │
  │  BatchNorm1d(64)   -> Stabilizes Gradient Descent    │
  │  ReLU Activation   -> Non-Linear Decision Boundary   │
  │  Dropout(0.20)     -> Mitigates Overfitting          │
  └──────────────────────────┬───────────────────────────┘
                             ▼
  ┌──────────────────────────────────────────────────────┐
  │  Layer 2: Linear(64, 32)                             │
  │  BatchNorm1d(32)   -> Normalizes Layer Dynamics      │
  │  ReLU Activation   -> Activation Transform           │
  │  Dropout(0.20)     -> Prevents Feature Co-adaptation │
  └──────────────────────────┬───────────────────────────┘
                             ▼
  ┌──────────────────────────────────────────────────────┐
  │  Layer 3: Linear(32, 16) -> ReLU                     │
  └──────────────────────────┬───────────────────────────┘
                             ▼
  ┌──────────────────────────────────────────────────────┐
  │  Output Layer: Linear(16, 1)                         │
  │  Loss: nn.BCEWithLogitsLoss() (Binary Cross Entropy) │
  │  Sigmoid -> Real-time Approval Probability [0.0, 1.0]│
  └──────────────────────────────────────────────────────┘
```

### Regularization Techniques Applied:
1. **Batch Normalization (`BatchNorm1d`):** Alleviates internal covariate shift, allowing faster learning rates and stabilizing mini-batch gradient updates.
2. **Dropout (`p=0.20`):** Randomly deactivates 20% of neuron activations during forward passes, preventing over-reliance on individual features.
3. **Early Stopping (`Patience=30`):** Automatically halts training at the optimal validation epoch (Epoch 47) to save best weights before memorization noise sets in.

---

## ⚡ Key Features

- **🏛️ High-Fidelity Banking UI:** Clean, modern commercial banking portal designed for credit officers without unnecessary clutter.
- **🆔 Seamless NID & CIB Verification:** Automatic National ID (NID) linkage with Credit Information Bureau (CIB) records with hover tooltips and live badge feedback.
- **⚡ 1-Click Fast Presets:** Instant loading of *Eligible Candidate* vs. *High-Risk Defaulter* profiles for lab demonstrations.
- **📈 Dynamic Financial Health Math:** Real-time calculation of **Household Income**, **Estimated EMI (৳)**, and **Debt-to-Income (DTI)** ratios.
- **🛡️ Explainable AI (XAI):** Neural predictions are translated into plain-text banking risk rationales for credit decision transparency.

---

## 📁 Repository Structure

```
credinet-bank/
├── data/
│   └── loan_data.csv                 # Raw benchmark loan dataset (614 records)
├── models/
│   ├── credinet_deep_mlp.pth         # PyTorch Deep MLP optimal weights
│   ├── credinet_shallow_mlp.pth      # Baseline Shallow MLP weights
│   ├── artifacts.pkl                 # StandardScaler & feature schema
│   └── metrics.json                  # Test set evaluation metrics
├── static/
│   ├── js/
│   │   └── tailwind.min.js           # Offline-bundled Tailwind CSS script
│   └── plots/
│       ├── loss_curve.png            # Convergence & Early Stopping curves
│       ├── confusion_matrix.png      # Deep MLP confusion matrix
│       └── roc_curve.png             # ROC curve comparison
├── templates/
│   └── index.html                    # Commercial banking frontend UI
├── app.py                            # Flask REST API & PyTorch inference engine
├── train_models.py                   # PyTorch training & evaluation pipeline
├── run_app.bat                       # 1-Click launcher script for Windows
├── Procfile                          # Cloud deployment configuration for Render/Heroku
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Mahfuz5634/credinet-bank.git
cd credinet-bank
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in any modern web browser.

> On Windows, you can simply double-click **`run_app.bat`** to automatically verify dependencies, train models if missing, and launch the portal in your default browser.

---

## 🌐 Deploy to Render.com (1-Click Free Cloud Hosting)

1. Fork or push this repository to your GitHub account (`https://github.com/Mahfuz5634/credinet-bank`).
2. Log in to [Render.com](https://dashboard.render.com/) and click **New +** &rarr; **Web Service**.
3. Select your repository: `Mahfuz5634/credinet-bank`.
4. Render will auto-detect settings from `Procfile`:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Plan:** Free
5. Click **Deploy Web Service** &mdash; your app will be live with a free global HTTPS URL!

---

## 👨‍💻 Author

* **Student:** Md. Mahfuz
* **Student ID:** `0432310005101057`
* **Course:** CSE 422: Neural Networks and Deep Learning
* **Repository:** [https://github.com/Mahfuz5634/credinet-bank](https://github.com/Mahfuz5634/credinet-bank)
