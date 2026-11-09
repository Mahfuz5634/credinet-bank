import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)

# -------------------------------------------------------------
# 1. Load and Preprocess Data
# -------------------------------------------------------------
DATA_PATH = os.path.join("data", "loan_data.csv")
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop Loan_ID as it's an identifier
if "Loan_ID" in df.columns:
    df = df.drop(columns=["Loan_ID"])

# Target variable: 'Y' -> 1, 'N' -> 0
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
df = df.dropna(subset=["Loan_Status"])
df["Loan_Status"] = df["Loan_Status"].astype(int)

# Handle Missing Values
# Categorical columns: fill with mode
cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Credit_History", "Property_Area"]
impute_values = {}

for col in cat_cols:
    mode_val = df[col].mode()[0]
    impute_values[col] = mode_val
    df[col] = df[col].fillna(mode_val)

# Numerical columns: fill with median
num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
for col in num_cols:
    median_val = df[col].median()
    impute_values[col] = float(median_val)
    df[col] = df[col].fillna(median_val)

# Feature Engineering
df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["EMI"] = (df["LoanAmount"] * 1000) / df["Loan_Amount_Term"].replace(0, 360)
df["Balance_Income"] = df["Total_Income"] - (df["EMI"] * 1.5)

# Convert Dependents '3+' to 3
df["Dependents"] = df["Dependents"].replace({"3+": 3}).astype(int)

# Binary Encoding
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(1).astype(int)
df["Married"] = df["Married"].map({"Yes": 1, "No": 0}).fillna(1).astype(int)
df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0}).fillna(1).astype(int)
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
df["Credit_History"] = df["Credit_History"].astype(float)

# One-Hot Encoding for Property_Area
df = pd.get_dummies(df, columns=["Property_Area"], drop_first=False)

# Make sure boolean columns from get_dummies are int
for col in ["Property_Area_Rural", "Property_Area_Semiurban", "Property_Area_Urban"]:
    if col in df.columns:
        df[col] = df[col].astype(int)
    else:
        df[col] = 0

# Separate Features (X) and Target (y)
feature_cols = [c for c in df.columns if c != "Loan_Status"]
X = df[feature_cols].copy()
y = df["Loan_Status"].values

print("Final Feature Columns:", feature_cols)
print(f"Total features: {len(feature_cols)}")

# Train / Test Split (80% train, 20% test, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

# Standard Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# PyTorch DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_dim = X_train_scaled.shape[1]

# -------------------------------------------------------------
# 2. PyTorch Model Architectures
# -------------------------------------------------------------
class ShallowMLP(nn.Module):
    """Basic Shallow Neural Network for baseline comparison."""
    def __init__(self, input_dim):
        super(ShallowMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class DeepMLP(nn.Module):
    """
    Deep Multi-Layer Perceptron (CrediNet Architecture)
    Includes:
      - Batch Normalization (reduces internal covariate shift)
      - Dropout (prevents overfitting)
      - Hierarchical representation layers (64 -> 32 -> 16 -> 1)
    """
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

# -------------------------------------------------------------
# 3. Model Training Function with Early Stopping
# -------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=150, lr=0.001, patience=20):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * X_b.size(0)
            
        train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                running_val_loss += loss.item() * X_b.size(0)
        
        val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return train_losses, val_losses

# -------------------------------------------------------------
# 4. Train Both Models
# -------------------------------------------------------------
print("\n--- Training Shallow MLP (Baseline) ---")
shallow_model = ShallowMLP(input_dim)
shallow_train_losses, shallow_val_losses = train_model(
    shallow_model, train_loader, test_loader, epochs=100, lr=0.002, patience=25
)

print("\n--- Training Deep MLP (CrediNet with BatchNorm & Dropout) ---")
deep_model = DeepMLP(input_dim, dropout_rate=0.20)
deep_train_losses, deep_val_losses = train_model(
    deep_model, train_loader, test_loader, epochs=150, lr=0.001, patience=30
)

# -------------------------------------------------------------
# 5. Evaluate and Compare
# -------------------------------------------------------------
def evaluate_model(model, X_test_tensor, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probabilities = torch.sigmoid(logits).numpy().ravel()
        predictions = (probabilities >= 0.5).astype(int)
        
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)
    cm = confusion_matrix(y_test, predictions)
    
    return {
        "accuracy": round(float(acc) * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "recall": round(float(rec) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "roc_auc": round(float(auc) * 100, 2),
        "confusion_matrix": cm.tolist(),
        "probabilities": probabilities,
        "predictions": predictions
    }

shallow_metrics = evaluate_model(shallow_model, X_test_tensor, y_test)
deep_metrics = evaluate_model(deep_model, X_test_tensor, y_test)

print("\n=== Model Comparison Results ===")
print(f"Shallow MLP: Accuracy = {shallow_metrics['accuracy']}%, F1 = {shallow_metrics['f1_score']}%, ROC-AUC = {shallow_metrics['roc_auc']}%")
print(f"Deep MLP:    Accuracy = {deep_metrics['accuracy']}%, F1 = {deep_metrics['f1_score']}%, ROC-AUC = {deep_metrics['roc_auc']}%")

# Save Models
torch.save(shallow_model.state_dict(), "models/credinet_shallow_mlp.pth")
torch.save(deep_model.state_dict(), "models/credinet_deep_mlp.pth")

# Save Metadata and Scaler
artifacts = {
    "feature_cols": feature_cols,
    "input_dim": input_dim,
    "impute_values": impute_values,
    "num_cols": num_cols,
    "scaler": scaler
}
joblib.dump(artifacts, "models/artifacts.pkl")

# Save Metrics JSON (without numpy arrays)
metrics_to_save = {
    "shallow": {
        "accuracy": shallow_metrics["accuracy"],
        "precision": shallow_metrics["precision"],
        "recall": shallow_metrics["recall"],
        "f1_score": shallow_metrics["f1_score"],
        "roc_auc": shallow_metrics["roc_auc"],
        "confusion_matrix": shallow_metrics["confusion_matrix"]
    },
    "deep": {
        "accuracy": deep_metrics["accuracy"],
        "precision": deep_metrics["precision"],
        "recall": deep_metrics["recall"],
        "f1_score": deep_metrics["f1_score"],
        "roc_auc": deep_metrics["roc_auc"],
        "confusion_matrix": deep_metrics["confusion_matrix"]
    },
    "test_samples": len(y_test),
    "total_samples": len(df)
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics_to_save, f, indent=4)

# -------------------------------------------------------------
# 6. Generate Publication-Quality Plots for Lab Presentation
# -------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Plot 1: Loss Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(shallow_train_losses, label="Train Loss", color="#3b82f6", lw=2)
ax1.plot(shallow_val_losses, label="Val Loss", color="#ef4444", lw=2, linestyle="--")
ax1.set_title("Shallow MLP: Loss Curves", fontsize=13, fontweight='bold')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Binary Cross Entropy Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(deep_train_losses, label="Train Loss (with BatchNorm/Dropout)", color="#10b981", lw=2)
ax2.plot(deep_val_losses, label="Val Loss (with Early Stopping)", color="#6366f1", lw=2, linestyle="--")
ax2.set_title("Deep MLP (CrediNet): Loss Curves", fontsize=13, fontweight='bold')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Binary Cross Entropy Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("static/plots/loss_curve.png", dpi=300)
plt.close()

# Plot 2: Confusion Matrix (Deep MLP)
plt.figure(figsize=(6, 5))
cm = np.array(deep_metrics["confusion_matrix"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Rejected (0)", "Approved (1)"],
            yticklabels=["Rejected (0)", "Approved (1)"])
plt.title("Deep MLP Confusion Matrix", fontsize=13, fontweight='bold')
plt.xlabel("Predicted Label", fontweight='bold')
plt.ylabel("Actual True Label", fontweight='bold')
plt.tight_layout()
plt.savefig("static/plots/confusion_matrix.png", dpi=300)
plt.close()

# Plot 3: ROC Curve Comparison
fpr_s, tpr_s, _ = roc_curve(y_test, shallow_metrics["probabilities"])
fpr_d, tpr_d, _ = roc_curve(y_test, deep_metrics["probabilities"])

plt.figure(figsize=(7, 5))
plt.plot(fpr_s, tpr_s, label=f"Shallow MLP (AUC = {shallow_metrics['roc_auc']}%)", color="#f59e0b", lw=2)
plt.plot(fpr_d, tpr_d, label=f"Deep MLP (AUC = {deep_metrics['roc_auc']}%)", color="#10b981", lw=2.5)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label="Random Guess")
plt.title("ROC Curves: Shallow vs Deep MLP", fontsize=13, fontweight='bold')
plt.xlabel("False Positive Rate", fontweight='bold')
plt.ylabel("True Positive Rate", fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("static/plots/roc_curve.png", dpi=300)
plt.close()

print("\nModel training and artifact generation completed successfully!")
