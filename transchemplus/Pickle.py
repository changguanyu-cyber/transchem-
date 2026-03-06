import pickle
import pandas as pd
import torch
import pickle
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch.nn.functional as F
from xgboost import XGBRegressor

def train_gpr(xtrain, ytrain, kernel_variance, kernel_lengthscale, white_kernel_variance, max_iterations):
    kernel = GPy.kern.Matern32(input_dim=xtrain.shape[1], variance=kernel_variance, lengthscale=kernel_lengthscale)
    kernel += GPy.kern.White(xtrain.shape[1], variance=white_kernel_variance)
    model = GPy.models.GPRegression(xtrain, ytrain, kernel)
    model.optimize(max_iters=max_iterations)
    return model

def predict_gpr(model, xtest):
    mean, variance = model.predict(xtest)
    std = np.sqrt(variance)
    return mean, std

# ---- Load PKL ----
with open("/root/autodl-tmp/GCN/gnn_sequences.pkl", "rb") as f:
    data = pickle.load(f)

ids = data["ids"]
print(ids[10])
if isinstance(ids, torch.Tensor):
    ids = ids.cpu().tolist()

sequences = data["sequences"]  # list of vectors
print(sequences[10])
print(len(ids), len(sequences))   # must equal

# ---- Load CSV ----
csv_path = "/root/autodl-tmp/Uncertianty_quantification_Polymer_informatics-main/data/experiment_polymer_database_2025-11-18.csv"
target_column = "Atomization_Energy_eV"
df = pd.read_csv(csv_path)

# ---- Match vector sequences with labels ----
ids = [int(i) if torch.is_tensor(i) else i for i in ids]
targets = df[target_column].iloc[ids].values

# ---- Prepare numpy ----
X = np.array(sequences, dtype=np.float32)   # shape [N, 128]
y = targets.astype(np.float32).reshape(-1, 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ---- Split train/test ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# =========================
# 🔥 XGBoost training
# =========================
xgb_model = XGBRegressor(
    n_estimators=300,       # number of trees (tunable)
    max_depth=6,            # tree depth (tunable)
    learning_rate=0.05,     # learning rate
    subsample=0.7,          # row sampling to prevent overfitting
    colsample_bytree=0.7,   # column sampling
    reg_lambda=1.0,         # L2 regularization
    tree_method="hist"
)

xgb_model.fit(X_train, y_train)

# =========================
# Inference and evaluation
# =========================
y_pred_test = xgb_model.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)
print(f"XGBoost R² on test set: {r2_test:.4f}")

# Test on the full dataset (optional)
y_pred_all = xgb_model.predict(X)
r2_all = r2_score(y, y_pred_all)
print(f"XGBoost R² on full dataset: {r2_all:.4f}")