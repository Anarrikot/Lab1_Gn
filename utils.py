# utils.py
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os, time

def load_and_preprocess(path="dataset/data.csv", target_col="MSRP", input_features=None, test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataset")
    df = df.dropna(subset=[target_col]).copy()
    # If input_features provided, use them; else take all numeric except target
    if input_features is None:
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        input_features = [c for c in num if c != target_col]
    X = df[input_features].copy()
    # Fill numeric NaNs
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
        X[c] = X[c].fillna(X[c].median())
    y = pd.to_numeric(df[target_col], errors='coerce').fillna(0).values.reshape(-1,1)
    # log-transform target to stabilize
    y_log = np.log1p(y)
    X = X.values.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train.astype(np.float32), y_test.astype(np.float32), scaler, input_features

def compute_metrics(y_true_log, y_pred_log):
    # y_true_log, y_pred_log are log1p
    mse_log = mean_squared_error(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mse_log)
    # back to original
    y_true = np.expm1(y_true_log).ravel()
    y_pred = np.expm1(y_pred_log).ravel()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"mse_log": float(mse_log), "rmse_log": float(rmse_log), "mse": float(mse), "rmse": float(rmse), "r2": float(r2)}

def plot_losses(histories, out="losses.png"):
    """
    histories: dict name -> list of loss values per epoch
    """
    plt.figure(figsize=(8,5))
    for name, lst in histories.items():
        plt.plot(lst, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log-target)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
