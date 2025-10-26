# main.py
import time, json
import numpy as np
from utils import load_and_preprocess, compute_metrics, plot_losses, ensure_dir
from model_numpy import NumpyMLP
from model_torch import TorchRegressor
from model_tf import build_tf_regressor

# Try imports lazily
try:
    import torch, torch.nn as nn, torch.optim as optim

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

OUT_DIR = "results"
ensure_dir(OUT_DIR)


def train_numpy(X_train, y_train, X_test, y_test, epochs=30, batch_size=64):
    input_dim = X_train.shape[1]
    model = NumpyMLP([input_dim, 128, 64, 1])
    history = []
    for ep in range(epochs):
        perm = np.random.permutation(X_train.shape[0])
        Xs = X_train[perm];
        ys = y_train[perm]
        epoch_loss = 0.0
        for i in range(0, Xs.shape[0], batch_size):
            xb = Xs[i:i + batch_size];
            yb = ys[i:i + batch_size]
            loss, grads = model.compute_loss_and_grads(xb, yb)
            model.step_nadam(grads, lr=0.003)
            epoch_loss += loss * xb.shape[0]
        epoch_loss /= Xs.shape[0]
        history.append(epoch_loss)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[NumPy] ep {ep + 1} loss {epoch_loss:.6f}")
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    return {"history": history, "metrics": metrics}


def train_torch(X_train, y_train, X_test, y_test, epochs=50, batch_size=128, lr=1e-3):
    if not TORCH_AVAILABLE:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=device)
    model = TorchRegressor(X_train.shape[1]).to(device)
    results = {}
    for opt_name in ["NAdam", "Adam"]:
        if opt_name == "NAdam":
            optimizer = optim.NAdam(model.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history = []
        for ep in range(epochs):
            perm = torch.randperm(X_tr.size(0))
            epoch_loss = 0.0
            for i in range(0, X_tr.size(0), batch_size):
                idx = perm[i:i + batch_size]
                xb = X_tr[idx];
                yb = y_tr[idx]
                optimizer.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= X_tr.size(0)
            history.append(epoch_loss)
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"[{opt_name}] ep {ep + 1} loss {epoch_loss:.6f}")
        with torch.no_grad():
            pred = model(X_te).cpu().numpy()
        metrics = compute_metrics(y_test, pred)
        results[opt_name] = {"history": history, "metrics": metrics}
    return results


def train_tf(X_train, y_train, X_test, y_test, epochs=50, batch_size=128, lr=1e-3):
    if not TF_AVAILABLE:
        return None
    model = build_tf_regressor(X_train.shape[1], hidden=[128, 64], dropout=0.1)
    results = {}
    for opt_name in ["Nadam", "Adam"]:
        if opt_name == "Nadam":
            opt = tf.keras.optimizers.Nadam(learning_rate=lr)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss="mse")
        history = model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=0)
        pred = model.predict(X_test)
        metrics = compute_metrics(y_test, pred)
        results[opt_name] = {"history": history.history['loss'], "metrics": metrics}
    return results


def main():
    print("Loading and preprocessing...")
    X_train, X_test, y_train, y_test, scaler, feature_list = load_and_preprocess(path="dataset/data.csv",
                                                                                 target_col="MSRP")
    print("Shapes:", X_train.shape, y_train.shape)
    report = {}
    # NumPy
    print("Training NumPy model...")
    t0 = time.time()
    res_np = train_numpy(X_train, y_train, X_test, y_test, epochs=30)
    report['numpy'] = {"time": time.time() - t0, **res_np}
    # PyTorch
    if TORCH_AVAILABLE:
        print("Training PyTorch models...")
        t0 = time.time()
        res_torch = train_torch(X_train, y_train, X_test, y_test, epochs=50)
        report['pytorch'] = {"time": time.time() - t0, **(res_torch or {})}
    else:
        print("PyTorch not available.")
    # TensorFlow
    if TF_AVAILABLE:
        print("Training TensorFlow models...")
        t0 = time.time()
        res_tf = train_tf(X_train, y_train, X_test, y_test, epochs=50)
        report['tensorflow'] = {"time": time.time() - t0, **(res_tf or {})}
    else:
        print("TensorFlow not available.")

    # Plot losses
    histories = {}
    histories['NumPy NAdam'] = report['numpy']['history']
    if 'pytorch' in report:
        if 'NAdam' in report['pytorch']:
            histories['PyTorch NAdam'] = report['pytorch']['NAdam']['history']
        if 'Adam' in report['pytorch']:
            histories['PyTorch Adam'] = report['pytorch']['Adam']['history']
    if 'tensorflow' in report:
        if 'Nadam' in report['tensorflow']:
            histories['TF Nadam'] = report['tensorflow']['Nadam']['history']
        if 'Adam' in report['tensorflow']:
            histories['TF Adam'] = report['tensorflow']['Adam']['history']

    def convert_floats(obj):
        if isinstance(obj, dict):
            return {k: convert_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_floats(i) for i in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

    report = convert_floats(report)

    plot_losses(histories, out=f"{OUT_DIR}/losses.png")
    with open(f"{OUT_DIR}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {OUT_DIR}/report.json and loss plot to {OUT_DIR}/losses.png")
    print("Experiment finished. Summary metrics:")
    for k, v in report.items():
        print(k, "=>", v.get('metrics', v))


if __name__ == "__main__":
    main()
