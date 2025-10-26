# model_numpy.py
import numpy as np

class NumpyMLP:
    """
    Простая MLP на NumPy с ReLU и выходом 1 (регрессия).
    В комплекте - NAdam оптимизатор (встроен как метод step_nadam).
    """
    def __init__(self, layer_sizes, seed=42):
        self.sizes = layer_sizes
        rng = np.random.RandomState(seed)
        self.params = {}
        for i in range(len(layer_sizes)-1):
            self.params[f"W{i}"] = rng.normal(0, 0.1, (layer_sizes[i], layer_sizes[i+1])).astype(np.float32)
            self.params[f"b{i}"] = np.zeros((1, layer_sizes[i+1]), dtype=np.float32)
        self.opt_m = {k: np.zeros_like(v) for k,v in self.params.items()}
        self.opt_v = {k: np.zeros_like(v) for k,v in self.params.items()}
        self.t = 0

    def forward(self, X):
        A = [X]
        Z = []
        L = len(self.sizes)-1
        a = X
        for i in range(L-1):
            z = a.dot(self.params[f"W{i}"]) + self.params[f"b{i}"]
            Z.append(z)
            a = np.maximum(0, z)
            A.append(a)
        z = a.dot(self.params[f"W{L-1}"]) + self.params[f"b{L-1}"]
        Z.append(z)
        A.append(z)
        return z, A, Z

    def compute_loss_and_grads(self, X, y):
        m = X.shape[0]
        z, A, Z = self.forward(X)
        loss = np.mean((z - y)**2)
        grads = {}
        L = len(self.sizes)-1
        dz = 2*(z - y)/m
        a_prev = A[-2]
        grads[f"W{L-1}"] = a_prev.T.dot(dz)
        grads[f"b{L-1}"] = np.sum(dz, axis=0, keepdims=True)
        da_prev = dz.dot(self.params[f"W{L-1}"].T)
        for i in range(L-2, -1, -1):
            z_i = Z[i]
            dz_i = da_prev * (z_i > 0)
            a_prev = A[i]
            grads[f"W{i}"] = a_prev.T.dot(dz_i)
            grads[f"b{i}"] = np.sum(dz_i, axis=0, keepdims=True)
            da_prev = dz_i.dot(self.params[f"W{i}"].T)
        return loss, grads

    def step_nadam(self, grads, lr=0.003, beta1=0.9, beta2=0.999, eps=1e-8, schedule_decay=0.004):
        # NAdam per Dozat (approx.)
        self.t += 1
        mu_t = beta1 * (1 - 0.5 * (0.96 ** (self.t * schedule_decay)))
        mu_t1 = beta1 * (1 - 0.5 * (0.96 ** ((self.t+1) * schedule_decay)))
        for k in self.params.keys():
            g = grads[k]
            m = self.opt_m[k]; v = self.opt_v[k]
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * (g * g)
            m_hat = m / (1 - beta1**self.t)
            v_hat = v / (1 - beta2**self.t)
            m_bar = (1 - mu_t) * g / (1 - beta1**self.t) + mu_t1 * m_hat
            update = m_bar / (np.sqrt(v_hat) + eps)
            self.params[k] -= lr * update

    def predict(self, X):
        z, _, _ = self.forward(X)
        return z
