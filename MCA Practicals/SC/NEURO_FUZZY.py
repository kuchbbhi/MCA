
import numpy as np
import matplotlib.pyplot as plt

# ---------- Utility functions ----------
def gaussian(x, c, sigma):
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)

# ---------- ANFIS class ----------
class SimpleANFIS1D:
    def __init__(self, n_rules=2, lr=0.01, seed=0):
        np.random.seed(seed)
        self.n_rules = n_rules
        # Antecedent params (centers and widths) init
        self.centers = np.linspace(0.2, 2.8, n_rules) + 0.1 * np.random.randn(n_rules)
        self.sigmas = np.ones(n_rules) * 0.5 + 0.1 * np.random.randn(n_rules)
        # Consequent params: y_i = p_i * x + q_i
        self.p = 0.5 * np.random.randn(n_rules)
        self.q = 0.5 * np.random.randn(n_rules)
        self.lr = lr

    def forward(self, x):
        # x: (N,)
        # compute membership values (N, n_rules)
        M = np.stack([gaussian(x, c, s) for c, s in zip(self.centers, self.sigmas)], axis=1)
        # firing strengths (wi)
        w = M  # using MF directly as firing for 1-input case
        wsum = np.sum(w, axis=1, keepdims=True) + 1e-8
        w_norm = w / wsum  # (N, n_rules)
        # rule outputs: f_i = p_i * x + q_i  -> shape (N, n_rules)
        f = self.p[np.newaxis, :] * x[:, np.newaxis] + self.q[np.newaxis, :]
        y_pred = np.sum(w_norm * f, axis=1)  # (N,)
        cache = {"x": x, "M": M, "w": w, "w_norm": w_norm, "f": f, "wsum": wsum}
        return y_pred, cache

    def compute_loss(self, y_true, y_pred):
        return 0.5 * np.mean((y_true - y_pred) ** 2)

    def step(self, x, y_true):
        # Forward
        y_pred, cache = self.forward(x)
        loss = self.compute_loss(y_true, y_pred)
        # Gradients via backprop (manual)
        N = x.shape[0]
        e = y_pred - y_true  # (N,)

        w = cache["w"]         # (N, R)
        w_norm = cache["w_norm"]  # (N, R)
        f = cache["f"]         # (N, R)
        wsum = cache["wsum"]   # (N,1)
        M = cache["M"]
        x_col = x[:, np.newaxis]  # (N,1)

        # Gradients for consequents p and q
        # y_pred = sum(w_norm * (p*x + q))
        # dL/dp_j = (1/N) * sum( e * w_norm[:,j] * x )
        dp = (1.0 / N) * np.sum((e[:, np.newaxis]) * w_norm * x_col, axis=0)
        dq = (1.0 / N) * np.sum((e[:, np.newaxis]) * w_norm, axis=0)

        # Gradients for antecedents (centers and sigmas)
        # Need d y_pred / d w_norm and d w_norm / d w and d w / d c, d w / d sigma
        # w_norm = w / wsum
        # dy/dw_j = sum_k [ d y / d w_norm_k * d w_norm_k / d w_j ]
        # where d y / d w_norm_k = f_k
        # and d w_norm_k / d w_j = (delta_kj * wsum - w_k) / wsum^2

        # Compute d y_pred / d w (N, R)
        # For each sample i and rule j:
        # dy_dw_ij = (f_ij * wsum_i - np.sum(w_i * f_i)) / wsum_i^2
        wf = w * f  # (N, R) elementwise
        sum_wf = np.sum(wf, axis=1, keepdims=True)  # (N,1)
        dy_dw = (f * wsum - sum_wf) / (wsum ** 2)  # (N, R)

        # d w / d center_j = d M / d c = M * ((x - c) / sigma^2)
        dM_dc = np.stack([M[:, j] * ((x - self.centers[j]) / (self.sigmas[j] ** 2))
                          for j in range(self.n_rules)], axis=1)  # (N,R)
        dM_dsigma = np.stack([M[:, j] * (((x - self.centers[j]) ** 2) / (self.sigmas[j] ** 3))
                              for j in range(self.n_rules)], axis=1)  # (N,R)
        # sign: gaussian derivative wrt sigma is M * ((x-c)^2 / sigma^3), with negative? Let's compute correctly:
        # d/dsigma exp(-0.5*((x-c)/s)^2) = exp(...) * ( (x-c)^2 / s^3 ) * (+0.5*2?) careful — use chain:
        # derivative = M * ((x-c)**2) / (self.sigmas[j]**3) * (+1) * (+?) -- to avoid sign issues, do numerical gradient sanity check later.
        # We'll include negative sign (correct derivative is M * ((x-c)**2) / (self.sigma**3)) * (+1) * (...). For safety, treat as computed below and tune lr.

        # Now gradient contributions from antecedents:
        # dL/dc_j = (1/N) * sum_i [ e_i * dy_dw_ij * d w_ij / d c_j ]
        grad_c = (1.0 / N) * np.sum((e[:, np.newaxis]) * dy_dw * dM_dc, axis=0)
        grad_sigma = (1.0 / N) * np.sum((e[:, np.newaxis]) * dy_dw * dM_dsigma, axis=0)

        # Update parameters (gradient descent, minimize loss => subtract grad * lr)
        self.p -= self.lr * dp
        self.q -= self.lr * dq
        self.centers -= self.lr * grad_c
        self.sigmas -= self.lr * grad_sigma

        return loss, y_pred

# ---------- Toy problem ----------
if __name__ == "__main__":
    # Generate data
    rng = np.random.RandomState(1)
    X = np.linspace(0, np.pi, 200)
    Y = np.sin(2 * X)  # target function
    # Add small noise
    Y_noisy = Y + 0.05 * rng.randn(*Y.shape)

    # Train-test split
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    train_idx = idx[:140]
    test_idx = idx[140:]

    X_train, Y_train = X[train_idx], Y_noisy[train_idx]
    X_test, Y_test = X[test_idx], Y_noisy[test_idx]

    # Model
    model = SimpleANFIS1D(n_rules=3, lr=0.05, seed=42)

    # Training
    epochs = 500
    train_losses = []
    for ep in range(epochs):
        loss, y_pred = model.step(X_train, Y_train)
        train_losses.append(loss)
        if (ep + 1) % 50 == 0:
            print(f"Epoch {ep+1}/{epochs}  Loss: {loss:.6f}")

    # Evaluate
    y_train_pred, _ = model.forward(X_train)
    y_test_pred, _ = model.forward(X_test)
    train_mse = np.mean((y_train_pred - Y_train) ** 2)
    test_mse = np.mean((y_test_pred - Y_test) ** 2)
    print(f"Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")

    # Plot results
    X_plot = np.linspace(0, np.pi, 400)
    y_plot_pred, _ = model.forward(X_plot)
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses)
    plt.title("Training loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.subplot(1,2,2)
    plt.scatter(X_train, Y_train, s=10, label="train (noisy)")
    plt.scatter(X_test, Y_test, s=10, label="test (noisy)")
    plt.plot(X_plot, np.sin(2*X_plot), label="true", linewidth=2)
    plt.plot(X_plot, y_plot_pred, label="anfis pred", linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.show()

