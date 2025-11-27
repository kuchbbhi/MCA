import numpy as np
import matplotlib.pyplot as plt


# ---------------- Gaussian MF ----------------
def gaussian(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)


# ---------------- Simple ANFIS ----------------
class SimpleANFIS1D:
    def __init__(self, n_rules=2, lr=0.01, seed=0):
        np.random.seed(seed)
        self.R = n_rules
        self.lr = lr

        self.centers = np.linspace(0.2, 2.8, self.R) + 0.1 * np.random.randn(self.R)
        self.sigmas = 0.5 + 0.1 * np.random.randn(self.R)
        self.p = 0.5 * np.random.randn(self.R)
        self.q = 0.5 * np.random.randn(self.R)

    def forward(self, x):
        M = np.stack(
            [gaussian(x, c, s) for c, s in zip(self.centers, self.sigmas)], axis=1
        )
        w = M
        w_norm = w / (w.sum(axis=1, keepdims=True) + 1e-8)
        f = self.p * x[:, None] + self.q
        return (w_norm * f).sum(axis=1), (x, M, w_norm, f)

    def step(self, x, y):
        y_pred, (x, M, w_norm, f) = self.forward(x)
        e = y_pred - y
        N = len(x)

        # ----- Consequent gradients -----
        dp = (e[:, None] * w_norm * x[:, None]).mean(axis=0)
        dq = (e[:, None] * w_norm).mean(axis=0)

        # ----- Antecedent gradients -----
        w = M
        wsum = w.sum(axis=1, keepdims=True)
        wf_sum = (w * f).sum(axis=1, keepdims=True)

        dy_dw = (f * wsum - wf_sum) / (wsum**2)  # shape (N,R)

        dM_dc = M * (x[:, None] - self.centers) / (self.sigmas**2)
        dM_ds = M * ((x[:, None] - self.centers) ** 2) / (self.sigmas**3)

        grad_c = (e[:, None] * dy_dw * dM_dc).mean(axis=0)
        grad_s = (e[:, None] * dy_dw * dM_ds).mean(axis=0)

        # ----- Update -----
        self.p -= self.lr * dp
        self.q -= self.lr * dq
        self.centers -= self.lr * grad_c
        self.sigmas -= self.lr * grad_s

        return 0.5 * np.mean(e**2)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    rng = np.random.RandomState(1)
    X = np.linspace(0, np.pi, 200)
    Y = np.sin(2 * X)
    Y_noisy = Y + 0.05 * rng.randn(len(X))

    idx = rng.permutation(len(X))
    X_train, Y_train = X[idx[:140]], Y_noisy[idx[:140]]
    X_test, Y_test = X[idx[140:]], Y_noisy[idx[140:]]

    model = SimpleANFIS1D(n_rules=3, lr=0.05, seed=42)

    # ----- Training -----
    losses = []
    for ep in range(500):
        loss = model.step(X_train, Y_train)
        losses.append(loss)
        if (ep + 1) % 50 == 0:
            print(f"Epoch {ep+1}/500 — Loss: {loss:.6f}")

    # ----- Evaluation -----
    y_train, _ = model.forward(X_train)
    y_test, _ = model.forward(X_test)
    print("Train MSE:", ((y_train - Y_train) ** 2).mean())
    print("Test  MSE:", ((y_test - Y_test) ** 2).mean())

    # ----- Plot -----
    X_plot = np.linspace(0, np.pi, 400)
    y_plot, _ = model.forward(X_plot)

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
    plt.scatter(X_train, Y_train, s=12, label="train")
    plt.scatter(X_test, Y_test, s=12, label="test")
    plt.plot(X_plot, np.sin(2 * X_plot), label="true")
    plt.plot(X_plot, y_plot, label="anfis")
    plt.legend()
    plt.tight_layout()
    plt.show()
