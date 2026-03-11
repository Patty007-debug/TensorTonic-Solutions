import numpy as np

def _sigmoid(z):
    """ Numerically stable sigmoid implementation. """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    X: (N, D) feature matrix
    y: (N,) labels (0 or 1)
    Returns: (w, b)
    """
    N, D = X.shape
    w = np.zeros(D)
    b = 0

    for step in range(steps):
        z = np.dot(X, w) + b
        p = _sigmoid(z)  # predicted probabilities

        # Compute loss (optional, for monitoring)
        loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))

        # Compute gradients
        dw = np.dot(X.T, (p - y)) / N
        db = np.sum(p - y) / N

        # Gradient descent update
        w -= lr * dw
        b -= lr * db

        if step % 100 == 0:
            print(f"Step {step}, loss={loss:.4f}")

    return w, b
    