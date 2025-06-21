import numpy as np

class Sgd:
    """
    Vanilla Stochastic Gradient Descent optimizer with proximal regularization.
    """
    def __init__(self, learning_rate):
        self.lr = learning_rate
        self.regularizer = None

    def add_regularizer(self, regularizer):
        """Attach a regularizer to this optimizer."""
        self.regularizer = regularizer

    def calculate_update(self, weights, gradient):
        # proximal step for regularizer
        if self.regularizer is not None:
            weights = weights - self.lr * self.regularizer.calculate_gradient(weights)
        # standard SGD step
        return weights - self.lr * gradient


class SgdWithMomentum:
    """
    SGD with momentum and proximal regularization.
    """
    def __init__(self, learning_rate, momentum):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

    def calculate_update(self, weights, gradient):
        # proximal step for regularizer
        if self.regularizer is not None:
            weights = weights - self.lr * self.regularizer.calculate_gradient(weights)
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)
        # momentum update
        self.velocity = self.momentum * self.velocity - self.lr * gradient
        return weights + self.velocity


class Adam:
    """
    Adam optimizer with proximal regularization.
    """
    def __init__(self, learning_rate, mu, rho, eps=1e-8):
        self.lr = learning_rate
        self.mu = mu
        self.rho = rho
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

    def calculate_update(self, weights, gradient):
        # proximal step for regularizer
        if self.regularizer is not None:
            weights = weights - self.lr * self.regularizer.calculate_gradient(weights)
        # initialize moments
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)
        self.t += 1
        # biased moments
        self.m = self.mu * self.m + (1 - self.mu) * gradient
        self.v = self.rho * self.v + (1 - self.rho) * (gradient ** 2)
        # bias correction
        m_hat = self.m / (1 - self.mu ** self.t)
        v_hat = self.v / (1 - self.rho ** self.t)
        # parameter update
        update = - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return weights + update
