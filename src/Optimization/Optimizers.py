import numpy as np

class Sgd:
    """
    Vanilla Stochastic Gradient Descent optimizer.
    Supports optional regularization.
    """
    def __init__(self, learning_rate):
        self.lr = learning_rate
        self.regularizer = None

    def calculate_update(self, weights, gradient):
        if self.regularizer is not None:
            gradient += self.regularizer.calculate_gradient(weights)
        return weights - self.lr * gradient


class SgdWithMomentum:
    """
    SGD with classical momentum and optional regularization.
    """
    def __init__(self, learning_rate, momentum):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.regularizer = None

    def calculate_update(self, weights, gradient):
        if self.regularizer is not None:
            gradient += self.regularizer.calculate_gradient(weights)
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)
        self.velocity = self.momentum * self.velocity - self.lr * gradient
        return weights + self.velocity


class Adam:
    """
    Adam optimizer with optional regularization.
    """
    def __init__(self, learning_rate, mu, rho, eps=1e-8):
        self.lr = learning_rate
        self.mu = mu      # beta1
        self.rho = rho    # beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
        self.regularizer = None

    def calculate_update(self, weights, gradient):
        if self.regularizer is not None:
            gradient += self.regularizer.calculate_gradient(weights)
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)
        self.t += 1
        self.m = self.mu * self.m + (1 - self.mu) * gradient
        self.v = self.rho * self.v + (1 - self.rho) * (gradient ** 2)
        m_hat = self.m / (1 - self.mu ** self.t)
        v_hat = self.v / (1 - self.rho ** self.t)
        update = - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return weights + update
