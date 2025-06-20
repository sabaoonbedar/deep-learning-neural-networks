import numpy as np

class Sgd:
    """
    Vanilla Stochastic Gradient Descent optimizer.
    On calculate_update, returns new weights: w_new = w - lr * gradient
    """
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def calculate_update(self, weights, gradient):
        return weights - self.lr * gradient

class SgdWithMomentum:
    """
    SGD with classical momentum.
    Velocity initialized to zero. Returns new weights: w_new = w + v,
    where v = momentum * v - lr * gradient.
    """
    def __init__(self, learning_rate, momentum):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def calculate_update(self, weights, gradient):
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)
        self.velocity = self.momentum * self.velocity - self.lr * gradient
        return weights + self.velocity

class Adam:
    """
    Adam optimizer.
    m and v initialized to zero arrays.
    On calculate_update, updates m, v, t;
    computes bias-corrected m_hat, v_hat;
    update = - lr * m_hat / (sqrt(v_hat) + eps);
    returns new weights: w_new = w + update.
    """
    def __init__(self, learning_rate, mu, rho, eps=1e-8):
        self.lr = learning_rate
        self.mu = mu      # beta1
        self.rho = rho    # beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def calculate_update(self, weights, gradient):
        # initialize moment vectors
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)
        # increment time step
        self.t += 1
        # update biased first and second moments
        self.m = self.mu * self.m + (1 - self.mu) * gradient
        self.v = self.rho * self.v + (1 - self.rho) * (gradient ** 2)
        # compute bias-corrected moments
        m_hat = self.m / (1 - self.mu ** self.t)
        v_hat = self.v / (1 - self.rho ** self.t)
        # compute parameter update
        update = - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        # always apply update
        return weights + update
