import numpy as np

class L1Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        return self.alpha * np.sum(np.abs(weights))


class L2Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * weights

    def norm(self, weights):
        return self.alpha * 0.5 * np.sum(weights ** 2)
