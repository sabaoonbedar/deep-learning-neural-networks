
import numpy as np

class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        # element‐wise sign(w) * α
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        # α * sum(|w|)
        return self.alpha * np.sum(np.abs(weights))


class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        # w * α
        return self.alpha * weights

    def norm(self, weights):
        # α * sum(w²)
        return self.alpha * np.sum(np.square(weights))
