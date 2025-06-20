import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    """
    SoftMax activation to produce class probabilities.
    """
    def __init__(self):
        super().__init__()
        # Non-trainable activation

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        # Shift for numerical stability
        shifted = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exps = np.exp(shifted)
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        # Vectorized Jacobian-product: dL/dz = y * (error - sum(error*y))
        y = self.output
        dot = np.sum(error_tensor * y, axis=1, keepdims=True)
        return y * (error_tensor - dot)