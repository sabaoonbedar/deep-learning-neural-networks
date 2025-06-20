import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):
    """
    Element-wise ReLU activation: max(0, x)
    """
    def __init__(self):
        super().__init__()
        # Non-trainable activation

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.input = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        # Gradient is 1 for input>0, else 0
        grad = error_tensor * (self.input > 0)
        return grad