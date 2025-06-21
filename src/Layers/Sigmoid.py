import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):
    """
    Element-wise sigmoid activation.
    """
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.output = 1.0 / (1.0 + np.exp(-input_tensor))
        return self.output

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        return error_tensor * self.output * (1.0 - self.output)
