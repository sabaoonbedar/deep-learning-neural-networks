import numpy as np
from .Base import BaseLayer

class TanH(BaseLayer):
    """
    Element-wise hyperbolic tangent activation.
    """
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.output = np.tanh(input_tensor)
        return self.output

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        return error_tensor * (1 - self.output ** 2)
