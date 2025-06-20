import numpy as np
from .Base import BaseLayer

class Flatten(BaseLayer):
    """
    Flattens input tensor to shape (batch_size, -1) on forward,
    and restores original shape on backward.
    """
    def __init__(self):
        super().__init__()
        # Non-trainable layer
        self.trainable = False
        self.input_shape = None

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        # Save original shape (batch_size, ...)
        self.input_shape = input_tensor.shape
        batch_size = input_tensor.shape[0]
        # Collapse all but batch dimension
        return input_tensor.reshape(batch_size, -1)

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        # Reshape error back to original input shape
        return error_tensor.reshape(self.input_shape)
