import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    """
    SoftMax activation layer.
    Converts logits to probabilities for classification tasks.
    """
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.output_tensor = None

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        input_stable = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_input = np.exp(input_stable)
        self.output_tensor = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        return self.output_tensor

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        # Assumes cross-entropy is used, so this is just (prediction - target)
        return error_tensor
