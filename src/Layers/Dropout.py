import numpy as np
from .Base import BaseLayer

class Dropout(BaseLayer):
    """
    Implements inverted dropout:
    - During training: randomly drops activations and scales others by 1/(1-p)
    - During testing: pass-through (no dropout)
    """
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.trainable = False
        self._mask = None
        self.testing_phase = False

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        else:
            self._mask = (np.random.rand(*input_tensor.shape) > self.probability).astype(input_tensor.dtype)
            return self._mask * input_tensor / (1.0 - self.probability)

    def backward(self, error_tensor):
        if self.testing_phase:
            return error_tensor
        else:
            return self._mask * error_tensor / (1.0 - self.probability)
