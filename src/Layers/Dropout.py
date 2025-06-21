# Layers/Dropout.py

import numpy as np
from .Base import BaseLayer

class Dropout(BaseLayer):
    """
    Implements inverted dropout:
    - During training: randomly zeroes activations with probability (1-p),
      and scales the rest by 1/p.
    - During testing: pass-through (no dropout).
    """
    def __init__(self, probability):
        super().__init__()
        # here `probability` is the keep probability p
        self.probability = probability
        self.trainable = False
        self._mask = None
        self.testing_phase = False

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        # keep each unit with probability p
        self._mask = (np.random.rand(*input_tensor.shape) < self.probability).astype(input_tensor.dtype)
        # scale up remaining units by 1/p
        return input_tensor * self._mask / self.probability

    def backward(self, error_tensor):
        if self.testing_phase:
            return error_tensor
        # propagate gradient only where mask==1, scaled the same way
        return error_tensor * self._mask / self.probability
