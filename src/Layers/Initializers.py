import numpy as np

class Constant:
    """
    Initialize all weights to a fixed constant.
    """
    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, shape, fan_in, fan_out):
        return np.full(shape, self.value)

class UniformRandom:
    """
    Uniform [0,1) initialization.
    """
    def initialize(self, shape, fan_in, fan_out):
        return np.random.rand(*shape)

class Xavier:
    """
    Glorot/Xavier normal initialization:
      values ~ N(0, scale^2), scale = sqrt(2 / (fan_in + fan_out)).
    """
    def initialize(self, shape, fan_in, fan_out):
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(*shape) * scale

class He:
    """
    Kaiming/He normal initialization for ReLU:
      values ~ N(0, scale^2), scale = sqrt(2 / fan_in).
    """
    def initialize(self, shape, fan_in, fan_out):
        scale = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape) * scale