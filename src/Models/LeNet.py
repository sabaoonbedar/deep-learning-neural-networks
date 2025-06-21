import numpy as np
from ..Layers import FullyConnected, Flatten, ReLU, SoftMax, Conv, Pooling, He, Constant
from ..Optimization.Optimizers import Sgd


class LeNet:
    """
    Classic LeNet-5 CNN architecture for digit classification.
    Assumes input shape (batch_size, 1, 28, 28) for grayscale images.
    """
    def __init__(self, optimizer, weight_initializer, bias_initializer):
        self.layers = []

        # First convolution block
        self.layers.append(Conv(stride_shape=(1, 1), conv_shape=(1, 5, 5), num_kernels=6))
        self.layers.append(ReLU())
        self.layers.append(Pooling(stride_shape=2, pool_shape=2))

        # Second convolution block
        self.layers.append(Conv(stride_shape=(1, 1), conv_shape=(6, 5, 5), num_kernels=16))
        self.layers.append(ReLU())
        self.layers.append(Pooling(stride_shape=2, pool_shape=2))

        # Fully connected layers
        self.layers.append(Flatten())
        self.layers.append(FullyConnected(16 * 4 * 4, 120))
        self.layers.append(ReLU())
        self.layers.append(FullyConnected(120, 84))
        self.layers.append(ReLU())
        self.layers.append(FullyConnected(84, 10))
        self.layers.append(SoftMax())

        # Initialize trainable layers
        for layer in self.layers:
            if hasattr(layer, 'trainable') and layer.trainable:
                layer.optimizer = optimizer
                if hasattr(layer, 'initialize'):
                    layer.initialize(weight_initializer, bias_initializer)

    def forward(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    def backward(self, error_tensor):
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
        return error_tensor

    @staticmethod
    def build():
        optimizer = Sgd(learning_rate=0.01)
        weight_init = He()
        bias_init = Constant(0.1)
        return LeNet(optimizer, weight_init, bias_init)
