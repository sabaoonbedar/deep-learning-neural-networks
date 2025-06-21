import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    """
    A trainable linear layer performing: output = X_ext @ W,
    where X_ext = [X | 1] incorporates bias into weights.
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        # This layer is trainable
        self.trainable = True

        # Save dimensions
        self.input_size = input_size
        self.output_size = output_size

        # Initialize weight matrix with an extra row for bias
        # Shape: (input_size + 1, output_size)
        self.weights = np.random.rand(input_size + 1, output_size)

        # Placeholder for weight gradients
        self.gradient_weights = None

        # Optimizer placeholder
        self._optimizer = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        # Assign optimizer (deepcopied in NeuralNetwork.append_layer)
        self._optimizer = opt

    def initialize(self, weight_initializer, bias_initializer):
        """
        Reinitialize weights and bias using provided initializers.
        Bias is stored in the last row of the weight matrix.
        """
        # fan_in excludes bias row, fan_out is output size
        fan_in = self.input_size
        fan_out = self.output_size

        # Initialize main weights (without bias row)
        W = weight_initializer.initialize(
            shape=(self.input_size, self.output_size),
            fan_in=fan_in,
            fan_out=fan_out
        )
        # Initialize bias row
        b = bias_initializer.initialize(
            shape=(1, self.output_size),
            fan_in=fan_in,
            fan_out=fan_out
        )
        # Stack weights and bias into full weight matrix
        self.weights = np.vstack([W, b])

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        # Extend input with 1s for bias term: shape (batch_size, input_size+1)
        ones = np.ones((input_tensor.shape[0], 1))
        self.input_ext = np.concatenate([input_tensor, ones], axis=1)
        # Linear transformation including bias
        return self.input_ext @ self.weights

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        # Compute gradient w.r.t. weights (including bias row)
        self.gradient_weights = self.input_ext.T @ error_tensor

        # Compute error propagated to inputs using current weights (before update)
        error_ext = error_tensor @ self.weights.T

        # Update weights if optimizer is provided
        if self._optimizer is not None:
            # Update includes bias row as part of weights
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)

        # Discard bias component to match input dimensions
        return error_ext[:, :-1]

    def calculate_regularization_loss(self):
        if self._optimizer is not None and self._optimizer.regularizer is not None:
            return self._optimizer.regularizer.norm(self.weights)
        return 0