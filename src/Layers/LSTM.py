import numpy as np
from .Base import BaseLayer

class LSTM(BaseLayer):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight initialization
        self.weights = np.random.randn(input_size + hidden_size, 4 * hidden_size)
        self.bias = np.zeros(4 * hidden_size)

        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)

        self._optimizer = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    def initialize(self, weight_initializer, bias_initializer):
        fan_in = self.input_size + self.hidden_size
        fan_out = 4 * self.hidden_size
        self.weights = weight_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    def forward(self, input_tensor):
        batch_size, seq_len, _ = input_tensor.shape
        self.h = np.zeros((batch_size, self.hidden_size))
        self.c = np.zeros((batch_size, self.hidden_size))

        self.inputs = []
        self.gates = []
        self.outputs = []

        for t in range(seq_len):
            x_t = input_tensor[:, t, :]
            combined = np.concatenate((x_t, self.h), axis=1)
            gates = combined @ self.weights + self.bias

            i = self._sigmoid(gates[:, :self.hidden_size])
            f = self._sigmoid(gates[:, self.hidden_size:2*self.hidden_size])
            o = self._sigmoid(gates[:, 2*self.hidden_size:3*self.hidden_size])
            g = np.tanh(gates[:, 3*self.hidden_size:])

            self.c = f * self.c + i * g
            self.h = o * np.tanh(self.c)

            self.inputs.append(combined)
            self.gates.append((i, f, o, g))
            self.outputs.append(self.h)

        return np.stack(self.outputs, axis=1)

    def backward(self, error_tensor):
        batch_size, seq_len, _ = error_tensor.shape
        d_next_h = np.zeros((batch_size, self.hidden_size))
        d_next_c = np.zeros((batch_size, self.hidden_size))

        grad_input = []
        self.gradient_weights.fill(0)
        self.gradient_bias.fill(0)

        for t in reversed(range(seq_len)):
            dh = error_tensor[:, t, :] + d_next_h
            i, f, o, g = self.gates[t]
            combined = self.inputs[t]

            tanh_c = np.tanh(self.c)
            do = dh * tanh_c * o * (1 - o)
            dc = dh * o * (1 - tanh_c ** 2) + d_next_c
            di = dc * g * i * (1 - i)
            dg = dc * i * (1 - g ** 2)
            df = dc * self.c * f * (1 - f)

            d_gates = np.concatenate((di, df, do, dg), axis=1)

            self.gradient_weights += combined.T @ d_gates
            self.gradient_bias += np.sum(d_gates, axis=0)

            d_combined = d_gates @ self.weights.T
            d_next_h = d_combined[:, self.input_size:]
            d_next_c = dc * f

            grad_input.append(d_combined[:, :self.input_size])

        grad_input.reverse()
        return np.stack(grad_input, axis=1)

    def calculate_regularization_loss(self):
        if self.optimizer and hasattr(self.optimizer, 'regularizer'):
            return self.optimizer.regularizer.norm(self.weights)
        return 0

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
