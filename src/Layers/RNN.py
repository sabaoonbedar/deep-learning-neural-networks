import numpy as np
from .FullyConnected import FullyConnected
from .TanH import TanH
class RNN:
    """
    A simple Elman RNN using tanh activations.
    """
    def __init__(self, input_size, hidden_size, output_size, sequence_length, optimizer, weight_initializer, bias_initializer):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.fc_x = FullyConnected(input_size, hidden_size)
        self.fc_h = FullyConnected(hidden_size, hidden_size)
        self.fc_out = FullyConnected(hidden_size, output_size)

        for layer in [self.fc_x, self.fc_h, self.fc_out]:
            layer.optimizer = optimizer
            layer.initialize(weight_initializer, bias_initializer)

        self.tanh = TanH()

        self.last_inputs = []
        self.last_hiddens = []

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        h = np.zeros((batch_size, self.hidden_size))
        self.last_hiddens = [h]
        self.last_inputs = []

        for t in range(self.sequence_length):
            x_t = input_tensor[:, t, :]
            self.last_inputs.append(x_t)
            h = self.tanh.forward(self.fc_x.forward(x_t) + self.fc_h.forward(h))
            self.last_hiddens.append(h)

        return self.fc_out.forward(h)

    def backward(self, error_tensor):
        grad_out = self.fc_out.backward(error_tensor)
        grad_h = grad_out

        for t in reversed(range(self.sequence_length)):
            h = self.last_hiddens[t + 1]
            h_prev = self.last_hiddens[t]
            x_t = self.last_inputs[t]

            dtanh = grad_h * (1 - h ** 2)  # tanh derivative
            grad_x = self.fc_x.backward(dtanh)
            grad_h = self.fc_h.backward(dtanh)

        return None  # If embedding in a larger model, return grad_x sequence

