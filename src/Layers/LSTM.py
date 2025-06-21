# Layers/LSTM.py

import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected

# Sigmoid helper
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class LSTM(BaseLayer):
    """
    Long Short-Term Memory layer.
    Processes a sequence of input vectors with recurrence.
    Input: tensor of shape (T, D_in).
    Output: tensor of shape (T, D_out).
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Single FC on concatenated [x, h] → 4*hidden (gates)
        self.fc = FullyConnected(input_size + hidden_size, 4 * hidden_size)
        # Output FC: hidden → output_size
        self.fc_out = FullyConnected(hidden_size, output_size)

        # Recurrence states
        self.h = None
        self.c = None

        # Store intermediate values for backprop
        self._saved = {
            'concat': [],   # [x||h]
            'gates': [],    # raw gate pre-activations
            'c_prev': [],   # previous cell state
            'c': []         # cell state after update
        }

        # optimizer placeholder
        self._optimizer = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        import copy
        self._optimizer = opt
        self.fc.optimizer = opt
        self.fc_out.optimizer = copy.deepcopy(opt)

    def initialize(self, weight_initializer, bias_initializer):
        self.fc.initialize(weight_initializer, bias_initializer)
        self.fc_out.initialize(weight_initializer, bias_initializer)

    @property
    def weights(self):
        # expose the combined gate weights
        return self.fc.weights

    @weights.setter
    def weights(self, w):
        self.fc.weights = w

    @property
    def bias(self):
        return self.fc.bias

    @bias.setter
    def bias(self, b):
        self.fc.bias = b

    @property
    def gradient_weights(self):
        return self.fc.gradient_weights

    @property
    def gradient_bias(self):
        return self.fc.gradient_bias

    def forward(self, input_tensor):
        # input_tensor: (T, D_in)
        T, D_in = input_tensor.shape
        H = self.hidden_size

        # initialize states
        self.h = np.zeros((1, H))
        self.c = np.zeros((1, H))

        # clear saved
        for key in self._saved:
            self._saved[key].clear()

        outputs = []
        for t in range(T):
            x_t = input_tensor[t:t+1]              # (1, D_in)
            concat = np.concatenate([x_t, self.h], axis=1)  # (1, D_in+H)
            self._saved['concat'].append(concat)
            gates_raw = self.fc.forward(concat)   # (1, 4H)
            self._saved['gates'].append(gates_raw)
            c_prev = self.c.copy()
            self._saved['c_prev'].append(c_prev)

            # split gates
            i_raw, f_raw, o_raw, g_raw = np.split(gates_raw, 4, axis=1)
            i = _sigmoid(i_raw)
            f = _sigmoid(f_raw)
            o = _sigmoid(o_raw)
            g = np.tanh(g_raw)

            # update cell and hidden
            self.c = f * c_prev + i * g
            self.h = o * np.tanh(self.c)
            self._saved['c'].append(self.c.copy())

            # output
            y = self.fc_out.forward(self.h)       # (1, D_out)
            outputs.append(y[0])                  # flatten

        return np.stack(outputs, axis=0)

    def backward(self, error_tensor):
        # error_tensor: (T, D_out)
        T, _ = error_tensor.shape
        H = self.hidden_size
        D_in = self.input_size

        grad_h = None
        grad_c = np.zeros((1, H))
        grad_x = [None] * T

        # backprop through time
        for t in reversed(range(T)):
            # output layer
            dy = error_tensor[t:t+1]            # (1, D_out)
            dh_out = self.fc_out.backward(dy)    # (1, H)
            dh = dh_out + (grad_h if grad_h is not None else 0)

            # retrieve saved
            gates_raw = self._saved['gates'][t]
            concat = self._saved['concat'][t]
            c_prev = self._saved['c_prev'][t]
            c_t = self._saved['c'][t]

            # split gates
            i_raw, f_raw, o_raw, g_raw = np.split(gates_raw, 4, axis=1)
            i = _sigmoid(i_raw)
            f = _sigmoid(f_raw)
            o = _sigmoid(o_raw)
            g = np.tanh(g_raw)

            # gradients
            d_c_t = grad_c + dh * o * (1 - np.tanh(c_t)**2)
            d_i = d_c_t * g
            d_g = d_c_t * i
            d_f = d_c_t * c_prev
            d_o = dh * np.tanh(c_t)

            # gate raw gradients
            di_raw = d_i * i * (1 - i)
            df_raw = d_f * f * (1 - f)
            do_raw = d_o * o * (1 - o)
            dg_raw = d_g * (1 - g**2)

            d_gates = np.concatenate([di_raw, df_raw, do_raw, dg_raw], axis=1)  # (1,4H)

            # back through combined FC
            dconcat = self.fc.backward(d_gates)    # (1, D_in+H)

            # split to x and h gradients
            d_x = dconcat[:, :D_in]    # (1, D_in)
            grad_h = dconcat[:, D_in:] # (1, H)
            grad_c = d_c_t * f         # (1, H)

            grad_x[t] = d_x[0]

        return np.stack(grad_x, axis=0)

    def calculate_regularization_loss(self):
        return self.fc.calculate_regularization_loss() + self.fc_out.calculate_regularization_loss()
