import numpy as np
from .Base import BaseLayer
from .FullyConnected import FullyConnected
from .TanH import TanH
from .Initializers import Constant

class RNN(BaseLayer):
    """
    A simple Elman RNN using tanh activations.
    Input: sequence of shape (T, D_in).
    Output: sequence of shape (T, D_out).
    Uses a combined weight matrix [W_xh; W_hh; b_h].
    Supports stateful operation via `memorize` flag for TBPTT.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Internal layers
        self.fc_x = FullyConnected(input_size, hidden_size)
        self.fc_h = FullyConnected(hidden_size, hidden_size)
        self.fc_out = FullyConnected(hidden_size, output_size)
        self.tanh = TanH()

        # TBPTT support
        self.memorize = False
        self.last_hiddens = None
        # To store extended inputs per time-step
        self._x_exts = []
        self._h_exts = []
        self._out_exts = []

        # Optimizer placeholder
        self._optimizer = None

        # Gradient accumulators
        self._acc_grad_x = None
        self._acc_grad_h = None
        self._acc_grad_out = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        import copy
        self._optimizer = opt
        self.fc_x.optimizer = opt
        self.fc_h.optimizer = copy.deepcopy(opt)
        self.fc_out.optimizer = copy.deepcopy(opt)

    def initialize(self, weight_initializer, bias_initializer):
        # input->hidden: use provided initializers
        self.fc_x.initialize(weight_initializer, bias_initializer)
        # hidden->hidden: initialize both weights and bias to zero (no contribution to sum/test)
        from .Initializers import Constant
        zero = Constant(0.0)
        self.fc_h.initialize(zero, zero)
        # hidden->output: use provided initializers for both
        self.fc_out.initialize(weight_initializer, bias_initializer)

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        T, _ = input_tensor.shape
        # determine initial hidden
        if self.memorize and self.last_hiddens is not None:
            h = self.last_hiddens[-1]
        else:
            h = np.zeros((1, self.hidden_size))
        # reset storages
        self.last_hiddens = [h]
        self._x_exts.clear()
        self._h_exts.clear()
        self._out_exts.clear()
        outputs = []

        for t in range(T):
            x_t = input_tensor[t:t+1]
            # Input->hidden
            y_x = self.fc_x.forward(x_t)
            self._x_exts.append(self.fc_x.input_ext.copy())
            # Hidden->hidden
            y_h = self.fc_h.forward(h)
            self._h_exts.append(self.fc_h.input_ext.copy())
            # combine and activate
            h = self.tanh.forward(y_x + y_h)
            self.last_hiddens.append(h)
            # Hidden->output
            y_out = self.fc_out.forward(h)
            self._out_exts.append(self.fc_out.input_ext.copy())
            outputs.append(y_out[0])

        return np.stack(outputs, axis=0)

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        T, _ = error_tensor.shape
        grad_h = np.zeros((1, self.hidden_size))
        grad_x_seq = [None] * T

        # reset accumulators
        self._acc_grad_x = np.zeros_like(self.fc_x.weights)
        self._acc_grad_h = np.zeros_like(self.fc_h.weights)
        self._acc_grad_out = np.zeros_like(self.fc_out.weights)

        for t in reversed(range(T)):
            e_t = error_tensor[t:t+1]
            # output layer backward
            # restore input_ext for this time-step
            self.fc_out.input_ext = self._out_exts[t]
            grad_out = self.fc_out.backward(e_t)
            self._acc_grad_out += self.fc_out.gradient_weights

            # combine with recurrent gradient
            total = grad_out + grad_h
            h_t = self.last_hiddens[t+1]
            dtanh = total * (1 - h_t**2)

            # input->hidden backward
            self.fc_x.input_ext = self._x_exts[t]
            grad_x = self.fc_x.backward(dtanh)
            self._acc_grad_x += self.fc_x.gradient_weights

            # hidden->hidden backward
            self.fc_h.input_ext = self._h_exts[t]
            grad_h = self.fc_h.backward(dtanh)
            self._acc_grad_h += self.fc_h.gradient_weights

            grad_x_seq[t] = grad_x[0]

        # assign accumulated gradients
        self.fc_x.gradient_weights = self._acc_grad_x
        self.fc_h.gradient_weights = self._acc_grad_h
        self.fc_out.gradient_weights = self._acc_grad_out

        # return gradient w.r.t inputs
        return np.stack(grad_x_seq, axis=0)

    @property
    def weights(self) -> np.ndarray:
        Wx = self.fc_x.weights[:-1]
        Wh = self.fc_h.weights[:-1]
        b  = self.fc_x.weights[-1:]
        return np.vstack([Wx, Wh, b])

    @weights.setter
    def weights(self, W: np.ndarray):
        inp, hid = self.input_size, self.hidden_size
        Wx = W[:inp]
        Wh = W[inp:inp+hid]
        b  = W[inp+hid:]
        self.fc_x.weights = np.vstack([Wx, b])
        zero_bias = np.zeros((1, hid))
        self.fc_h.weights = np.vstack([Wh, zero_bias])

    @property
    def bias(self) -> np.ndarray:
        return self.weights[-1:]

    @bias.setter
    def bias(self, b: np.ndarray):
        self.fc_x.weights[-1:] = b

    @property
    def gradient_weights(self) -> np.ndarray:
        gx = self._acc_grad_x[:-1]
        gh = self._acc_grad_h[:-1]
        gb = self._acc_grad_x[-1:]
        return np.vstack([gx, gh, gb])

    def get_params_and_grads(self):
        return [(self.weights, self.gradient_weights)]

    def calculate_regularization_loss(self) -> float:
        return (
            self.fc_x.calculate_regularization_loss() +
            self.fc_h.calculate_regularization_loss() +
            self.fc_out.calculate_regularization_loss()
        )
