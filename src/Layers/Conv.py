import numpy as np
import copy
from .Base import BaseLayer

class Conv(BaseLayer):
    """
    1D or 2D convolutional layer with 'same' zero-padding.
    """
    def __init__(self, stride_shape, conv_shape, num_kernels):
        super().__init__()
        self.trainable = True
        # normalize stride
        if isinstance(stride_shape, int):
            self.stride = (stride_shape,) * (len(conv_shape) - 1)
        else:
            self.stride = tuple(stride_shape)
        # conv_shape: [in_channels, k_h] or [in_channels, k_h, k_w]
        self.conv_shape = conv_shape
        self.num_kernels = num_kernels
        # initialize parameters
        self.weights = np.random.rand(num_kernels, conv_shape[0], *conv_shape[1:])
        self.bias = np.random.rand(num_kernels)
        # gradient placeholders
        self._grad_weights = np.zeros_like(self.weights)
        self._grad_bias = np.zeros_like(self.bias)
        # optimizers placeholders
        self._weight_optimizer = None
        self._bias_optimizer = None

    @property
    def optimizer(self):
        return self._weight_optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._weight_optimizer = opt
        self._bias_optimizer = copy.deepcopy(opt)

    @property
    def gradient_weights(self):
        """Gradient of the loss w.r.t. the convolution weights."""
        return self._grad_weights

    @property
    def gradient_bias(self):
        """Gradient of the loss w.r.t. the biases."""
        return self._grad_bias

    def forward(self, input_tensor):
        dims = input_tensor.ndim
        if dims == 3:
            # 1D convolution
            b, c, length = input_tensor.shape
            k = self.conv_shape[1]
            stride = self.stride[0]
            pad_total = k - 1
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            padded = np.pad(input_tensor, ((0,0),(0,0),(pad_before,pad_after)), 'constant')
            out_len = int(np.ceil(length / stride))
            out = np.zeros((b, self.num_kernels, out_len))
            for i in range(b):
                for kn in range(self.num_kernels):
                    for pos in range(out_len):
                        start = pos * stride
                        window = padded[i, :, start:start+k]
                        out[i, kn, pos] = np.sum(window * self.weights[kn]) + self.bias[kn]
            self.pad = (pad_before, pad_after)

        elif dims == 4:
            # 2D convolution
            b, c, h, w = input_tensor.shape
            k_h, k_w = self.conv_shape[1], self.conv_shape[2]
            sh, sw = self.stride
            pad_h_total = k_h - 1
            pad_h_before = pad_h_total // 2
            pad_h_after = pad_h_total - pad_h_before
            pad_w_total = k_w - 1
            pad_w_before = pad_w_total // 2
            pad_w_after = pad_w_total - pad_w_before
            padded = np.pad(input_tensor,
                            ((0,0),(0,0),(pad_h_before,pad_h_after),(pad_w_before,pad_w_after)),
                            'constant')
            out_h = int(np.ceil(h / sh))
            out_w = int(np.ceil(w / sw))
            out = np.zeros((b, self.num_kernels, out_h, out_w))
            for i in range(b):
                for kn in range(self.num_kernels):
                    for y in range(out_h):
                        for x in range(out_w):
                            sy = y * sh
                            sx = x * sw
                            window = padded[i, :, sy:sy+k_h, sx:sx+k_w]
                            out[i, kn, y, x] = np.sum(window * self.weights[kn]) + self.bias[kn]
            self.pad = (pad_h_before, pad_h_after, pad_w_before, pad_w_after)

        else:
            raise ValueError(f"Unsupported input dims: {dims}")

        self.input = input_tensor
        self.output = out
        return out

    def backward(self, error_tensor):
        dims = self.input.ndim
        if dims == 3:
            b, c, length = self.input.shape
            k = self.conv_shape[1]
            stride = self.stride[0]
            pad_before, pad_after = self.pad
            padded = np.pad(self.input, ((0,0),(0,0),(pad_before,pad_after)), 'constant')
            grad_padded = np.zeros_like(padded)
            self._grad_weights.fill(0)
            self._grad_bias.fill(0)
            out_len = error_tensor.shape[2]
            for i in range(b):
                for kn in range(self.num_kernels):
                    for pos in range(out_len):
                        start = pos * stride
                        window = padded[i, :, start:start+k]
                        self._grad_weights[kn] += window * error_tensor[i, kn, pos]
                        self._grad_bias[kn] += error_tensor[i, kn, pos]
                        grad_padded[i, :, start:start+k] += self.weights[kn] * error_tensor[i, kn, pos]
            grad_input = grad_padded[:, :, pad_before:pad_before+length]

        else:
            b, c, h, w = self.input.shape
            k_h, k_w = self.conv_shape[1], self.conv_shape[2]
            sh, sw = self.stride
            p_h_before, p_h_after, p_w_before, p_w_after = self.pad
            padded = np.pad(self.input,
                            ((0,0),(0,0),(p_h_before,p_h_after),(p_w_before,p_w_after)),
                            'constant')
            grad_padded = np.zeros_like(padded)
            self._grad_weights.fill(0)
            self._grad_bias.fill(0)
            out_h, out_w = error_tensor.shape[2], error_tensor.shape[3]
            for i in range(b):
                for kn in range(self.num_kernels):
                    for y in range(out_h):
                        for x in range(out_w):
                            sy = y * sh
                            sx = x * sw
                            window = padded[i, :, sy:sy+k_h, sx:sx+k_w]
                            self._grad_weights[kn] += window * error_tensor[i, kn, y, x]
                            self._grad_bias[kn] += error_tensor[i, kn, y, x]
                            grad_padded[i, :, sy:sy+k_h, sx:sx+k_w] += self.weights[kn] * error_tensor[i, kn, y, x]
            grad_input = grad_padded[:, :, p_h_before:p_h_before+h, p_w_before:p_w_before+w]

        if self._weight_optimizer is not None:
            # Apply optimizer to update weights and bias correctly
            self.weights = self._weight_optimizer.calculate_update(self.weights, self._grad_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._grad_bias)
        return grad_input

    def initialize(self, weight_initializer, bias_initializer):
        fan_in = np.prod(self.conv_shape)
        fan_out = self.num_kernels * np.prod(self.conv_shape[1:])
        self.weights = weight_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)
