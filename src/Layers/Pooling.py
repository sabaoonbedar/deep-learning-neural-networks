import numpy as np
from .Base import BaseLayer

class Pooling(BaseLayer):
    """
    2D max-pooling layer with valid padding (no padding).
    Supports overlapping pooling (stride < pool size).
    """
    def __init__(self, stride_shape, pool_shape):
        super().__init__()
        # Non-trainable layer
        self.trainable = False
        # Normalize stride and pool shape to tuples (h, w)
        if isinstance(stride_shape, int):
            self.stride = (stride_shape, stride_shape)
        else:
            self.stride = tuple(stride_shape)
        if isinstance(pool_shape, int):
            self.pool_shape = (pool_shape, pool_shape)
        else:
            self.pool_shape = tuple(pool_shape)
        # To store absolute max indices for backward
        self.max_indices = {}

    def forward(self, input_tensor):
        """
        input_tensor: shape (batch_size, channels, height, width)
        returns: (batch_size, channels, out_h, out_w)
        """
        self.input = input_tensor
        b, c, h, w = input_tensor.shape
        ph, pw = self.pool_shape
        sh, sw = self.stride
        out_h = (h - ph) // sh + 1
        out_w = (w - pw) // sw + 1
        out = np.zeros((b, c, out_h, out_w))
        self.max_indices.clear()
        for i in range(b):
            for ch in range(c):
                for y in range(out_h):
                    for x in range(out_w):
                        y0 = y * sh
                        x0 = x * sw
                        window = input_tensor[i, ch, y0:y0+ph, x0:x0+pw]
                        # find absolute indices
                        local_idx = np.unravel_index(np.argmax(window), window.shape)
                        abs_y = y0 + local_idx[0]
                        abs_x = x0 + local_idx[1]
                        self.max_indices[(i, ch, y, x)] = (abs_y, abs_x)
                        out[i, ch, y, x] = window[local_idx]
        return out

    def backward(self, error_tensor):
        """
        error_tensor: shape (batch_size, channels, out_h, out_w)
        returns: gradient wrt input, shape same as input
        """
        b, c, h, w = self.input.shape
        grad_input = np.zeros_like(self.input)
        # Distribute gradients to the max locations
        for (i, ch, y, x), (abs_y, abs_x) in self.max_indices.items():
            grad_input[i, ch, abs_y, abs_x] += error_tensor[i, ch, y, x]
        return grad_input
