import numpy as np
from .BaseLayer import BaseLayer

class BatchNormalization(BaseLayer):
    def __init__(self, channels, momentum=0.9, epsilon=1e-12):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.momentum = momentum
        self.epsilon = epsilon

        # Initialize scale (gamma) and shift (beta)
        self.initialize(None, None)

        # Running (moving average) statistics
        self._running_mean = None
        self._running_var = None

        # Gradients
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias    = np.zeros_like(self.bias)

        # Optimizers for gamma and beta
        self._optimizer_weights = None
        self._optimizer_bias    = None

        # For reformatting between image (4D) and vector (2D)
        self._image_shape = None

    def initialize(self, weight_initializer=None, bias_initializer=None):
        # gamma = 1, beta = 0 regardless of external initializers
        self.weights = np.ones((1, self.channels))
        self.bias    = np.zeros((1, self.channels))

    @property
    def optimizer(self):
        return self._optimizer_weights

    @optimizer.setter
    def optimizer(self, opt):
        import copy
        self._optimizer_weights = opt
        self._optimizer_bias    = copy.deepcopy(opt)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def reformat(self, tensor):
        # Image-like 4D to 2D vector and back
        if tensor.ndim == 4:
            N, C, H, W = tensor.shape
            self._image_shape = tensor.shape
            # transpose to NHWC then flatten to (N*H*W, C)
            return tensor.transpose(0, 2, 3, 1).reshape(-1, C)
        elif tensor.ndim == 2:
            if self._image_shape is None:
                raise ValueError("Cannot reformat vector before seeing an image shape")
            N, C, H, W = self._image_shape
            # reshape back and transpose to NCHW
            return tensor.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            raise ValueError(f"Unexpected tensor ndim={tensor.ndim} in reformat")

    def forward(self, input_tensor):
        # 1) flatten input
        x_vec = self.reformat(input_tensor)

        # 2) compute batch statistics
        batch_mean = np.mean(x_vec, axis=0, keepdims=True)
        batch_var  = np.var(x_vec, axis=0, keepdims=True)

        # 3) update or use running stats
        if not self.testing_phase:
            if self._running_mean is None:
                self._running_mean = batch_mean
                self._running_var  = batch_var
            else:
                self._running_mean = self.momentum * self._running_mean + (1 - self.momentum) * batch_mean
                self._running_var  = self.momentum * self._running_var  + (1 - self.momentum) * batch_var
            mean, var = batch_mean, batch_var
        else:
            mean, var = self._running_mean, self._running_var

        # 4) normalize
        self.input_centered = x_vec - mean
        self.norm = self.input_centered / np.sqrt(var + self.epsilon)

        # 5) scale & shift
        out_vec = self.norm * self.weights + self.bias

        # 6) restore shape
        return self.reformat(out_vec)

    def backward(self, error_tensor):
        # 1) flatten gradient
        grad_vec = self.reformat(error_tensor)
        N = grad_vec.shape[0]

        # 2) parameter gradients
        self._gradient_weights = np.sum(grad_vec * self.norm, axis=0, keepdims=True)
        self._gradient_bias    = np.sum(grad_vec, axis=0, keepdims=True)

        # 3) backprop through normalization
        var = self._running_var if self.testing_phase else np.var(self.input_centered + np.mean(self.input_centered, axis=0), axis=0, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.epsilon)
        dx_norm = grad_vec * self.weights

        dvar = np.sum(dx_norm * self.input_centered * -0.5 * std_inv**3, axis=0, keepdims=True)
        dmean = np.sum(dx_norm * -std_inv, axis=0, keepdims=True) + dvar * np.mean(-2 * self.input_centered, axis=0, keepdims=True)

        dx = dx_norm * std_inv + dvar * 2 * self.input_centered / N + dmean / N

        # 4) parameter update
        if self._optimizer_weights is not None:
            self.weights = self._optimizer_weights.update(self.weights, self._gradient_weights)
            self.bias    = self._optimizer_bias.update(self.bias,       self._gradient_bias)

        # 5) restore shape
        return self.reformat(dx)
