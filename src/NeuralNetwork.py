import copy
import pickle

class NeuralNetwork:
    """
    Manages layers, data flow, training loop,
    supports per-layer optimizers plus weight/bias initializers.
    """
    def __init__(self, optimizer, weight_initializer, bias_initializer):
        """
        Args:
            optimizer: A prototype optimizer (e.g. SGD, Adam) – cloned per layer
            weight_initializer: instance of weight initializer
            bias_initializer: instance of bias initializer
        """
        self.optimizer = optimizer
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self._label = None
        self.loss = []

    def append_layer(self, layer):
        # assign optimizer copy for trainable layers
        if getattr(layer, 'trainable', False):
            layer.optimizer = copy.deepcopy(self.optimizer)
        # initialize weights/bias if supported
        if hasattr(layer, 'initialize'):
            layer.initialize(self.weight_initializer, self.bias_initializer)
        self.layers.append(layer)

    def forward(self):
        x, label = self.data_layer.next()
        self._label = label

        # Forward pass through all layers
        for layer in self.layers:
            x = layer.forward(x)

        # Compute loss
        loss = self.loss_layer.forward(x, label)

        # Add regularization loss from each trainable layer
        for layer in self.layers:
            if hasattr(layer, 'calculate_regularization_loss'):
                loss += layer.calculate_regularization_loss()

        return loss

    def backward(self):
        error = self.loss_layer.backward(self._label)
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def train(self, iterations):
        self.loss = []
        for _ in range(iterations):
            l = self.forward()
            self.loss.append(l)
            self.backward()
        return self.loss

    def test(self, input_tensor):
        x = input_tensor
        for layer in self.layers:
            x = layer.forward(x)
        return x

    @staticmethod
    def save(path, network):
        with open(path, 'wb') as f:
            pickle.dump(network, f)

    @staticmethod
    def load(path, data_layer):
        with open(path, 'rb') as f:
            net = pickle.load(f)
        net.data_layer = data_layer
        return net
