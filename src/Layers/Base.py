class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.testing_phase = False  # Used in Dropout/BatchNorm

    def calculate_regularization_loss(self):
        return 0  # Default for non-trainable layers
