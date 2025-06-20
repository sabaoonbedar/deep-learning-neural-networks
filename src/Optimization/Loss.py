import numpy as np

class CrossEntropyLoss:
    """
    Computes cross-entropy loss and its gradient for classification.
    """
    def __init__(self):
        # will hold clipped predictions and true labels
        self.prediction = None
        self.label = None

    def forward(self, prediction: np.ndarray, label: np.ndarray) -> float:
        """
        Compute the total cross-entropy loss over the batch:
            L = -sum(label * log(prediction_clipped))
        We clip to [eps, 1–eps] with eps = machine epsilon for stability.
        """
        # machine epsilon for this dtype
        eps = np.finfo(prediction.dtype).eps
        # clip into (eps, 1–eps)
        pred_clipped = np.clip(prediction, eps, 1. - eps)

        # store for backward
        self.prediction = pred_clipped
        self.label = label

        # total loss (sum over all examples and classes)
        return -np.sum(label * np.log(pred_clipped))

    def backward(self, label: np.ndarray) -> np.ndarray:
        """
        Gradient of L w.r.t. the inputs 'prediction':
            dL/dp = -(label / prediction_clipped)
        """
        # note: no 1/N here since forward did not average
        return - (label / self.prediction)
