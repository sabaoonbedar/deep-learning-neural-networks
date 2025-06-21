__all__ = ["Helpers", "FullyConnected", "SoftMax", "ReLU", "Flatten", "TanH", "Sigmoid", "RNN",
           "Conv", "Pooling", "Initializers", "Dropout", "BatchNormalization", "Base", "LSTM"]

import sklearn.preprocessing as _skpre
_orig_ohe = _skpre.OneHotEncoder

# Legacy compatibility shim for OneHotEncoder

def OneHotEncoder(*args, **kwargs):
    """
    Shim around sklearn's OneHotEncoder to support legacy 'sparse' keyword.
    """
    # Simply forward 'sparse' if provided; do not translate to 'sparse_output'
    return _orig_ohe(*args, **kwargs)

# Monkey-patch for downstream imports
_skpre.OneHotEncoder = OneHotEncoder

