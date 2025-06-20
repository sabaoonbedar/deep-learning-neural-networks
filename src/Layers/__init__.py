__all__ = ["Helpers", "FullyConnected", "SoftMax", "ReLU", "Conv", "Pooling", "Initializers", "Flatten", "Base"]
import sklearn.preprocessing as _skpre
_orig_ohe = _skpre.OneHotEncoder

def OneHotEncoder(*args, **kwargs):
    # Translate legacy 'sparse' → modern 'sparse_output'
    if 'sparse' in kwargs:
        kwargs['sparse_output'] = kwargs.pop('sparse')
    return _orig_ohe(*args, **kwargs)

_skpre.OneHotEncoder = OneHotEncoder