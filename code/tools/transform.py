import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LogTransform:
    def __init__(self, y):
        self.miny = float(np.min(y))
        miny2 = sorted(set(np.array(y.squeeze())))[1]
        self.eps = (miny2 - self.miny) / 10
        self.bias = 0
        self.bias = np.max(self.transform(y)) + 1

    def transform(self, y):
        return np.log(y - self.miny + self.eps) - self.bias

    def inverse_transform(self, logy):
        return np.exp(logy + self.bias) + self.miny - self.eps


class CustomLogTargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.log_transformer = None

    def fit(self, target):
        self.log_transformer = LogTransform(target)
        return self

    def transform(self, target):
        return self.log_transformer.transform(target)

    def inverse_transform(self, log_target):
        return self.log_transformer.inverse_transform(log_target)


def magnitude(x: float) -> int:
    """Returns negative order of magnitude of a nonzero floating point number, i.e.
        x = a*10^-n -> n

    Parameters
    ----------
    x : float
        Nonzero floating point number.

    Returns
    -------
    int
        Order of magnitude.

    """
    return -int(np.floor(np.log10(x)))


def magnitude_inv(n: int) -> float:
    """(Pseudo-)inverse transformation of 'magnitude', i.e. n -> 10^{-n}

    Parameters
    ----------
    n : int
        Neg. order of magnitude.

    Returns
    -------
    float

    """
    return 10 ** float(-n)


def magnitude_transform(a: np.array) -> np.array:
    """Vectorized version of magnitude
    

    Parameters
    ----------
    a : np.array of nonzero floats
        Array of numbers to transform.

    Returns
    -------
    np.array of ints
        Neg. magnitudes of numbers in a.

    """
    return -np.vectorize(magnitude)(a)


def magnitude_inv_transform(a: np.array) -> np.array:
    """Vectorized version of magnitude_inf
    

    Parameters
    ----------
    a : np.array of nonnegative integers

    Returns
    -------
    np.array of floats

    """
    return np.vectorize(magnitude_inv)(a)


class TargetMagnitudeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, target):
        return self

    def transform(self, target):
        if isinstance(target, pd.Series):
            return target.apply(magnitude)
        else:
            return magnitude_transform(target)

    def inverse_transform(self, target):
        if isinstance(target, pd.Series):
            return target.apply(magnitude_inv)
        else:
            return magnitude_inv_transform(target)
