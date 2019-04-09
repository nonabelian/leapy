import numpy as np

from .. import OneHotEncoderRuntime

def test_transform():
    ohe = OneHotEncoderRuntime()
    X = np.array([['a'], ['b']])
    X_act = ohe.fit_transform(X)
    X_exp = np.array([[1, 0], [0, 1]])

    assert np.all(X_exp == X_act)
