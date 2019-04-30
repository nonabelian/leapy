import numpy as np

from .. import SelectorRuntime

def test_transform():
    sel = SelectorRuntime()
    X = np.array([[1], [2]])
    X_act = sel.fit_transform(X)
    X_exp = np.array([[1], [2]])

    assert np.all(X_exp == X_act)
