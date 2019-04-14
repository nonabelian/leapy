import numpy as np

from .. import SelectorRuntime

def test_transform():
    sel = SelectorRuntime()
    X = np.array([['a'], ['b']])
    X_act = sel.fit_transform(X)
    X_exp = np.array([['a'], ['b']])

    assert np.all(X_exp == X_act)
