from timeit import timeit
from string import ascii_lowercase
from random import choice
import warnings

import numpy as np
import dask.array as da
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from leapy.dask.transformers import OneHotEncoder


def test_fit_passes():
    X = da.from_array(np.array([['a'], ['b']]), chunks=(2, 1))
    ohe = OneHotEncoder()
    ohe.fit(X)

    assert 1 == 1

def test_transform():
    X = da.from_array(np.array([['a'], ['b']]), chunks=(2, 1))
    ohe = OneHotEncoder()
    ohe.fit(X)

    X_act = ohe.transform(X).compute()
    X_exp = np.array([[1, 0], [0, 1]])

    assert np.all(X_exp == X_act)

def test_scikit_learn_compatibility():
    X = da.from_array(np.array([['a'], ['b']]), chunks=(2, 1))
    y = da.from_array(np.array([0, 1]), chunks=(2,))

    pipe = Pipeline([('ohe', OneHotEncoder()),
                     ('clf', DummyClassifier(strategy='constant',
                                             constant=1))
                    ])
    pipe.fit(X, y)

    y_act = pipe.predict(X)
    y_exp = np.array([1, 1])

    assert np.all(y_exp == y_act)

def test_speed():
    threshold = 10 * 1E-6  # 10 microseconds
    categories = [''.join(choice(ascii_lowercase) for _ in range(5))
                  for i in range(2000)]

    X_np = np.array(categories).reshape(-1, 1)
    X = da.from_array(X_np, chunks=(100, X_np.shape[1]))

    ohe = OneHotEncoder()
    ohe.fit(X)
    x = choice(X).reshape(1, -1)

    speed = timeit("ohe.transform(x).compute()" ,
                   globals=locals(),
                   number=1)

    if (speed >= threshold):
        warnings.warn("Slow performance {0:0.2f} >= {1:0.2f} microseconds"
                      .format(speed * 1E6, threshold * 1E6))
