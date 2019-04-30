import numpy as np
import scipy.sparse as ss
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from .. import FeaturePipeline


class DummyTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, constant=1, sparse=True):
        self.constant = constant
        self.sparse = sparse

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        return ss.csr_matrix(self.constant * np.ones((X.shape[0], 1)))

    def fit_transform(self, X, y=None, **fit_kwargs):
        self.fit(X, y, **fit_kwargs)
        return self.transform(X)


def test_one_schema_drop():
    X_np = np.array([[0, 1], [1, 0]])

    pipe = FeaturePipeline([('dt', DummyTransformer(), [0])])

    X_exp = np.ones((X_np.shape[0], 1))
    X_act = pipe.fit_transform(X_np)

    assert np.all(X_exp == X_act.toarray())

    pipe = FeaturePipeline([('dt', DummyTransformer(), [0])], sparse=False)
    X_act = pipe.fit_transform(X_np)

    assert np.all(X_exp == X_act)


def test_one_schema_keep():
    X_np = np.array([[0, 1], [1, 0]])

    pipe = FeaturePipeline([('dt', DummyTransformer(), [0])],
                           drop=False)

    X_exp = np.hstack([X_np, np.ones((X_np.shape[0], 1))])
    X_act = pipe.fit_transform(X_np)
    assert np.all(X_exp == X_act.toarray())

    pipe = FeaturePipeline([('dt', DummyTransformer(), [0])],
                           drop=False,
                           sparse=False)
    X_exp = np.hstack([X_np, np.ones((X_np.shape[0], 1))])
    X_act = pipe.fit_transform(X_np)
    assert np.all(X_exp == X_act)


def test_many_schema_drop():
    X_np = np.array([[0, 1], [1, 0]])

    pipe = FeaturePipeline([('dt_1', DummyTransformer(), [0]),
                            ('dt_0', DummyTransformer(constant=0), [1])])
    X_exp = np.hstack([np.ones((X_np.shape[0], 1)),
                       np.zeros((X_np.shape[0], 1))])
    X_act = pipe.fit_transform(X_np)
    assert np.all(X_exp == X_act.toarray())

    pipe = FeaturePipeline([('dt_1', DummyTransformer(), [0]),
                            ('dt_0', DummyTransformer(constant=0), [1])],
                           sparse=False)
    X_exp = np.hstack([np.ones((X_np.shape[0], 1)),
                       np.zeros((X_np.shape[0], 1))])
    X_act = pipe.fit_transform(X_np)
    assert np.all(X_exp == X_act)


def test_scikit_api():
    pipe = FeaturePipeline([
        ('dt_1', DummyTransformer(), [0]),
        ('dt_0', DummyTransformer(constant=0), [1])])

    params = pipe.get_params()
    print(params)

    assert 'steps' in list(params.keys())
    assert 'drop' in list(params.keys())
    assert 'sparse' in list(params.keys())
    assert 'dt_1' in list(params.keys())
    assert 'dt_0' in list(params.keys())
    assert 'dt_1__constant' in list(params.keys())
    assert 'dt_0__constant' in list(params.keys())
    assert 'dt_1__sparse' in list(params.keys())
    assert 'dt_0__sparse' in list(params.keys())


def test_scikit_api_pipeline():
    pipe = Pipeline([
        ('fp', FeaturePipeline([
            ('dt_1', DummyTransformer(), [0]),
            ('dt_0', DummyTransformer(constant=0), [1])])),
        ('clf', DummyClassifier())])

    params = pipe.get_params()
    pipe.set_params(**params)

    assert 1 == 1
