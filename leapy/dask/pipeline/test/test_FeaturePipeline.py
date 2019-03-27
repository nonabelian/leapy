from unittest import TestCase

import numpy as np
import dask.array as da

from leapy.dask.pipeline import FeaturePipeline


class DummyTransformer(object):

    def __init__(self, constant=1):
        self.constant = constant

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        return self.constant * da.ones((X.shape[0], 1),
                                       chunks=(X.chunks[0], 1))

    def fit_transform(self, X, y=None, **fit_kwargs):
        self.fit(X, y, **fit_kwargs)
        return self.transform(X)


class TestFeaturePipeline(TestCase):

    def test_one_schema_drop(self):
        X_np = np.array([[0, 1], [1, 0]])
        X = da.from_array(X_np, chunks=X_np.shape)

        pipe = FeaturePipeline([('dt', DummyTransformer(), [0])])

        X_exp = np.ones((X_np.shape[0], 1))
        X_act = pipe.fit_transform(X).compute()

        self.assertTrue(np.all(X_exp == X_act))

    def test_one_schema_keep(self):
        X_np = np.array([[0, 1], [1, 0]])
        X = da.from_array(X_np, chunks=X_np.shape)

        pipe = FeaturePipeline([('dt', DummyTransformer(), [0])],
                               drop=False)

        X_exp = np.hstack([X_np, np.ones((X_np.shape[0], 1))])
        X_act = pipe.fit_transform(X).compute()

        self.assertTrue(np.all(X_exp == X_act))

    def test_many_schema_drop(self):
        X_np = np.array([[0, 1], [1, 0]])
        X = da.from_array(X_np, chunks=X_np.shape)

        pipe = FeaturePipeline([('dt_1', DummyTransformer(), [0]),
                                ('dt_0', DummyTransformer(constant=0), [1])])

        X_exp = np.hstack([np.ones((X_np.shape[0], 1)),
                           np.zeros((X_np.shape[0], 1))])
        X_act = pipe.fit_transform(X).compute()

        self.assertTrue(np.all(X_exp == X_act))
