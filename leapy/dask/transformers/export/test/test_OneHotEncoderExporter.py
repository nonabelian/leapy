import numpy as np
import dask.array as da

from ... import OneHotEncoder
from .. import OneHotEncoderExporter


def test_ohe_export_function():
    ohe = OneHotEncoder(sparse=False)
    X_np = np.array([['a'], ['b']])
    X = da.from_array(X_np, chunks=X_np.shape)
    X_act = ohe.fit_transform(X).compute()
    ohe_runtime = OneHotEncoderExporter.to_runtime(ohe)
    X_exp = ohe_runtime.transform(X_np)

    assert np.all(X_exp == X_act)


def test_add_to_class_export():
    ohe = OneHotEncoder(sparse=False)
    X_np = np.array([['a'], ['b']])
    X = da.from_array(X_np, chunks=X_np.shape)
    X_act = ohe.fit_transform(X).compute()
    ohe_runtime = ohe.to_runtime()
    X_exp = ohe_runtime.transform(X_np)

    assert np.all(X_exp == X_act)
