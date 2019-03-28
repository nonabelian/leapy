from datetime import datetime
from datetime import timedelta

import numpy as np
import dask.array as da

from simple.transformers import HourOfDayFromDatetimeString


def test_hod_transform():
    X_np = np.array([['2019-03-25 19:00:00'],
                     ['2019-03-25 09:00:00']])
    X = da.from_array(X_np, chunks=X_np.shape)
    hod = HourOfDayFromDatetimeString()
    hours_exp = np.array([[19], [9]])
    hours_act = hod.transform(X).compute()

    assert np.all(hours_exp == hours_act)


def test_hod_export():
    X_np = np.array([['2019-03-25 19:00:00'],
                     ['2019-03-25 09:00:00']])
    hod = HourOfDayFromDatetimeString()
    hours_exp = np.array([[19], [9]])
    hod_runtime = hod.to_runtime()
    hours_act = hod_runtime.transform(X_np)

    assert np.all(hours_exp == hours_act)
