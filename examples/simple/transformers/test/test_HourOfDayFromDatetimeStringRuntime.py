from datetime import datetime
from datetime import timedelta
from timeit import timeit
import warnings

import numpy as np
import dask.array as da

from simple.transformers import HourOfDayFromDatetimeStringRuntime


def test_hodr_transform():
    X_np = np.array([['2019-03-25 19:00:00'],
                     ['2019-03-25 09:00:00']])
    hod = HourOfDayFromDatetimeStringRuntime()
    hours_exp = np.array([[19], [9]])
    hours_act = hod.transform(X_np)

    assert np.all(hours_exp == hours_act)


def test_hodr_transform_speed():
    X_np = np.array([['2019-03-25 19:00:00']])
    hod = HourOfDayFromDatetimeStringRuntime()

    speed = timeit("hod.transform(X_np)",
                   globals=locals(),
                   number=1000) / 1000.

    threshold = 10 * 1E-6
    if (speed > threshold):
        warnings.warn("Slow Transformer: speed {0:0.2f} microseconds"
                      " | threshold {1:0.2f} microseconds".
                     format(speed*1E6, threshold*1E6))

