from datetime import datetime

import numpy as np
import dask.array as da
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from .HourOfDayFromDatetimeStringRuntime \
        import HourOfDayFromDatetimeStringRuntime


class HourOfDayFromDatetimeString(BaseEstimator, TransformerMixin):
    """ Extract hour from timestamp of form '2019-03-25 19:00:00'."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def extract_day(ts):
            # XXX: bizarre dask test_data
            if ts[0] == '1':
                return [0]
            if isinstance(ts[0], int):
                return [0]
            return np.array([[datetime.strptime(t[0], '%Y-%m-%d %H:%M:%S')
                                      .hour]
                             for t in ts])

        return  X.map_blocks(extract_day, dtype=np.object)

    def to_runtime(self):
        # No copying necessary for this simple transformer
        return HourOfDayFromDatetimeStringRuntime()
