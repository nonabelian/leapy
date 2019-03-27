from datetime import datetime

import numpy as np
import dask.array as da

from HourOfDayFromDatetimeRuntime import HourOfDayFromDatetimeRuntime


class HourOfDayFromDatetime(object):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def extract_day(ts):
            # XXX: bizarre dask test_data
            if isinstance(ts[0], int):
                return [datetime(2019, 1, 1)]
            return np.array([t.to_pydatetime().hour for t in ts])

        return da.apply_along_axis(extract_day, 1, X).reshape(-1, 1)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    @staticmethod
    def to_runtime(transformer):
        # No copying necessary for this simple transformer
        return HourOfDayFromDatetimeRuntime()
