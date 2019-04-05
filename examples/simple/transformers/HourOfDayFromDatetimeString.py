from datetime import datetime

import numpy as np
import dask.array as da

from .HourOfDayFromDatetimeStringRuntime \
        import HourOfDayFromDatetimeStringRuntime


class HourOfDayFromDatetimeString(object):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def extract_day(ts):
            # XXX: bizarre dask test_data
            if ts[0] == '1':
                return [0]
            if isinstance(ts[0], int):
                return [0]
            return np.array([datetime.strptime(t, '%Y-%m-%d %H:%M:%S').hour
                             for t in ts])

        return da.apply_along_axis(extract_day, 1, X).reshape(-1, 1)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    def to_runtime(self):
        # No copying necessary for this simple transformer
        return HourOfDayFromDatetimeStringRuntime()