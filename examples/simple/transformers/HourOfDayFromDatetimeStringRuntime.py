import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class HourOfDayFromDatetimeStringRuntime(BaseEstimator, TransformerMixin):
    """ Extract hour from timestamp of form '2019-03-25 19:00:00'."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def extract_day(ts):
            # This is about 3x slower
            # return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').hour
            return int(ts[11:13])

        return np.array(list(map(extract_day, X.ravel()))).reshape(-1, 1)
