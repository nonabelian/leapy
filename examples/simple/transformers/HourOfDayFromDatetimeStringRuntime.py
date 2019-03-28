import numpy as np


class HourOfDayFromDatetimeStringRuntime(object):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def extract_day(ts):
            # This is about 3x slower
            # return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').hour
            return int(ts[11:13])

        return np.array(list(map(extract_day, X.ravel()))).reshape(-1, 1)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)
