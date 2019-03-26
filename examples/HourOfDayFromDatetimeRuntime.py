import numpy as np


class HourOfDayFromDatetimeRuntime(object):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def extract_day(ts):
            return ts.to_pydatetime().hour

        return np.array(list(map(extract_day, X.ravel()))).reshape(-1, 1)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)
