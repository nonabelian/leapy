import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class SelectorRuntime(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
