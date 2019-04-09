import numpy as np
from sklearn.preprocessing import OneHotEncoder


class OneHotEncoderRuntime(OneHotEncoder):

    def fit(self, X, y=None):
        super().fit(X, y=y)

        self.cat_idx_map = dict()
        counter = 0
        for c in np.concatenate(self.categories_):
            self.cat_idx_map[c] = counter
            counter += 1

        self.num_features = len(np.concatenate(self.categories_))

        return self

    def transform(self, X):
        X_tf = np.zeros(shape=(X.shape[0], self.num_features))

        for i, row in enumerate(X):
            for val in row:
                X_tf[i, self.cat_idx_map[val]] = 1

        return X_tf
