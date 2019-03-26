import numpy as np


class FeaturePipelineRuntime(object):

    def __init__(self, transformer_lst, drop=True):
        self.named_steps, self.named_steps_cols = \
                self._create_steps_dict(transformer_lst)
        self.named_schema = dict()
        self.drop = drop

        self.input_shape = None

    def _create_steps_dict(self, transformer_lst):
        named_steps = dict()
        named_steps_cols = dict()
        for (name, transformer, cols) in transformer_lst:
            named_steps.update({name: transformer})
            named_steps_cols.update({name: cols})

        return named_steps, named_steps_cols

    def fit(self, X, y=None, **fit_kwargs):
        self.input_shape = X.shape

        X_out = X.copy()
        for name, transformer in self.named_steps.items():
            cols = self.named_steps_cols[name]
            X_col = X_out[:, cols]
            X_tf = transformer.fit_transform(X_col)

            input_cols = len(cols)
            output_cols = X_tf.shape[1]

            self.named_schema.update({name: [input_cols, output_cols]})
            X_out = np.concatenate([X_out, X_tf], axis=1)

        self.feature_size = 0
        for name, (input_cols, output_cols) in self.named_schema.items():
            self.feature_size += output_cols

    def transform(self, X):
        if self.drop:
            X_out = np.ndarray(shape=(X.shape[0], self.feature_size))
            start_col_idx = 0
        else:
            X_out = np.ndarray(shape=(X.shape[0],
                                      X.shape[1] + self.feature_size))
            X_out[:, :X.shape[1]] = X
            start_col_idx = X.shape[1]

        for name, transformer in self.named_steps.items():
            cols = self.named_steps_cols[name]
            output_cols = self.named_schema[name][1]
            X_tf = transformer.transform(X[:, cols])
            X_out[:, start_col_idx:start_col_idx + output_cols] = X_tf

            start_col_idx += output_cols

        return X_out

    def fit_transform(self, X,  y=None, **fit_kwargs):
        self.fit(X, y, **fit_kwargs)
        return self.transform(X)
