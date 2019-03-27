import numpy as np
import dask.array as da

from leapy.runtime.pipeline import FeaturePipelineRuntime


class FeaturePipeline(object):

    def __init__(self, transformer_lst, drop=True):
        self.named_steps, self.named_steps_cols = \
                self._create_steps_dict(transformer_lst)
        self.named_schema = dict()
        self.drop = drop

        self.feature_size = None
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
            X_out = da.concatenate([X_out, X_tf], axis=1)
            X_out = X_out.rechunk({1: X_out.shape[1]})

        self.feature_size = 0
        for name, (input_cols, output_cols) in self.named_schema.items():
            self.feature_size += output_cols

    def transform(self, X):
        if self.drop:
            X_out = None
        else:
            X_out = X.copy()

        for name, transformer in self.named_steps.items():
            cols = self.named_steps_cols[name]
            X_col = X[:, cols]
            X_tf = transformer.transform(X_col)

            if X_out is None:
                X_out = X_tf
            else:
                X_out = da.concatenate([X_out, X_tf], axis=1)

            X_out = X_out.rechunk({1: X_out.shape[1]})

        return X_out.astype(np.float32)

    def fit_transform(self, X,  y=None, **fit_kwargs):
        self.fit(X, y, **fit_kwargs)
        return self.transform(X)

    def to_runtime(self):
        pipe_runtime = FeaturePipelineRuntime(
            [(name, step.to_runtime(), self.named_steps_cols[name])
             for name, step in self.named_steps.items()])
        pipe_runtime.named_schema = self.named_schema
        pipe_runtime.drop = self.drop
        pipe_runtime.feature_size = self.feature_size
        pipe_runtime.input_shape = self.input_shape

        return pipe_runtime
