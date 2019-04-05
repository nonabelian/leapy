import numpy as np
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.externals import six


class FeaturePipelineRuntime(_BaseComposition):

    def __init__(self, steps, drop=True):
        self.steps = steps
        self.drop = drop
        self.named_steps, self.named_steps_cols = \
                self._create_steps_dict(steps)
        self.named_schema = dict()

        self.input_shape = None

    def _get_params(self, attr, deep=True):
        out = super().get_params(deep=deep)
        if not deep:
            return out
        estimators = getattr(self, attr)
        estimators = [(name, estimator) for name, estimator, _ in estimators]
        out.update(estimators)
        for name, estimator in estimators:
            if hasattr(estimator, 'get_params'):
                for key, value in six.iteritems(
                        estimator.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
        return out

    def get_params(self, deep=True):
        return self._get_params('steps', deep=deep)

    def _set_params(self, attr, **params):
        # Ensure strict ordering of parameter setting:
        # 1. All steps
        if attr in params:
            setattr(self, attr, params.pop(attr))
        # 2. Step replacement
        items = getattr(self, attr)
        names = []
        if items:
            names, _, _ = zip(*items)
        for name in list(six.iterkeys(params)):
            if '__' not in name and name in names:
                self._replace_estimator(attr, name, params.pop(name))
        # 3. Step parameters and other initialisation arguments
        super().set_params(**params)
        return self

    def set_params(self, **kwargs):
        self._set_params('steps', **kwargs)
        return self

    def _replace_estimator(self, attr, name, new_val):
        # assumes `name` is a valid estimator name
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _, cols) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val, cols)
                break
        setattr(self, attr, new_estimators)

    def _create_steps_dict(self, steps):
        named_steps = dict()
        named_steps_cols = dict()
        for (name, transformer, cols) in steps:
            named_steps.update({name: transformer})
            named_steps_cols.update({name: cols})

        return named_steps, named_steps_cols

    def fit(self, X, y=None, **fit_kwargs):
        self.input_shape = X.shape

        for name, transformer in self.named_steps.items():
            cols = self.named_steps_cols[name]
            X_col = X[:, cols]
            X_tf = transformer.fit_transform(X_col)

            input_cols = len(cols)
            output_cols = X_tf.shape[1]

            self.named_schema.update({name: [input_cols, output_cols]})

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
