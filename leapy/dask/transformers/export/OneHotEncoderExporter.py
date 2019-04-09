import numpy as np

from ....runtime.transformers import OneHotEncoderRuntime


class OneHotEncoderExporter():
    @staticmethod
    def to_runtime(self):
        categories = self.categories_
        ohe_runtime = OneHotEncoderRuntime(categories=categories,
                                           sparse=self.sparse)

        data = np.array([cat[0] for cat in categories]).reshape(1, -1)
        ohe_runtime.fit(data)

        ohe_runtime.num_features = \
                len(np.concatenate(ohe_runtime.categories_))

        return ohe_runtime
