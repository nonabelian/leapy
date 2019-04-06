import numpy as np

import leapy
from leapy.runtime.transformers import OneHotEncoderRuntime


class OneHotEncoderExporter():
    @staticmethod
    def to_runtime(self):
        cats = self.categories_
        ohe_runtime = OneHotEncoderRuntime(categories=cats,
                                           sparse=False)

        data = np.array([cat[0] for cat in cats]).reshape(1, -1)
        ohe_runtime.fit(data)

        ohe_runtime.num_features = \
                len(np.concatenate(ohe_runtime.categories_))

        return ohe_runtime
