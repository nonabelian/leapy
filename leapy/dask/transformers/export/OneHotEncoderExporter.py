import numpy as np

from leapy.runtime.transformers import OneHotEncoderRuntime


class OneHotEncoderExporter():
    def __call__(self, ohe):
        cats = ohe.categories_
        ohe_runtime = OneHotEncoderRuntime(categories=cats,
                                           sparse=False)

        data = np.array([cat[0] for cat in cats]).reshape(1, -1)
        ohe_runtime.fit(data)

        return ohe_runtime
