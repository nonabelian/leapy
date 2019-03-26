from timeit import timeit
from string import ascii_lowercase
from random import choice

import leapy

import numpy as np
import dask.array as da
from sklearn.pipeline import Pipeline
from dask_ml.linear_model import LogisticRegression

from leapy.dask.transformers import OneHotEncoder
from leapy.dask.pipeline import FeaturePipeline


if __name__ == '__main__':
    threshold = 10 * 1E-6  # 10 microseconds
    categories = [''.join(choice(ascii_lowercase) for _ in range(5))
                  for i in range(2000)]

    X_np = np.array(categories).reshape(-1, 1)
    X = da.from_array(X_np, chunks=(100, X_np.shape[1]))

    y_np = np.random.choice([0, 1], size=X_np.shape[0])
    y = da.from_array(y_np, chunks=(100,))

    feature_pipe = FeaturePipeline([('ohe', OneHotEncoder(sparse=False), [0])])

    pipe = Pipeline([
        ('feature_pipe', feature_pipe),
        ('clf', LogisticRegression())
    ])

    pipe.fit(X, y)

    print("Speed Test")
    print("=" * 30)

    x = choice(X).reshape(1, -1)
    x_np = x.compute()

    speed = timeit("pipe.predict(x).compute()",
                   globals=locals(),
                   number=100) / 100.
    print("Pipeline speed: {0:0.2f} microseconds".format(speed*1E6))

    pipe_runtime = Pipeline.export(pipe)

    speed_runtime = timeit("pipe_runtime.predict(x_np)",
                           globals=locals(),
                           number=1000) / 1000.
    print("Runtime pipeline speed: {0:0.2f} microseconds"
          .format(speed_runtime*1E6))
