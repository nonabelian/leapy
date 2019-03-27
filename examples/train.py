import sys
sys.path.append('../')
import pickle
from string import ascii_lowercase
from random import choice
from timeit import timeit
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import dask.array as da
from dask_ml.linear_model import LogisticRegression

# import leapy
from leapy.dask.pipeline import FeaturePipeline
from leapy.dask.transformers import OneHotEncoder
from HourOfDayFromDatetime import HourOfDayFromDatetime


def print_speed_comparison(X, pipe, pipe_runtime):
    print("Speed Test")
    print("=" * 30)

    x = choice(X).reshape(1, -1)
    x_np = x.compute()
    x = da.from_array(x_np, chunks=(1, x_np.shape[1]))

    speed = timeit("pipe.predict(x).compute()",
                   globals=locals(),
                   number=100) / 100.
    print("Pipeline speed: {0:0.2f} microseconds".format(speed*1E6))

    speed_runtime = timeit("pipe_runtime.predict(x_np)",
                           globals=locals(),
                           number=1000) / 1000.
    print("Runtime pipeline speed: {0:0.2f} microseconds"
          .format(speed_runtime*1E6))


if __name__ == '__main__':
    # Lets make some fake data
    num_pts = 2000
    start_date = datetime(2019, 1, 1, 2)
    end_date = start_date+ timedelta(hours=num_pts-1)
    df = pd.DataFrame(pd.date_range(start=start_date,
                                    end=end_date,
                                    freq='H'),
                      columns=['dt'])
    categories = [''.join(choice(ascii_lowercase)
                  for _ in range(5))
                  for i in range(num_pts)]

    df['cat_1'] = np.array(categories)
    df['cat_2'] = np.array(categories)
    df['label'] = np.random.choice([0, 1], size=df.shape[0])

    features = ['dt', 'cat_1', 'cat_2']
    X = da.from_array(df[features].values, chunks=df[features].shape)
    y = da.from_array(df['label'].values, chunks=(df.shape[0],))

    pipe = Pipeline([
        ('feature_pipe',FeaturePipeline([
            ('ohe_0', OneHotEncoder(sparse=False), [1]),
            ('ohe_1', OneHotEncoder(sparse=False), [2]),
            # ('ohe_2', OneHotEncoder(sparse=False), [1]),
            # ('ohe_3', OneHotEncoder(sparse=False), [2]),
            # ('ohe_4', OneHotEncoder(sparse=False), [1]),
            # ('ohe_5', OneHotEncoder(sparse=False), [1]),
            # ('ohe_6', OneHotEncoder(sparse=False), [1]),
            # ('ohe_7', OneHotEncoder(sparse=False), [1]),
            # ('ohe_8', OneHotEncoder(sparse=False), [1]),
            # ('ohe_9', OneHotEncoder(sparse=False), [1]),
            ('hod_0', HourOfDayFromDatetime(), [0]),
            ('hod_1', HourOfDayFromDatetime(), [0]),
            ('hod_2', HourOfDayFromDatetime(), [0]),
            ('hod_3', HourOfDayFromDatetime(), [0]),
            ('hod_4', HourOfDayFromDatetime(), [0]),
            ('hod_5', HourOfDayFromDatetime(), [0]),
            ('hod_6', HourOfDayFromDatetime(), [0]),
            ('hod_7', HourOfDayFromDatetime(), [0]),
            ('hod_8', HourOfDayFromDatetime(), [0]),
            ('hod_9', HourOfDayFromDatetime(), [0])])
        ),
        ('clf', LogisticRegression())
    ])

    pipe.fit(X, y)
    pipe_runtime = pipe.to_runtime()

    print_speed_comparison(X, pipe, pipe_runtime)

    # Save for model serving
    with open('pipe_runtime.pkl', 'wb') as f:
        pickle.dump(pipe_runtime, f)
