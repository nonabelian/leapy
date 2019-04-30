import sys
sys.path.append('../../../../')
import pickle
import json

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from leapy.sklearn.pipeline import FeaturePipeline
from leapy.sklearn.transformers import Selector
from sklearn.linear_model import LogisticRegression
from leapy.serve import init


if __name__ == '__main__':

    X = np.array([[1, 'a'], [2, 'b']], dtype=np.object)
    df = pd.DataFrame(X, columns=['test_int', 'test_str'])
    y = np.array([0, 1])

    pipeline = Pipeline([
        ('fp', FeaturePipeline([
            ('test_int', Selector(), [df.columns.get_loc('test_int')])
        ])),
        ('clf', LogisticRegression())
    ])
    pipeline.fit(X, y)

    pipeline_runtime = pipeline.to_runtime()

    init('./model_repo', pipeline_runtime, df.head())
