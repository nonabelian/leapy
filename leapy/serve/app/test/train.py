import sys
sys.path.append('../../../../')
import pickle
import json

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from leapy.runtime.pipeline import FeaturePipelineRuntime
from leapy.runtime.transformers import SelectorRuntime
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':

    X = np.array([[1, 'a'], [2, 'b']], dtype=np.object)
    df = pd.DataFrame(X, columns=['test_int', 'test_str'])
    y = np.array([0, 1])

    pipeline_runtime = Pipeline([
        ('fp', FeaturePipelineRuntime([
            ('test_int',
             SelectorRuntime(),
             [df.columns.get_loc('test_int')])
        ])),
        ('clf', LogisticRegression())
    ])
    pipeline_runtime.fit(X, y)

    with open('model_repo/pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline_runtime, f)

    data_df = df.iloc[0:1, :].to_json(orient='records')
    data = json.loads(data_df)[0]
    y_pred = pipeline_runtime.predict(X[0:1,:])
    test_point = {'data': data,
                  'target': y_pred.tolist()}
    with open('model_repo/test_point.json', 'w') as f:
        json.dump(test_point, f)
