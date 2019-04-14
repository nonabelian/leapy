import os
import json
import pickle

def init(repo, pipeline_runtime, df):

    # Save pipeline model
    pipeline_file = os.path.join(repo, 'pipeline.pkl')

    with open(pipeline_file, 'wb') as f:
        pickle.dump(pipeline_runtime, f)

    # Save test prediction point and result to JSON
    x = df.iloc[0:1, :].values
    data_df = df.iloc[0:1, :].to_json(orient='records')
    data = json.loads(data_df)[0]
    y_pred = pipeline_runtime.predict(x)
    test_point = {'data': data,
                  'target': y_pred.tolist()}
    test_point_file = os.path.join(repo, 'test_point.json')
    with open(test_point_file, 'w') as f:
        json.dump(test_point, f)
