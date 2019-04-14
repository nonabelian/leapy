import sys
import pickle
import json
import argparse
import logging
from logging.config import dictConfig

from flask import Flask
from flask import jsonify
from flask import request
import numpy as np

from config import EnvironmentConfig


dictConfig(EnvironmentConfig.LOGGING_CONFIG)

MODEL_SCHEMA = EnvironmentConfig.MODEL_SCHEMA


with open(EnvironmentConfig.PIPELINE_FILE, 'rb') as f:
    PIPELINE = pickle.load(f)

with open(EnvironmentConfig.TEST_POINT, 'r') as f:
    TEST_POINT = json.load(f)


app = Flask(__name__)
app.config.from_object('config.EnvironmentConfig')


@app.route('/predict', methods=['POST'])
def predict():
    data = MODEL_SCHEMA.load(request.get_json())
    pt = np.array([[v for v in data.values()]],
                  dtype=np.object)
    y_pred = float(PIPELINE.predict(pt)[0])

    return jsonify({'prediction': y_pred})


@app.route('/health')
def health():
    logger = logging.getLogger('/health')
    data_json = json.dumps(TEST_POINT['data'])
    data = MODEL_SCHEMA.loads(data_json)
    pt = np.array([[v for v in data.values()]],
                  dtype=np.object)
    y_pred = PIPELINE.predict(pt)
    y_exp = np.array(TEST_POINT['target'])
    if np.all(y_exp == y_pred):
        logger.info("Predictions match expected.")
        return jsonify({'status': 'healthy'})
    else:
        logger.info("Predictions DO NOT match expected.")
        return jsonify({'status': 'unhealthy'})


if __name__ == '__main__':
    argv = sys.argv

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)

    args = parser.parse_args(argv[1:])

    app.run('0.0.0.0', port=args.port, threaded=True)

