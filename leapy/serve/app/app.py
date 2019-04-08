import sys
import pickle
import argparse
from logging.config import dictConfig

from flask import Flask
from flask import jsonify
from flask import request
import numpy as np

from schema import model_schema
from config import EnvironmentConfig


dictConfig(EnvironmentConfig.LOGGING_CONFIG)


with open(EnvironmentConfig.PIPELINE_FILE, 'rb') as f:
    PIPELINE = pickle.load(f)


app = Flask(__name__)
app.config.from_object('config.EnvironmentConfig')


@app.route('/predict', METHODS=['POST'])
def predict():
    data = model_schema.load(request.get_json())
    pt = np.array([[v for k, v in data.items()]])
    y_pred = PIPELINE.predict(data)

    return jsonify({'prediction': y_pred})


@app.route('/health')
def health():

    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    argv = sys.argv

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)

    args = parser.parse_args(argv[1:])

    app.run('0.0.0.0', port=args.port, threaded=True)

