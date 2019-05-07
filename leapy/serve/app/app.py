import sys
import pickle
import json
import argparse
import logging
from logging.config import dictConfig

from vibora import Vibora
from vibora import Request
from vibora.responses import JsonResponse
import numpy as np

from config import EnvironmentConfig


dictConfig(EnvironmentConfig.LOGGING_CONFIG)

MODEL_SCHEMA = EnvironmentConfig.MODEL_SCHEMA


with open(EnvironmentConfig.PIPELINE_FILE, 'rb') as f:
    PIPELINE = pickle.load(f)

with open(EnvironmentConfig.TEST_POINT, 'r') as f:
    TEST_POINT = json.load(f)


app = Vibora()


@app.route('/predict', methods=['POST'])
async def predict(request: Request):
    j_data = await request.json()
    data = MODEL_SCHEMA.load(j_data)
    pt = np.array([[v for v in data.values()]],
                  dtype=np.object)
    y_pred = float(PIPELINE.predict(pt)[0])

    return JsonResponse({'prediction': y_pred})


@app.route('/health')
async def health():
    logger = logging.getLogger('/health')
    data_json = json.dumps(TEST_POINT['data'])
    data = MODEL_SCHEMA.loads(data_json)
    pt = np.array([[v for v in data.values()]],
                  dtype=np.object)
    y_pred = PIPELINE.predict(pt)
    y_exp = np.array(TEST_POINT['target'])
    if np.all(y_exp == y_pred):
        logger.info("Predictions match expected.")
        return JsonResponse({'status': 'healthy'})
    else:
        logger.info("Predictions DO NOT match expected.")
        return JsonResponse({'status': 'unhealthy'})


if __name__ == '__main__':
    argv = sys.argv

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)

    args = parser.parse_args(argv[1:])

    app.run(host='0.0.0.0', debug=EnvironmentConfig.DEBUG,  port=args.port)
