import os
import sys
import logging


class Config:
    def __init__(self):
        self.ENV = 'production'
        self.DEBUG = False
        self.TESTING = False
        self.LOGGING_LEVEL = logging.WARNING
        self.FORMAT_STRING = '%(asctime)s {}: '.format('app')\
                             + '%(name)-12s %(levelname)-8s %(message)s'
        self.LOGGING_CONFIG = {
            'version': 1,
            'formatters': {'f': {'format': self.FORMAT_STRING} },
            'handlers': {'h': {'class': 'logging.StreamHandler',
                               'formatter': 'f',
                               'level': self.LOGGING_LEVEL}
                        },
            'root': {'handlers': ['h'],
                     'level': self.LOGGING_LEVEL
                    }
        }

        self.PIPELINE_FILE = os.path.join('/data', 'pipeline.pkl')
        self.TEST_POINT = os.path.join('/data', 'test_point.json')

        from schema import MODEL_SCHEMA
        self.MODEL_SCHEMA = MODEL_SCHEMA


class ProductionConfig(Config):
    def __init__(self):
        super().__init__()


class DevelopmentConfig(Config):
    def __init__(self):
        super().__init__()
        self.ENV = 'development'
        self.DEBUG = True
        self.TESTING = False
        self.LOGGING_LEVEL = logging.DEBUG
        self.LOGGING_CONFIG['handlers']['h']['level'] = self.LOGGING_LEVEL
        self.LOGGING_CONFIG['root']['level'] = self.LOGGING_LEVEL

        self.PIPELINE_FILE = os.path.join('test', 'model_repo', 'pipeline.pkl')
        self.TEST_POINT = os.path.join('test', 'model_repo', 'test_point.json')

        from test.schema import MODEL_SCHEMA as TEST_MODEL_SCHEMA
        self.MODEL_SCHEMA = TEST_MODEL_SCHEMA

        sys.path.append('../../../')


if 'ENV' in os.environ.keys():
    if os.environ['ENV'] == 'testing':
        EnvironmentConfig = DevelopmentConfig()
else:
    EnvironmentConfig = ProductionConfig()
