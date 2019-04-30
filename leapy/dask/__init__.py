from dask_ml.linear_model import LogisticRegression
from .transformers import OneHotEncoder
from .transformers import Selector
from .transformers.export import OneHotEncoderExporter
from .transformers.export import TrivialExporter


setattr(LogisticRegression, 'to_runtime', TrivialExporter.to_runtime)
# setattr(SGDClassifier, 'to_runtime', TrivialExporter.to_runtime)
# setattr(Incremental, 'to_runtime', TrivialExporter.to_runtime)


setattr(OneHotEncoder, 'to_runtime', OneHotEncoderExporter.to_runtime)
setattr(Selector, 'to_runtime', TrivialExporter.to_runtime)
