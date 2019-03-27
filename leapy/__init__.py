from types import MethodType

from sklearn.pipeline import Pipeline
from leapy.dask.transformers import OneHotEncoder
from dask_ml.linear_model import LogisticRegression
from .dask.transformers.export import OneHotEncoderExporter
from .dask.transformers.export import TrivialExporter

class PipelineExporter():
    @staticmethod
    def to_runtime(self):
        pipe_runtime = Pipeline([(name, step.to_runtime()) for name, step in
                                 self.named_steps.items()])

        return pipe_runtime

setattr(Pipeline, 'to_runtime', PipelineExporter.to_runtime)
setattr(OneHotEncoder, 'to_runtime', OneHotEncoderExporter.to_runtime)
setattr(LogisticRegression, 'to_runtime', TrivialExporter.to_runtime)
# setattr(SGDClassifier, 'to_runtime', TrivialExporter())
# setattr(Incremental, 'to_runtime', TrivialExporter())
