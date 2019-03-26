from sklearn.pipeline import Pipeline
from leapy.dask.transformers import OneHotEncoder
from dask_ml.linear_model import LogisticRegression
from .dask.transformers.export import OneHotEncoderExporter
from .dask.transformers.export import TrivialExporter

def export(pipeline):
    pipe_runtime = Pipeline([(name, step.export(step)) for name, step in
                             pipeline.named_steps.items()])

    return pipe_runtime

setattr(Pipeline, 'export', export)

setattr(OneHotEncoder, 'export', OneHotEncoderExporter())
setattr(LogisticRegression, 'export', TrivialExporter())
# setattr(SGDClassifier, 'export', TrivialExporter())
# setattr(Incremental, 'export', TrivialExporter())
