from sklearn.pipeline import Pipeline


__version__ = '0.2.0'


class PipelineExporter():
    @staticmethod
    def to_runtime(self):
        pipe_runtime = Pipeline([(name, step.to_runtime()) for name, step in
                                 self.named_steps.items()])

        return pipe_runtime


setattr(Pipeline, 'to_runtime', PipelineExporter.to_runtime)
