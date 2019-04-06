![fast](images/leapy.gif)

**Just give me the dataz!**

Welcome!  Leapy is a library for real-time, sub-millisecond inference;
it provides customizable machine learning pipeline export for fast
model serving. 
These pipelines are targeted for using Dask's scalable machine learning,
which follows the Scikit-Learn API. However, you can use this framework
directly with Scikit-Learn as well.

Leapy is inspired by [MLeap](https://github.com/combust/mleap).

### Benefits

Dask is a Python distributed computing environment in Python with a
burgeoning machine learning component, compatible with Scikit-Learn's API.
Using Leapy's framework, we can serve these pipelines in real-time!

This means:

* No JVM: No reliance on JVM from using Spark.
* Python: All Python development and custom transformers -- no Scala & Java
          needed!
* Scale: Scikit-Learn logic and pipelines scaled to clusters.
* Fast: You're in control of how fast your transformers are!
* Simple: Easily build and deploy models with Docker.
* Reliable: Encourages a test-driven approach.
<!--* MLflow: Serve runtime models (as Scikit-Learn models) through `mlflow`.-->

### Examples

* [Simple](examples/simple)
    -- Super simple example of creating, testing, and using custom
    transformers
* [XGBoost](examples/)
    -- Advanced example of using XGBoost with Dask, saving, and serving the
    model.

### Benchmarks

A simple example of what we're going for -- computing a one-hot-encoding,
with ~200K labels, of a single data point (dask array `x_da` and numpy array
`x = x_da.compute()`):

![sample benchmark](images/sample_benchmark.png)

Where `ohe_dml` (from `dask_ml`) and `ohe` (from `leapy`) are essentially the
same; `ohe_sk` is from `scikit-learn` and `ohe_runtime` is from
`ohe.to_runtime()`. And, running `compute()` on Dask transforms above bumps
the time up to about 1 second.

With the time we save using `ohe_runtime`, we can squeeze in many more
transformations and an estimator to still come in under 1ms.

### Example Usage

Start with a dataset in dask arrays, `X` and `y`, and create a Scikit-Learn
pipeline:

```python
from dask_ml.linear_model import LogisticRegression
from leapy.dask.transformers import OneHotEncoder
from leapy.dask.pipeline import FeaturePipeline

pipe = Pipeline([
        ('fp', FeaturePipeline([('ohe',
                                 OneHotEncoder(sparse=False),
                                 [0, 1])])),
        ('clf', LogisticRegression())
])

pipe.fit(X, y)
```

Then we export to a runtime pipeline, and save:

```python
pipe_runtime = pipe.to_runtime()

with open('pipe_runtime.pkl', 'wb') as f:
    pickle.dump(pipe_runtime, f)
```

This model is ready to be served! [Docker](docs/DOCKER.md)
