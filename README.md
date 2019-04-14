![fast](images/leapy.gif)

**Just give me the dataz!**

Welcome!  Leapy is a library for real-time, sub-millisecond inference;
it provides customizable machine learning pipeline export for fast
model serving. 
These pipelines are targeted for using Dask's scalable machine learning,
which follows the Scikit-Learn API. However, you can use this framework
directly with Scikit-Learn as well.

```python
pipe = Pipeline([
        ('fp', FeaturePipeline([('ohe',
                                 OneHotEncoder(sparse=False),
                                 [0, 1])])),
        ('clf', LogisticRegression())
])

pipe.fit(X, y)

pipeline_runtime = pipe.to_runtime()               # ⚡⚡⚡ 
init('./model_repo', pipeline_runtime, df.head())  # Ready to deploy
```

And serve this super fast pipeline:
```
$ leap serve --repo ./model_repo
$ curl localhost:8080:/health
{
  "status": "healthy"
}
```
(See below for benchmarks and a more detailed usage example.)

### Benefits

Dask is a Python distributed computing environment in Python with a
burgeoning machine learning component, compatible with Scikit-Learn's API.
Using Leapy's framework, we can serve these pipelines in real-time!

This means:

* No JVM: No reliance on JVM from using Spark.
* Python: All Python development and custom transformers -- no Scala & Java
          needed!
* Scale: Scikit-Learn logic and pipelines scaled to clusters.
* Fast: You're in control of how fast your transformers are.
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

Start with a dataset in dask arrays, `X`, `y`, and dataframe `ddf`:
pipeline:
```python
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd

X_np = np.array([[1, 'a'], [2, 'b']], dtype=np.object)
df_pd = pd.DataFrame(X_np, columns=['test_int', 'test_str'])
y_np = np.array([0, 1])

X = da.from_array(X_np, chunks=X_np.shape)
y = da.from_array(y_np, chunks=y_np.shape)
ddf = dd.from_pandas(df_pd, npartitions=1)
```

Create our pipeline:

```python
from sklearn.pipeline import Pipeline
from dask_ml.linear_model import LogisticRegression
from leapy.dask.transformers import OneHotEncoder
from leapy.dask.pipeline import FeaturePipeline
from leapy.serve import init

pipe = Pipeline([
        ('fp', FeaturePipeline([
            # One-Hot-Encode 'test_str' feature, drop 'test_int'
            ('ohe', OneHotEncoder(sparse=False), [1])])),
        ('clf', LogisticRegression())
])

pipe.fit(X, y)
```

Then we export to a runtime pipeline and get ready for model serving:

```python
pipe_runtime = pipe.to_runtime()
init('./model_repo', pipe_runtime, ddf.head())
```

Finally we serve the model:
```
$ leap serve --repo ./model_repo
$ curl localhost:8080/predict \
    -X POST \
    -H "Content-Type: application/json" \
    --data '{"test_int": 1, "test_str": "b"}'
{
  "prediction": 1.0
}
```


For more on model serving see [leapy/serve/README.md](leapy/serve/README.md).

## Acknowledgments

Leapy is inspired by [MLeap](https://github.com/combust/mleap).
