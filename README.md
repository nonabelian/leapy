# leapy

~~~~~~~~~~~~~~~~~**Work in progress**~~~~~~~~~~~~~~~~~

Welcome!  Leapy is a library for real-time, sub-millisecond inference;
it provides machine learning pipeline export for fast model serving. 
These pipelines are based on Dask's scalable machine learning, which
follows the Scikit-Learn API.

Leapy is inspired by [MLeap](https://github.com/combust/mleap).

### Why?

Dask provides a Python distributed computing environment with a
burgeoning machine learning component (compatible with Scikit-Learn's API).
With leapy we can serve these models in real-time.

This means:

* No reliance on JVM from using Spark.
* All Python development and custom transformers -- no Scala & Java needed!
* Scale Scikit-Learn logic and pipelines to clusters.

<!--Pros vs `mleap`:-->
<!--* Custom transformers: Only need to write a Dask transformer, a runtime-->
  <!--transformer, and a transformer export function.  All in Python;-->
  <!--all with familiar numpy/scikit-learn APIs.-->
<!--* Custom transformer: You might want to write a Spark transformer, runtime-->
  <!--transformer, associated export function, and Python bindings.  Programming-->
  <!--in Java, Scala, and Python.-->
<!--* No JVM required.-->

<!--Cons vs `mleap`:-->
<!--* Many fewer ML algorithms in `dask_ml` right now (this can change!).-->
<!--* Less production usage, community support.-->

### Benchmarks



### Example Usage

Start with a dataset:

```python
from string import ascii_lowercase
from random import choice
import numpy as np
import dask.array as da

categories = [''.join(choice(ascii_lowercase) for _ in range(5))
              for i in range(2000)]

X_np = np.array(categories).reshape(-1, 1)
X = da.from_array(X_np, chunks=(100, X_np.shape[1]))

y_np = np.random.choice([0, 1], size=X_np.shape[0])
y = da.from_array(y_np, chunks=(100,))
```

Now we create our Dask ML pipeline:

```python
from sklearn.pipeline import Pipeline
from dask_ml.linear_model import LogisticRegression

from leapy.dask.transformers import OneHotEncoder
from leapy.dask.pipeline import FeaturePipeline

feature_pipe = FeaturePipeline([('ohe', OneHotEncoder(sparse=False), [0])])

pipe = Pipeline([
    ('feature_pipe', feature_pipe),
    ('clf', LogisticRegression())
])

pipe.fit(X, y)
```

And we export to a runtime pipeline, and save:

```python
import pickle

pipe.fit(X, y)
pipe_runtime = Pipeline.export(pipe)

with open('pipe_runtime.pkl', 'wb') as f:
    pickle.dump(pipe_runtime, f)
```

Finally we can serve the model (this uses our minimalistic model
serving code from [link]):

```
$ docker build -f Dockerfile -t leapy/example .
$ docker run -d -v /path/to/pipeline/:/opt/ -p 0.0.0.0:8080:8080 -t leapy/example
```

For example, we can get predictions like the following:

```
$ curl --header "Content-Type: application/json" \
       --request POST \
       --data '{"input_feature_1": ...}' \
       localhost:8080/api/predict
```

Or update the model (to the newest pipeline in the mounted directory):

```
$ curl localhost:8080/api/update
```
