# Model serving

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
