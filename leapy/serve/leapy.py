import os
import sys
import argparse
from io import BytesIO

import docker


APP_DIRECTORY = os.path.join(os.path.abspath(__file__),
                             'app')

DOCKERFILE_FMT = """
FROM ubuntu:16.04
RUN apt-get update --fix-missing && apt-get install -y wget bzip2
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
COPY {APP_DIRECTORY} /app
COPY {MODEL_CODE_ABSOLUTE}/* /app/
RUN /opt/conda/bin/conda env update -f /app/conda.yaml
EXPOSE 8080
WORKDIR /app
CMD ["/opt/conda/bin/python", "app.py"]
"""


def serve(args):
    tag = 'leapy/' + args.tag
    name = tag.replace('/', '_')
    repo = args.repo
    model_code = os.path.abspath(args.modelcode)

    # Dockerfile
    file_params = {'APP_DIRECTORY': APP_DIRECTORY,
                   'MODEL_CODE': model_code}
    dockerfile = DOCKERFILE_FMT.format(**file_params)
    df = BytesIO(dockerfile.encode('utf-8'))
    print(dockerfile)
    return 1

    # Client
    client = docker.from_env()

    # Build
    print(f"building container {tag} ...")
    build = client.api.build(fileobj=df, rm=True, tag=tag)

    # Run
    print(f"Starting Container {tag}...")
    c_params = {'detach': True,
                'name': name,
                'port': {'8080/tcp': ('127.0.0.1', 8080)},
                'volumes': {repo: {'bind': '/data',
                                   'mode': 'ro'}}
               }
    client.containers.run(tag, **c_params)
    print(f"Container '{tag}' running with name {name}")


def terminate(args):
    name = args.name
    client = docker.from_env()
    container = client.containers.get(name)
    container.stop()
    print(f"Container {name} terminated")


def main(argv=None):

    argv = argv or sys.argv

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_serve = subparsers.add_parser('serve')
    parser_serve.add_argument('--repo', type=str, default='.')
    parser_serve.add_argument('--tag', type=str, default='model')
    parser_serve.add_argument('--codedir', type=str, default='.')
    parser_serve.set_defaults(func=serve)

    parser_terminate = subparsers.add_parser('terminate')
    parser_terminate.add_argument('--name')
    parser_terminate.set_defaults(func=terminate)

    args = parser.parse_args(argv[1:])
    return args.func(args)


if __name__ == '__main__':

    sys.exit(main())
