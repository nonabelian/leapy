import os
import sys
import argparse
from tempfile import mkdtemp
from shutil import rmtree
from shutil import copytree

import docker


APP_DIRECTORY = os.path.relpath(
    os.path.join(os.path.dirname(__file__), 'app'))

DOCKERFILE = """
FROM ubuntu:16.04
RUN apt-get update --fix-missing && apt-get install -y wget bzip2
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
COPY app/app.py /app/app.py
COPY app/config.py /app/config.py
COPY ./* /app/
RUN /opt/conda/bin/conda env update -f /app/conda.yaml
EXPOSE 8080
WORKDIR /app
CMD ["/opt/conda/bin/python", "app.py"]
"""


def serve(args):
    tag = 'leapy/' + args.tag
    name = tag.replace('/', '_')
    repo = os.path.dirname(os.path.abspath(args.repo))
    model_code = os.path.relpath(args.codedir)

    build_dir = mkdtemp()
    project_dir = os.path.join(build_dir, 'project')

    copytree(model_code, project_dir)

    print("Creating Dockerfile")
    dockerfile_out = os.path.join(build_dir, 'project', 'Dockerfile')
    if os.path.exists(dockerfile_out):
        rmtree(build_dir)
        raise ValueError("Dockerfile already exists! Remove first.")
    with open(dockerfile_out, 'w') as f:
        f.write(DOCKERFILE)

    if os.path.exists(os.path.join(model_code, 'app.py')) \
       or os.path.exists(os.path.join(model_code, 'config.py')):
        rmtree(build_dir)
        raise ValueError("app.py or config.py found! Cannot proceed.")
    copytree(APP_DIRECTORY, os.path.join(project_dir, 'app'))

    # Client
    client = docker.from_env()

    # Build
    print(f"Building container {tag} ...")
    for step in client.api.build(path=project_dir,
                                 rm=True,
                                 tag=tag,
                                 nocache=True
                                ):
        print(step)

    print(f"Cleaning up working directory: {build_dir}")
    rmtree(build_dir)

    if args.run == 'on':
        image = client.images.get(name=tag)
        # Run
        print(f"Starting Container {tag}...")
        c_params = {'detach': True,
                    'name': name,
                    'ports': {'8080/tcp': ('127.0.0.1', 8080)},
                    'volumes': {repo: {'bind': '/data',
                                       'mode': 'ro'}}
                   }
        client.containers.run(image, **c_params)
        print(f"Container '{tag}' running with name {name}")
    print("DONE")


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
    parser_serve.add_argument('--run', type=str, default='on')
    parser_serve.set_defaults(func=serve)

    parser_terminate = subparsers.add_parser('terminate')
    parser_terminate.add_argument('--name')
    parser_terminate.set_defaults(func=terminate)

    args = parser.parse_args(argv[1:])
    return args.func(args)


if __name__ == '__main__':

    sys.exit(main())
