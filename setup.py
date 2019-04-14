from setuptools import setup
from setuptools import find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='leapy',
      version='0.0.3',
      description='Real-time inference pipelines',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/nonabelian/leapy',
      author='Dylan Albrecht',
      author_email='djalbrecht@email.wm.edu',
      license='new BSD',
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Programming Language :: Python',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 3.7'],
      packages=find_packages(),
      install_requires=['pandas>=0.24.2',
                        'dask>=1.1.4',
                        'dask-ml>=0.12.0',
                        'numpy>=1.16.2',
                        'scikit-learn>=0.20.3',
                        'marshmallow>=2.19.2'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest',
                     'numpy',
                     'dask'],
      zip_safe=False)
