#!/usr/bin/env python
from catkin_pkg.python_setup import generate_distutils_setup
from distutils.core import setup

d = generate_distutils_setup(
    packages=['recognisers'],
    package_dir={'': 'src'}
)
setup(**d)
