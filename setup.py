# -*- coding: utf-8 -*-

import os
from setuptools import setup

long_description = ("Calculates equilibrium band diagrams for planar "
                    "multilayer semiconductor structures. To learn "
                    "more go to "
                    "http://pythonhosted.org/eq_band_diagram/")

descrip = ("Calculates equilibrium band diagrams for planar multilayer "
           "semiconductor structures.")

setup(
    name = "eq_band_diagram",
    version = "0.1.0",
    author = "Steven Byrnes",
    author_email = "steven.byrnes@gmail.com",
    description = descrip,
    license = "MIT",
    keywords = "semiconductor physics, poisson equation, boltzmann equation, finite differences",
    url = "http://pythonhosted.org/eq_band_diagram/",
    py_modules=['eq_band_diagram'],
    long_description=long_description,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        ]
)
