"""Builds a .whl for the pyfite package.
"""
import os
from pathlib import Path

import setuptools

os.chdir(Path(__file__).parent)

with open('README.md', 'r') as fh:
    long_description = fh.read()
with open('requirements.txt', 'r') as fh:
    requirements = [req for req in fh.readlines() if req]

setuptools.setup(
    name="pyfite",
    version="0.0.2",
    author="Ryan Hite",
    author_email="rhite@ara.com",
    description="Basic module containing FITE helper classes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['pyfite'],
    exclude_package_data={'pyfite':['_tests/*','*.md','*.txt']},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
