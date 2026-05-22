"""
pip install -e bindings/python
"""
from setuptools import setup, find_packages

setup(
    name="dm",
    version="1.0.0",
    description="Python bindings for the DM data-mining framework",
    packages=find_packages(),
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
