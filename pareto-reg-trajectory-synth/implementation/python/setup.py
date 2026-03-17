"""Setup script for regsynth-py."""
from setuptools import setup, find_packages
import os

def read_long_description():
    here = os.path.abspath(os.path.dirname(__file__))
    readme = os.path.join(here, "README.md")
    if os.path.exists(readme):
        with open(readme, encoding="utf-8") as f:
            return f.read()
    return "RegSynth: Regulatory compliance synthesis toolkit"

setup(
    name="regsynth-py",
    version="0.1.0",
    description="Python frontend for RegSynth regulatory compliance synthesis",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="RegSynth Team",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "regsynth=regsynth_py.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering",
    ],
)
