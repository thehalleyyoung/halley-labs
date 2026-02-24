#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="litmus-inf",
    version="0.2.0",
    description="LITMUS∞: Cross-Architecture Memory Model Portability Checker",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Anonymous",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(),
    py_modules=["portcheck", "api", "ast_analyzer", "fence_cost", "nlgen",
                "cuda_checker", "rust_atomics", "ci_integration",
                "litmus_check", "benchmark_suite"],
    entry_points={
        "console_scripts": [
            "litmus-check=litmus_check:main",
            "litmus-inf=api:_main",
            "litmus-portcheck=portcheck:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Testing",
        "Topic :: System :: Hardware",
    ],
)
