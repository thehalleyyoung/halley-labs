"""BioProver packaging configuration (legacy setup.py)."""

from setuptools import setup, find_packages

setup(
    name="bioprover",
    version="0.1.0",
    author="BioProver Team",
    description=(
        "CEGAR-based verification and parameter repair "
        "for synthetic biology circuits"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bioprover/bioprover",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "sympy>=1.9",
        "networkx>=2.6",
        "z3-solver>=4.8",
    ],
    extras_require={
        "viz": ["matplotlib>=3.4"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "mypy>=0.950",
            "ruff>=0.1",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bioprover=bioprover.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Typing :: Typed",
    ],
    keywords="synthetic-biology verification CEGAR temporal-logic",
)
