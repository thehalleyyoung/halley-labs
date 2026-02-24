"""Setup script for divelicit."""

from setuptools import setup, find_packages

setup(
    name="divelicit",
    version="0.1.0",
    description="Optimal transport-based diverse selection for LLM responses",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="DivFlow Team",
    license="Research Use Only",
    python_requires=">=3.9",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=[
        "numpy>=1.22",
    ],
    extras_require={
        "full": [
            "scipy>=1.9",
            "scikit-learn>=1.1",
            "openai>=1.0",
            "anthropic>=0.20",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "divelicit=src.api:elicit_diverse",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
