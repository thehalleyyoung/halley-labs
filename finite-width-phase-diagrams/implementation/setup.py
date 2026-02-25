"""Setup script for phase-diagrams package."""

from setuptools import setup, find_packages

setup(
    name="phase-diagrams",
    version="0.2.0",
    description="Predict lazy-to-rich learning transitions in neural networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Finite-Width Phase Diagrams Team",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
    ],
    extras_require={
        "torch": ["torch>=1.12"],
        "plot": ["matplotlib>=3.5"],
        "all": ["torch>=1.12", "matplotlib>=3.5"],
    },
    entry_points={
        "console_scripts": [
            "phase-diagrams=cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
