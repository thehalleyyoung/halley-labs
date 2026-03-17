from setuptools import setup, find_packages

setup(
    name="collusion-proof",
    version="0.1.0",
    description="CollusionProof: Algorithmic Collusion Certification System",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="CollusionProof Team",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "pandas>=1.3",
        "matplotlib>=3.4",
        "seaborn>=0.11",
        "scikit-learn>=1.0",
        "click>=8.0",
        "pydantic>=2.0",
        "tqdm>=4.62",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov", "black", "ruff", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "collusion-proof=collusion_proof.cli.main:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
    ],
)
