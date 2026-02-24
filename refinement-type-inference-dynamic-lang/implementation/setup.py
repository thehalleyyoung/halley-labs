from setuptools import setup, find_packages
import os

setup(
    name="guardharvest",
    version="0.1.0",
    description="Find null/div-by-zero/bounds bugs in Python — zero annotations",
    long_description=open("../README.md").read() if os.path.exists("../README.md") else "",
    long_description_content_type="text/markdown",
    author="Guard Harvest Authors",
    license="MIT",
    packages=["guardharvest"],
    package_dir={"guardharvest": "src"},
    python_requires=">=3.9",
    install_requires=[],
    extras_require={
        "smt": ["z3-solver>=4.12"],
        "dev": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "guard-harvest=guardharvest.cli.main:main",
            "tensorguard=guardharvest.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Quality Assurance",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
