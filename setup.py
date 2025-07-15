"""
Modern setup configuration for BioPath SHAP Demo
"""

from setuptools import setup, find_packages
import sys

# Python version check
if sys.version_info < (3.8):
    raise RuntimeError("BioPath SHAP Demo requires Python 3.8 or higher")

setup(
    name="biopath-shap-demo",
    version="2.0.0",
    description="Modern explainable AI for natural compound bioactivity prediction",
    author="OmniPath Technologies",
    author_email="biopath@omnipath.ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "rdkit>=2023.9.1",
        "shap>=0.42.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "xgboost>=1.7.0",
        "optuna>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
        ],
        "visualization": [
            "plotly>=5.17.0",
            "bokeh>=3.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

