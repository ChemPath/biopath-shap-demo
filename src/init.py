"""
Setup script for BioPath SHAP Demo package.

This module configures the BioPath demonstration package for natural compound
bioactivity prediction with explainable AI capabilities.
"""

from setuptools import setup, find_packages
import os
import sys

# Ensure Python 3.8+
if sys.version_info < (3, 8):
    raise RuntimeError("BioPath SHAP Demo requires Python 3.8 or higher")

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "BioPath SHAP Demo: Explainable Natural Compound Bioactivity Prediction"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Package metadata
PACKAGE_NAME = "biopath-shap-demo"
VERSION = "0.1.0"
DESCRIPTION = "Explainable AI for natural compound bioactivity prediction"
AUTHOR = "OmniPath Technologies"
AUTHOR_EMAIL = "biopath-demo@omnipath.ai"
URL = "https://github.com/omnipath/biopath-shap-demo"
LICENSE = "MIT"

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Healthcare Industry",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Operating System :: OS Independent",
]

# Keywords for discoverability
KEYWORDS = [
    "bioinformatics",
    "cheminformatics", 
    "machine-learning",
    "explainable-ai",
    "shap",
    "natural-products",
    "drug-discovery",
    "traditional-medicine",
    "molecular-descriptors",
    "bioactivity-prediction"
]

# Core dependencies (essential for basic functionality)
CORE_DEPENDENCIES = [
    "numpy>=1.21.0",
    "pandas>=1.5.0",
    "scikit-learn>=1.2.0",
    "rdkit>=2022.9.1",
    "shap>=0.42.0",
    "matplotlib>=3.6.0",
    "seaborn>=0.12.0",
]

# Optional dependencies for extended functionality
EXTRAS_REQUIRE = {
    "visualization": [
        "plotly>=5.11.0",
        "bokeh>=2.4.0",
    ],
    "deep-learning": [
        "tensorflow>=2.10.0",
        "torch>=1.12.0",
    ],
    "cloud": [
        "google-cloud-storage>=2.5.0",
        "google-cloud-bigquery>=3.3.0",
    ],
    "optimization": [
        "optuna>=3.0.0",
        "hyperopt>=0.2.7",
        "numba>=0.56.0",
    ],
    "development": [
        "pytest>=7.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=0.991",
        "jupyter>=1.0.0",
        "sphinx>=5.0.0",
    ],
    "all": [
        "plotly>=5.11.0",
        "bokeh>=2.4.0",
        "tensorflow>=2.10.0", 
        "torch>=1.12.0",
        "google-cloud-storage>=2.5.0",
        "google-cloud-bigquery>=3.3.0",
        "optuna>=3.0.0",
        "hyperopt>=0.2.7",
        "numba>=0.56.0",
        "pytest>=7.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=0.991",
        "jupyter>=1.0.0",
        "sphinx>=5.0.0",
    ]
}

# Entry points for command-line tools
ENTRY_POINTS = {
    "console_scripts": [
        "biopath-predict=src.models.bioactivity_predictor:main",
        "biopath-explain=src.explainers.bio_shap_explainer:main",
        "biopath-features=src.data_preprocessing.molecular_features:main",
    ],
}

# Package data to include
PACKAGE_DATA = {
    "biopath_shap_demo": [
        "data/sample_compounds.csv",
        "docs/*.md",
        "examples/*.py",
        "examples/*.ipynb",
    ],
}

# Data files to include in distribution
DATA_FILES = [
    ("biopath_shap_demo/data", ["data/sample_compounds.csv"]),
    ("biopath_shap_demo/examples", ["examples/demo_script.py"]),
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=CORE_DEPENDENCIES,
    extras_require=EXTRAS_REQUIRE,
    
    # Package metadata
    classifiers=CLASSIFIERS,
    keywords=", ".join(KEYWORDS),
    
    # Entry points and data
    entry_points=ENTRY_POINTS,
    package_data=PACKAGE_DATA,
    data_files=DATA_FILES,
    include_package_data=True,
    zip_safe=False,
    
    # Project URLs for PyPI
    project_urls={
        "Homepage": URL,
        "Documentation": f"{URL}/docs",
        "Bug Reports": f"{URL}/issues",
        "Source": URL,
        "Download": f"{URL}/archive/v{VERSION}.tar.gz",
        "Funding": "https://omnipath.ai/funding",
        "Commercial Support": "https://omnipath.ai/support",
    },
    
    # Additional metadata
    platforms=["any"],
    test_suite="tests",
    tests_require=[
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.8.0",
    ],
    
    # Setuptools specific options
    options={
        "bdist_wheel": {
            "universal": False,  # Not compatible with Python 2
        },
        "build_py": {
            "exclude_packages": ["tests", "tests.*"],
        },
    },
)

# Post-installation message
def print_post_install_message():
    """Print helpful information after installation."""
    print("\n" + "="*60)
    print("🧬 BioPath SHAP Demo Installation Complete!")
    print("="*60)
    print("\nQuick Start:")
    print("1. Import the package:")
    print("   >>> from biopath_shap_demo import MolecularFeatureCalculator")
    print("\n2. Run the demo:")
    print("   >>> python examples/demo_script.py")
    print("\n3. Explore notebooks:")
    print("   >>> jupyter notebook notebooks/")
    print("\n4. Command-line tools:")
    print("   >>> biopath-predict --help")
    print("   >>> biopath-explain --help")
    print("   >>> biopath-features --help")
    print("\nDocumentation: https://github.com/omnipath/biopath-shap-demo/docs")
    print("Support: biopath-demo@omnipath.ai")
    print("="*60)

# Custom install command to show post-install message
try:
    from setuptools.command.install import install
    
    class CustomInstall(install):
        def run(self):
            install.run(self)
            print_post_install_message()
    
    # Override install command
    setup.cmdclass = {"install": CustomInstall}
    
except ImportError:
    # Fallback for older setuptools versions
    pass

# Validation checks
def validate_setup():
    """Validate setup configuration."""
    issues = []
    
    # Check for required files
    required_files = ["README.md", "requirements.txt", "src/__init__.py"]
    for file_path in required_files:
        if not os.path.exists(file_path):
            issues.append(f"Missing required file: {file_path}")
    
    # Check Python version compatibility
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    
    # Check for RDKit availability (critical dependency)
    try:
        import rdkit
    except ImportError:
        issues.append("RDKit not available - required for molecular computing")
    
    if issues:
        print("⚠️  Setup Validation Issues:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease resolve these issues before installation.")
        return False
    
    print("✅ Setup validation passed")
    return True

# Run validation if executed directly
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ["--help", "-h"]:
        validate_setup()
