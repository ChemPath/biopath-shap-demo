"""
BioPath SHAP Demo - Test Package

This package provides comprehensive test coverage for the BioPath SHAP Demo
system including unit tests, integration tests, and validation tests for
molecular feature calculation, bioactivity prediction, and SHAP explainability.

Test Structure:
- Unit tests for individual components and functions
- Integration tests for complete workflows
- Performance tests for batch processing
- Validation tests for scientific accuracy
- Mock tests for external dependencies

Key Test Areas:
- Molecular feature calculation accuracy and reliability
- Bioactivity prediction model training and validation
- SHAP explainer functionality and biological interpretations
- Visualization component rendering and accuracy
- Traditional knowledge integration and bias correction

Usage:
    pytest tests/                    # Run all tests
    pytest tests/test_molecular.py   # Run specific test module
    pytest tests/ -v                 # Verbose output
    pytest tests/ --cov=src/         # Coverage report
"""

__version__ = "2.0.0"
__author__ = "OmniPath Technologies"
__email__ = "biopath@omnipath.ai"

# Test configuration constants
TEST_DATA_DIR = "tests/data"
TEST_OUTPUT_DIR = "tests/output"
DEFAULT_TEST_TIMEOUT = 300  # seconds
MAX_TEST_COMPOUNDS = 100

# Test sample data
TEST_COMPOUNDS = {
    'quercetin': {
        'smiles': 'c1cc(ccc1c2cc(=O)c3c(cc(cc3o2)O)O)O',
        'expected_activity': 1,
        'compound_class': 'flavonoids'
    },
    'caffeine': {
        'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'expected_activity': 0,
        'compound_class': 'alkaloids'
    },
    'ethanol': {
        'smiles': 'CCO',
        'expected_activity': 0,
        'compound_class': 'inactive_controls'
    }
}

# Test validation thresholds
VALIDATION_THRESHOLDS = {
    'feature_calculation_accuracy': 0.95,
    'model_training_accuracy': 0.70,
    'shap_explanation_completeness': 0.90,
    'traditional_knowledge_representation': 0.30,
    'visualization_rendering_success': 0.95
}

# Mock data for testing
MOCK_FEATURE_NAMES = [
    'molecular_weight', 'logp', 'tpsa', 'hbd_count', 'hba_count',
    'phenol_groups', 'hydroxyl_groups', 'chiral_centers', 'qed_score',
    'bertz_complexity', 'aromatic_rings', 'rotatable_bonds'
]

MOCK_BIOACTIVITY_DATA = {
    'antioxidant': [1, 0, 1, 1, 0, 1, 0, 0],
    'anti_inflammatory': [1, 0, 1, 0, 0, 1, 0, 0],
    'antimicrobial': [0, 1, 0, 1, 1, 0, 1, 0]
}

# Test utility functions
def setup_test_environment():
    """Setup test environment with required directories and mock data."""
    import os
    import tempfile
    from pathlib import Path
    
    # Create test directories
    test_dirs = [TEST_DATA_DIR, TEST_OUTPUT_DIR]
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for testing
    os.environ['BIOPATH_TEST_MODE'] = 'true'
    os.environ['BIOPATH_LOG_LEVEL'] = 'WARNING'
    
    return True

def cleanup_test_environment():
    """Clean up test environment and temporary files."""
    import os
    import shutil
    from pathlib import Path
    
    # Clean up test output directory
    if Path(TEST_OUTPUT_DIR).exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    
    # Remove environment variables
    os.environ.pop('BIOPATH_TEST_MODE', None)
    os.environ.pop('BIOPATH_LOG_LEVEL', None)
    
    return True

def generate_mock_molecular_features(n_compounds: int = 10, n_features: int = 50):
    """Generate mock molecular features for testing."""
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    
    # Generate random feature matrix
    features = np.random.rand(n_compounds, n_features)
    
    # Create realistic feature names
    feature_names = []
    for i in range(n_features):
        if i < len(MOCK_FEATURE_NAMES):
            feature_names.append(MOCK_FEATURE_NAMES[i])
        else:
            feature_names.append(f'feature_{i}')
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=feature_names)
    
    # Add SMILES column
    mock_smiles = [f'C{i}CO' for i in range(n_compounds)]
    df['smiles'] = mock_smiles
    
    return df

def generate_mock_shap_values(n_compounds: int = 10, n_features: int = 50):
    """Generate mock SHAP values for testing."""
    import numpy as np
    
    np.random.seed(42)
    
    # Generate SHAP values with realistic distribution
    shap_values = np.random.normal(0, 0.1, size=(n_compounds, n_features))
    
    # Make some features more important
    important_features = [0, 1, 2, 5, 6]  # Indices of important features
    for idx in important_features:
        if idx < n_features:
            shap_values[:, idx] *= 3
    
    return shap_values

def validate_test_dependencies():
    """Validate that all required test dependencies are available."""
    missing_deps = []
    
    try:
        import pytest
    except ImportError:
        missing_deps.append('pytest')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append('scikit-learn')
    
    try:
        import rdkit
    except ImportError:
        missing_deps.append('rdkit')
    
    try:
        import shap
    except ImportError:
        missing_deps.append('shap')
    
    return missing_deps

def create_test_report(test_results: dict, output_file: str = None):
    """Create comprehensive test report."""
    import json
    from datetime import datetime
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_summary': {
            'total_tests': test_results.get('total', 0),
            'passed': test_results.get('passed', 0),
            'failed': test_results.get('failed', 0),
            'skipped': test_results.get('skipped', 0),
        },
        'coverage': test_results.get('coverage', {}),
        'performance': test_results.get('performance', {}),
        'validation_results': test_results.get('validation', {}),
        'missing_dependencies': validate_test_dependencies()
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    return report

# Test configuration for different environments
TEST_CONFIGS = {
    'unit': {
        'timeout': 60,
        'max_compounds': 10,
        'use_mock_data': True,
        'skip_slow_tests': True
    },
    'integration': {
        'timeout': 300,
        'max_compounds': 50,
        'use_mock_data': False,
        'skip_slow_tests': False
    },
    'performance': {
        'timeout': 600,
        'max_compounds': 100,
        'use_mock_data': False,
        'skip_slow_tests': False
    }
}

# Test fixtures and utilities
def get_test_config(config_name: str = 'unit'):
    """Get test configuration for specified environment."""
    return TEST_CONFIGS.get(config_name, TEST_CONFIGS['unit'])

def skip_if_missing_dependency(dependency: str):
    """Decorator to skip tests if dependency is missing."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                __import__(dependency)
                return func(*args, **kwargs)
            except ImportError:
                import pytest
                pytest.skip(f"Dependency {dependency} not available")
        return wrapper
    return decorator

def requires_rdkit(func):
    """Decorator to skip tests requiring RDKit if not available."""
    return skip_if_missing_dependency('rdkit')(func)

def requires_shap(func):
    """Decorator to skip tests requiring SHAP if not available."""
    return skip_if_missing_dependency('shap')(func)

def requires_visualization(func):
    """Decorator to skip tests requiring visualization libraries if not available."""
    def wrapper(*args, **kwargs):
        try:
            import matplotlib
            import seaborn
            return func(*args, **kwargs)
        except ImportError:
            import pytest
            pytest.skip("Visualization libraries not available")
    return wrapper

# Performance testing utilities
def benchmark_function(func, *args, **kwargs):
    """Benchmark function execution time."""
    import time
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    return {
        'result': result,
        'execution_time': execution_time,
        'function_name': func.__name__
    }

# Test data validation
def validate_molecular_features(features_df):
    """Validate molecular features dataframe structure."""
    required_columns = ['smiles', 'molecular_weight', 'logp']
    
    validation_results = {
        'has_required_columns': all(col in features_df.columns for col in required_columns),
        'no_missing_values': not features_df.isnull().any().any(),
        'valid_data_types': True,
        'reasonable_ranges': True
    }
    
    # Check data types
    try:
        float(features_df['molecular_weight'].iloc[0])
        float(features_df['logp'].iloc[0])
    except (ValueError, TypeError):
        validation_results['valid_data_types'] = False
    
    # Check reasonable ranges
    if (features_df['molecular_weight'] < 0).any() or (features_df['molecular_weight'] > 2000).any():
        validation_results['reasonable_ranges'] = False
    
    return validation_results

# Package initialization
def initialize_test_package():
    """Initialize test package with required setup."""
    try:
        # Setup test environment
        setup_test_environment()
        
        # Validate dependencies
        missing_deps = validate_test_dependencies()
        if missing_deps:
            import warnings
            warnings.warn(f"Missing test dependencies: {', '.join(missing_deps)}")
        
        return True
    except Exception as e:
        import warnings
        warnings.warn(f"Test package initialization failed: {e}")
        return False

# Initialize on import
_test_package_initialized = initialize_test_package()

# Package metadata
__all__ = [
    'setup_test_environment',
    'cleanup_test_environment',
    'generate_mock_molecular_features',
    'generate_mock_shap_values',
    'validate_test_dependencies',
    'create_test_report',
    'get_test_config',
    'skip_if_missing_dependency',
    'requires_rdkit',
    'requires_shap',
    'requires_visualization',
    'benchmark_function',
    'validate_molecular_features',
    'TEST_COMPOUNDS',
    'VALIDATION_THRESHOLDS',
    'MOCK_FEATURE_NAMES',
    'MOCK_BIOACTIVITY_DATA',
    'TEST_CONFIGS'
]

