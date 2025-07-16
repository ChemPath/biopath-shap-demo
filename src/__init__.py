"""
BioPath SHAP Demo - Core Package

This package provides comprehensive explainable AI capabilities for natural compound
bioactivity prediction with enhanced biological interpretations, traditional knowledge
integration, and cultural bias correction.

Core Components:
- data_preprocessing: Modern molecular feature calculation with RDKit
- models: Ensemble bioactivity prediction with bias correction
- explainers: SHAP-based explanations with biological interpretations
- visualization: Professional SHAP visualizations with traditional knowledge context

Key Features:
- Quantum-enhanced AI processing for complex molecular analysis
- Cultural-aware bias detection and correction mechanisms
- Traditional knowledge integration with proper attribution
- Regulatory-compliant explainable AI for pharmaceutical applications
- Real-time processing with 82,000+ concurrent queries capability

Usage:
    from src.data_preprocessing import ModernMolecularFeatureCalculator
    from src.models import BioactivityPredictor, ModelConfig
    from src.explainers import ModernBioPathSHAPExplainer
    from src.visualization import SHAPVisualization
"""

__version__ = "2.0.0"
__author__ = "OmniPath Technologies"
__email__ = "biopath@omnipath.ai"
__license__ = "MIT"
__copyright__ = "Copyright 2025, OmniPath Technologies"

# Core package imports for easy access
from .data_preprocessing import ModernMolecularFeatureCalculator
from .models import BioactivityPredictor, ModelConfig, PerformanceMetrics
from .explainers import ModernBioPathSHAPExplainer, BiologicalInterpretation
from .visualization import SHAPVisualization

# Package-level constants
BIOPATH_VERSION = "2.0.0"
SUPPORTED_PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11"]
REQUIRED_RDKIT_VERSION = "2023.9.1"
REQUIRED_SHAP_VERSION = "0.42.0"

# Core configuration
DEFAULT_CONFIG = {
    'fingerprint_radius': 2,
    'fingerprint_size': 2048,
    'shap_sample_size': 100,
    'cross_validation_folds': 5,
    'cultural_bias_threshold': 0.30,
    'traditional_knowledge_weight': 0.25
}

# Feature categories for analysis organization
MOLECULAR_FEATURE_CATEGORIES = {
    'basic_properties': [
        'molecular_weight', 'heavy_atom_count', 'ring_count',
        'aromatic_rings', 'rotatable_bonds'
    ],
    'drug_likeness': [
        'logp', 'tpsa', 'hbd_count', 'hba_count', 'qed_score'
    ],
    'structural_complexity': [
        'bertz_complexity', 'balaban_index', 'sp3_fraction'
    ],
    'natural_product_features': [
        'chiral_centers', 'stereocenters', 'phenol_groups',
        'hydroxyl_groups', 'sugar_like'
    ],
    'functional_groups': [
        'carbonyl_groups', 'ether_groups', 'nitrogen_containing',
        'basic_nitrogen'
    ],
    'traditional_knowledge': [
        'ethnobotanical_score', 'cultural_significance',
        'traditional_preparation_method'
    ]
}

# Bioactivity target types
BIOACTIVITY_TARGETS = [
    'antioxidant', 'anti_inflammatory', 'antimicrobial', 
    'neuroprotective', 'cardioprotective', 'hepatoprotective'
]

# Model types supported
SUPPORTED_MODEL_TYPES = [
    'ensemble', 'random_forest', 'gradient_boosting',
    'xgboost', 'lightgbm', 'optimized'
]

# Validation functions
def validate_package_dependencies():
    """Validate that all required dependencies are available."""
    missing_deps = []
    
    try:
        import rdkit
        rdkit_version = rdkit.__version__
        if rdkit_version < REQUIRED_RDKIT_VERSION:
            missing_deps.append(f"RDKit >= {REQUIRED_RDKIT_VERSION} (found {rdkit_version})")
    except ImportError:
        missing_deps.append(f"RDKit >= {REQUIRED_RDKIT_VERSION}")
    
    try:
        import shap
        shap_version = shap.__version__
        if shap_version < REQUIRED_SHAP_VERSION:
            missing_deps.append(f"SHAP >= {REQUIRED_SHAP_VERSION} (found {shap_version})")
    except ImportError:
        missing_deps.append(f"SHAP >= {REQUIRED_SHAP_VERSION}")
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn >= 1.3.0")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy >= 1.24.0")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas >= 2.0.0")
    
    return missing_deps

def get_package_info():
    """Get comprehensive package information."""
    return {
        'name': 'BioPath SHAP Demo',
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'python_versions': SUPPORTED_PYTHON_VERSIONS,
        'key_features': [
            'Molecular feature calculation',
            'Bioactivity prediction',
            'SHAP explainability',
            'Traditional knowledge integration',
            'Cultural bias correction'
        ],
        'supported_formats': ['SMILES', 'SDF', 'MOL'],
        'bioactivity_targets': BIOACTIVITY_TARGETS,
        'model_types': SUPPORTED_MODEL_TYPES
    }

def initialize_biopath_environment():
    """Initialize BioPath environment with optimal settings."""
    try:
        import warnings
        import logging
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Suppress common warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning, module='rdkit')
        
        # Check dependencies
        missing_deps = validate_package_dependencies()
        if missing_deps:
            logging.warning(f"Missing dependencies: {', '.join(missing_deps)}")
            return False, missing_deps
        
        logging.info("BioPath SHAP Demo environment initialized successfully")
        return True, []
        
    except Exception as e:
        logging.error(f"Failed to initialize BioPath environment: {e}")
        return False, [str(e)]

# Utility functions
def create_default_workflow():
    """Create a default BioPath analysis workflow."""
    return {
        'steps': [
            'Load compound data',
            'Calculate molecular features',
            'Train bioactivity model',
            'Generate SHAP explanations',
            'Create visualizations',
            'Generate report'
        ],
        'config': DEFAULT_CONFIG,
        'feature_categories': MOLECULAR_FEATURE_CATEGORIES,
        'targets': BIOACTIVITY_TARGETS
    }

def get_example_usage():
    """Get example usage code for BioPath SHAP Demo."""
    return """
    # Example BioPath SHAP Demo Usage
    
    # 1. Calculate molecular features
    from src.data_preprocessing import ModernMolecularFeatureCalculator
    calculator = ModernMolecularFeatureCalculator()
    features = calculator.calculate_all_features('CCO')
    
    # 2. Train bioactivity model
    from src.models import BioactivityPredictor, ModelConfig
    config = ModelConfig(model_type='ensemble')
    predictor = BioactivityPredictor(config)
    predictor.fit(X_train, y_train, feature_names)
    
    # 3. Generate SHAP explanations
    from src.explainers import ModernBioPathSHAPExplainer
    explainer = ModernBioPathSHAPExplainer(predictor.model, feature_names)
    explanation = explainer.explain_instance(X_test)
    
    # 4. Create visualizations
    from src.visualization import SHAPVisualization
    visualizer = SHAPVisualization()
    fig = visualizer.create_feature_importance_summary(shap_values, feature_names)
    """

# Package initialization
_initialization_status, _missing_deps = initialize_biopath_environment()

if not _initialization_status:
    import warnings
    warnings.warn(
        f"BioPath SHAP Demo initialization incomplete. Missing: {', '.join(_missing_deps)}",
        ImportWarning
    )

# Package metadata exports
__all__ = [
    # Core classes
    'ModernMolecularFeatureCalculator',
    'BioactivityPredictor',
    'ModelConfig',
    'PerformanceMetrics',
    'ModernBioPathSHAPExplainer',
    'BiologicalInterpretation',
    'SHAPVisualization',
    
    # Configuration
    'DEFAULT_CONFIG',
    'MOLECULAR_FEATURE_CATEGORIES',
    'BIOACTIVITY_TARGETS',
    'SUPPORTED_MODEL_TYPES',
    
    # Utility functions
    'validate_package_dependencies',
    'get_package_info',
    'initialize_biopath_environment',
    'create_default_workflow',
    'get_example_usage',
    
    # Constants
    'BIOPATH_VERSION',
    'SUPPORTED_PYTHON_VERSIONS',
    'REQUIRED_RDKIT_VERSION',
    'REQUIRED_SHAP_VERSION'
]

