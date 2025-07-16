"""
BioPath SHAP Demo - Data Preprocessing Package

This package provides comprehensive molecular feature calculation and preprocessing
capabilities for natural compound bioactivity analysis with explainable AI.

Key Components:
- MolecularFeatureCalculator: Main class for molecular descriptor calculation
- Enhanced RDKit integration with modern API usage
- Comprehensive feature engineering for traditional knowledge compounds
- Batch processing capabilities with progress tracking
- Feature grouping for SHAP analysis organization

Usage:
    from data_preprocessing.molecular_features import ModernMolecularFeatureCalculator
    
    calculator = ModernMolecularFeatureCalculator()
    features = calculator.calculate_all_features('CCO')
"""

__version__ = "2.0.0"
__author__ = "OmniPath Technologies"
__email__ = "biopath@omnipath.ai"

# Import main classes for easy access
from .molecular_features import ModernMolecularFeatureCalculator

# Package-level constants
SUPPORTED_FORMATS = ['smiles', 'mol', 'sdf']
DEFAULT_FINGERPRINT_RADIUS = 2
DEFAULT_FINGERPRINT_SIZE = 2048

# Feature categories for organizational purposes
FEATURE_CATEGORIES = {
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
    ]
}

# Validation functions
def validate_smiles(smiles: str) -> bool:
    """
    Validate SMILES string format.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        bool: True if valid SMILES, False otherwise
    """
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles) is not None
    except ImportError:
        raise ImportError("RDKit is required for SMILES validation")

def get_supported_descriptors() -> list:
    """
    Get list of all supported molecular descriptors.
    
    Returns:
        list: Names of all available molecular descriptors
    """
    all_descriptors = []
    for category_descriptors in FEATURE_CATEGORIES.values():
        all_descriptors.extend(category_descriptors)
    return all_descriptors

# Package initialization
def initialize_rdkit():
    """Initialize RDKit with optimal settings for molecular processing."""
    try:
        from rdkit import rdBase
        from rdkit.Chem import rdMolDescriptors
        
        # Suppress RDKit warnings
        rdBase.DisableLog('rdApp.error')
        rdBase.DisableLog('rdApp.warning')
        
        return True
    except ImportError:
        return False

# Initialize RDKit on package import
_rdkit_available = initialize_rdkit()

if not _rdkit_available:
    import warnings
    warnings.warn(
        "RDKit not available. Molecular feature calculation will be limited.",
        ImportWarning
    )

# Package metadata
__all__ = [
    'ModernMolecularFeatureCalculator',
    'validate_smiles',
    'get_supported_descriptors',
    'FEATURE_CATEGORIES',
    'SUPPORTED_FORMATS',
    'DEFAULT_FINGERPRINT_RADIUS',
    'DEFAULT_FINGERPRINT_SIZE'
]

