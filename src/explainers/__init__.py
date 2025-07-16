"""
BioPath SHAP Demo - Explainers Package

This package provides comprehensive SHAP (SHapley Additive exPlanations) analysis
capabilities for natural compound bioactivity predictions with enhanced biological
interpretations and traditional knowledge integration.

Key Components:
- ModernBioPathSHAPExplainer: Advanced SHAP explainer with biological interpretations
- BiologicalInterpretation: Data class for structured biological explanations
- Traditional knowledge integration framework
- Cultural-aware bias correction mechanisms
- Regulatory compliance explanations

Usage:
    from explainers.bio_shap_explainer import ModernBioPathSHAPExplainer
    
    explainer = ModernBioPathSHAPExplainer(model, feature_names)
    explanations = explainer.explain_instance(compound_features)
"""

__version__ = "2.0.0"
__author__ = "OmniPath Technologies"
__email__ = "biopath@omnipath.ai"

# Import main classes for easy access
from .bio_shap_explainer import ModernBioPathSHAPExplainer, BiologicalInterpretation

# Package-level constants
SUPPORTED_MODEL_TYPES = [
    'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 
    'sklearn_ensemble', 'neural_network'
]

DEFAULT_SHAP_SAMPLE_SIZE = 100
MIN_CULTURAL_REPRESENTATION = 0.30
MAX_SHAP_FEATURES = 20

# Biological interpretation confidence levels
CONFIDENCE_LEVELS = {
    'high': 0.85,
    'medium': 0.65,
    'low': 0.45
}

# Feature importance thresholds
IMPORTANCE_THRESHOLDS = {
    'strong': 0.1,
    'moderate': 0.05,
    'weak': 0.01
}

# Traditional knowledge integration weights
TRADITIONAL_KNOWLEDGE_WEIGHTS = {
    'phenolic_compounds': 0.94,
    'alkaloids': 0.89,
    'terpenoids': 0.82,
    'flavonoids': 0.91,
    'saponins': 0.76
}

# Validation functions
def validate_model_compatibility(model) -> bool:
    """
    Validate if a model is compatible with SHAP analysis.
    
    Args:
        model: Machine learning model to validate
        
    Returns:
        bool: True if compatible, False otherwise
    """
    try:
        # Check if model has predict method
        if not hasattr(model, 'predict'):
            return False
        
        # Check for tree-based models (preferred for TreeExplainer)
        model_type = str(type(model)).lower()
        if any(tree_type in model_type for tree_type in ['forest', 'tree', 'gradient']):
            return True
        
        # Check for predict_proba method (for probabilistic models)
        if hasattr(model, 'predict_proba'):
            return True
        
        return True
    except Exception:
        return False

def get_biological_interpretation_template(feature_name: str) -> dict:
    """
    Get biological interpretation template for a feature.
    
    Args:
        feature_name: Name of the molecular feature
        
    Returns:
        dict: Template with biological meanings and confidence scores
    """
    templates = {
        'molecular_weight': {
            'positive': 'Higher molecular weight indicates complex natural products with potential multi-target activity',
            'negative': 'Lower molecular weight favors better bioavailability and cellular penetration',
            'confidence': 0.85,
            'traditional_link': 'Heavy molecules often correspond to potent alkaloids in traditional preparations'
        },
        'phenol_groups': {
            'positive': 'Multiple phenolic groups strongly correlate with antioxidant and anti-inflammatory activity',
            'negative': 'Excessive phenolic content may cause protein precipitation and reduced bioavailability',
            'confidence': 0.94,
            'traditional_link': 'Phenolic compounds are primary bioactive constituents in medicinal teas and tinctures'
        },
        'logp': {
            'positive': 'Increased lipophilicity enhances membrane penetration and protein binding',
            'negative': 'Lower lipophilicity improves water solubility and reduces toxicity potential',
            'confidence': 0.92,
            'traditional_link': 'Lipophilic compounds extracted through traditional oil-based methods'
        }
    }
    
    return templates.get(feature_name, {
        'positive': f'Higher {feature_name} contributes positively to predicted bioactivity',
        'negative': f'Lower {feature_name} contributes negatively to predicted bioactivity',
        'confidence': 0.5,
        'traditional_link': 'Traditional knowledge context not available'
    })

def calculate_cultural_bias_correction(shap_values, feature_names, cultural_features):
    """
    Calculate bias correction to ensure adequate traditional knowledge representation.
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        cultural_features: List of culturally relevant features
        
    Returns:
        dict: Bias correction factors
    """
    try:
        import numpy as np
        
        # Calculate cultural feature importance
        cultural_importance = 0
        total_importance = np.sum(np.abs(shap_values))
        
        for i, feature in enumerate(feature_names):
            if feature in cultural_features:
                cultural_importance += np.abs(shap_values[i])
        
        cultural_weight = cultural_importance / total_importance if total_importance > 0 else 0
        
        # Apply correction if cultural representation is too low
        if cultural_weight < MIN_CULTURAL_REPRESENTATION:
            correction_factor = MIN_CULTURAL_REPRESENTATION / cultural_weight if cultural_weight > 0 else 1.0
            return {
                'needs_correction': True,
                'correction_factor': correction_factor,
                'cultural_weight': cultural_weight
            }
        
        return {
            'needs_correction': False,
            'correction_factor': 1.0,
            'cultural_weight': cultural_weight
        }
        
    except Exception as e:
        return {
            'needs_correction': False,
            'correction_factor': 1.0,
            'cultural_weight': 0.0,
            'error': str(e)
        }

# Utility functions for SHAP analysis
def get_feature_groups_for_analysis():
    """Get predefined feature groups for organized SHAP analysis."""
    return {
        'Basic Properties': [
            'molecular_weight', 'heavy_atom_count', 'ring_count',
            'aromatic_rings', 'rotatable_bonds'
        ],
        'Drug-likeness': [
            'logp', 'tpsa', 'hbd_count', 'hba_count', 'qed_score'
        ],
        'Natural Product Features': [
            'chiral_centers', 'phenol_groups', 'hydroxyl_groups',
            'sugar_like', 'stereocenters'
        ],
        'Structural Complexity': [
            'bertz_complexity', 'balaban_index', 'sp3_fraction'
        ],
        'Functional Groups': [
            'carbonyl_groups', 'ether_groups', 'nitrogen_containing',
            'basic_nitrogen'
        ]
    }

def format_shap_report(explanations, output_format='markdown'):
    """
    Format SHAP explanations into a structured report.
    
    Args:
        explanations: List of SHAP explanation dictionaries
        output_format: Output format ('markdown', 'html', 'json')
        
    Returns:
        str: Formatted report
    """
    if not explanations:
        return "No explanations provided for report generation."
    
    if output_format == 'markdown':
        lines = [
            "# BioPath SHAP Analysis Report",
            "## Explainable AI for Natural Compound Bioactivity",
            "",
            f"**Compounds Analyzed:** {len(explanations)}",
            ""
        ]
        
        for i, exp in enumerate(explanations, 1):
            compound_id = exp.get('compound_id', f'Compound_{i}')
            lines.extend([
                f"### {compound_id}",
                f"- **Prediction:** {'Active' if exp.get('prediction') else 'Inactive'}",
                "- **Key Features:**"
            ])
            
            interpretations = exp.get('biological_interpretations', [])
            for interp in interpretations[:5]:  # Top 5 features
                lines.append(f"  - {interp.feature_name}: {interp.biological_meaning}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    elif output_format == 'json':
        import json
        return json.dumps(explanations, indent=2, default=str)
    
    else:
        return "Unsupported output format. Use 'markdown' or 'json'."

# Package initialization
def initialize_explainers():
    """Initialize explainers package with optimal settings."""
    try:
        import warnings
        import logging
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Suppress SHAP warnings
        warnings.filterwarnings('ignore', category=FutureWarning, module='shap')
        
        return True
    except ImportError:
        return False

# Initialize package on import
_explainers_initialized = initialize_explainers()

if not _explainers_initialized:
    import warnings
    warnings.warn(
        "Some dependencies not available. SHAP analysis may be limited.",
        ImportWarning
    )

# Package metadata
__all__ = [
    'ModernBioPathSHAPExplainer',
    'BiologicalInterpretation',
    'validate_model_compatibility',
    'get_biological_interpretation_template',
    'calculate_cultural_bias_correction',
    'get_feature_groups_for_analysis',
    'format_shap_report',
    'SUPPORTED_MODEL_TYPES',
    'DEFAULT_SHAP_SAMPLE_SIZE',
    'MIN_CULTURAL_REPRESENTATION',
    'CONFIDENCE_LEVELS',
    'IMPORTANCE_THRESHOLDS',
    'TRADITIONAL_KNOWLEDGE_WEIGHTS'
]

