"""
BioPath SHAP Demo - Models Package

This package provides comprehensive machine learning models for natural compound
bioactivity prediction with ensemble methods, hyperparameter optimization, and
traditional knowledge integration capabilities.

Key Components:
- BioactivityPredictor: Main class for bioactivity prediction modeling
- ModelConfig: Configuration class for model parameters and optimization
- Ensemble methods with Random Forest, Gradient Boosting, and XGBoost
- Hyperparameter optimization with Optuna
- Cross-validation and model evaluation
- Traditional knowledge bias correction

Usage:
    from models.bioactivity_predictor import BioactivityPredictor, ModelConfig
    
    config = ModelConfig(model_type='ensemble')
    predictor = BioactivityPredictor(config)
    predictor.fit(X_train, y_train, feature_names)
"""

__version__ = "2.0.0"
__author__ = "OmniPath Technologies"
__email__ = "biopath@omnipath.ai"

# Import main classes for easy access
from .bioactivity_predictor import BioactivityPredictor, ModelConfig, PerformanceMetrics

# Package-level constants
SUPPORTED_MODEL_TYPES = [
    'ensemble', 'random_forest', 'gradient_boosting', 'xgboost', 
    'lightgbm', 'optimized'
]

DEFAULT_CV_FOLDS = 5
DEFAULT_OPTIMIZATION_TRIALS = 100
MIN_SAMPLES_FOR_OPTIMIZATION = 50

# Model performance thresholds
PERFORMANCE_THRESHOLDS = {
    'accuracy': 0.75,
    'precision': 0.70,
    'recall': 0.65,
    'f1_score': 0.70,
    'roc_auc': 0.80
}

# Traditional knowledge integration weights
TRADITIONAL_KNOWLEDGE_FEATURES = [
    'phenol_groups', 'hydroxyl_groups', 'chiral_centers', 'stereocenters',
    'carbonyl_groups', 'ether_groups', 'nitrogen_containing', 'sugar_like'
]

# Hyperparameter optimization spaces
OPTIMIZATION_SPACES = {
    'random_forest': {
        'n_estimators': (50, 500),
        'max_depth': (3, 20),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10),
        'max_features': ['sqrt', 'log2', None]
    },
    'gradient_boosting': {
        'n_estimators': (50, 300),
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10)
    },
    'xgboost': {
        'n_estimators': (50, 300),
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'min_child_weight': (1, 10),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0)
    }
}

# Validation functions
def validate_model_type(model_type: str) -> bool:
    """
    Validate if model type is supported.
    
    Args:
        model_type: Model type to validate
        
    Returns:
        bool: True if supported, False otherwise
    """
    return model_type in SUPPORTED_MODEL_TYPES

def validate_training_data(X, y) -> tuple:
    """
    Validate training data format and consistency.
    
    Args:
        X: Feature matrix
        y: Target labels
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        import numpy as np
        
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Check dimensions
        if len(X.shape) != 2:
            return False, "Feature matrix X must be 2-dimensional"
        
        if len(y.shape) != 1:
            return False, "Target labels y must be 1-dimensional"
        
        # Check sample consistency
        if X.shape[0] != y.shape[0]:
            return False, "Number of samples in X and y must match"
        
        # Check for minimum samples
        if X.shape[0] < 10:
            return False, "Minimum 10 samples required for training"
        
        # Check for valid labels (assuming binary classification)
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            return False, "Binary classification requires exactly 2 unique labels"
        
        if not all(label in [0, 1] for label in unique_labels):
            return False, "Labels must be 0 and 1 for binary classification"
        
        # Check for missing values
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            return False, "Training data contains missing values"
        
        return True, "Data validation successful"
        
    except Exception as e:
        return False, f"Data validation error: {str(e)}"

def get_model_recommendations(n_samples: int, n_features: int) -> dict:
    """
    Get model recommendations based on dataset characteristics.
    
    Args:
        n_samples: Number of training samples
        n_features: Number of features
        
    Returns:
        dict: Model recommendations and configurations
    """
    recommendations = {
        'primary_model': 'ensemble',
        'use_optimization': True,
        'cv_folds': DEFAULT_CV_FOLDS,
        'feature_selection': False
    }
    
    # Adjust recommendations based on dataset size
    if n_samples < 100:
        recommendations['primary_model'] = 'random_forest'
        recommendations['use_optimization'] = False
        recommendations['cv_folds'] = 3
    elif n_samples < 500:
        recommendations['primary_model'] = 'gradient_boosting'
        recommendations['use_optimization'] = True
        recommendations['cv_folds'] = 5
    else:
        recommendations['primary_model'] = 'ensemble'
        recommendations['use_optimization'] = True
        recommendations['cv_folds'] = 5
    
    # Feature selection recommendation
    if n_features > n_samples:
        recommendations['feature_selection'] = True
    
    return recommendations

def calculate_traditional_knowledge_bias(feature_names: list) -> float:
    """
    Calculate traditional knowledge representation bias in features.
    
    Args:
        feature_names: List of feature names
        
    Returns:
        float: Bias score (0-1, higher means better representation)
    """
    if not feature_names:
        return 0.0
    
    traditional_count = sum(1 for feature in feature_names 
                          if any(tk_feature in feature.lower() 
                                for tk_feature in TRADITIONAL_KNOWLEDGE_FEATURES))
    
    return traditional_count / len(feature_names)

def apply_cultural_bias_correction(predictions, feature_importance, bias_threshold=0.3):
    """
    Apply bias correction to ensure adequate traditional knowledge representation.
    
    Args:
        predictions: Model predictions
        feature_importance: Feature importance scores
        bias_threshold: Minimum traditional knowledge representation threshold
        
    Returns:
        dict: Corrected predictions and bias metrics
    """
    try:
        import numpy as np
        
        # Calculate current traditional knowledge representation
        tk_importance = sum(importance for feature, importance in feature_importance.items()
                          if any(tk_feature in feature.lower() 
                                for tk_feature in TRADITIONAL_KNOWLEDGE_FEATURES))
        
        total_importance = sum(feature_importance.values())
        tk_ratio = tk_importance / total_importance if total_importance > 0 else 0
        
        # Apply correction if representation is too low
        if tk_ratio < bias_threshold:
            correction_factor = bias_threshold / tk_ratio if tk_ratio > 0 else 1.0
            
            # Apply mild correction to predictions
            corrected_predictions = predictions.copy()
            correction_strength = min(0.1, (bias_threshold - tk_ratio) / bias_threshold)
            
            # Boost predictions slightly for compounds with traditional knowledge features
            for i in range(len(corrected_predictions)):
                if corrected_predictions[i] > 0.5:  # Only boost active predictions
                    corrected_predictions[i] = min(1.0, corrected_predictions[i] + correction_strength)
            
            return {
                'corrected_predictions': corrected_predictions,
                'bias_detected': True,
                'original_tk_ratio': tk_ratio,
                'correction_factor': correction_factor,
                'correction_strength': correction_strength
            }
        
        return {
            'corrected_predictions': predictions,
            'bias_detected': False,
            'original_tk_ratio': tk_ratio,
            'correction_factor': 1.0,
            'correction_strength': 0.0
        }
        
    except Exception as e:
        return {
            'corrected_predictions': predictions,
            'bias_detected': False,
            'error': str(e)
        }

# Utility functions for model evaluation
def calculate_performance_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate comprehensive performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        dict: Performance metrics
    """
    try:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        return metrics
        
    except Exception as e:
        return {'error': str(e)}

def format_model_report(performance_metrics, model_config, feature_names):
    """
    Format comprehensive model performance report.
    
    Args:
        performance_metrics: Performance metrics dictionary
        model_config: Model configuration
        feature_names: List of feature names
        
    Returns:
        str: Formatted report
    """
    lines = [
        "# BioPath Model Performance Report",
        "## Natural Compound Bioactivity Prediction",
        "",
        f"**Model Type:** {model_config.model_type}",
        f"**Features:** {len(feature_names)}",
        f"**Cross-Validation:** {model_config.cross_validation_folds} folds",
        ""
    ]
    
    if 'accuracy' in performance_metrics:
        lines.extend([
            "## Performance Metrics",
            f"- **Accuracy:** {performance_metrics['accuracy']:.3f}",
            f"- **Precision:** {performance_metrics['precision']:.3f}",
            f"- **Recall:** {performance_metrics['recall']:.3f}",
            f"- **F1-Score:** {performance_metrics['f1_score']:.3f}",
            ""
        ])
    
    # Traditional knowledge bias assessment
    tk_bias = calculate_traditional_knowledge_bias(feature_names)
    lines.extend([
        "## Traditional Knowledge Integration",
        f"- **Traditional Feature Representation:** {tk_bias:.1%}",
        f"- **Bias Status:** {'Adequate' if tk_bias > 0.3 else 'Requires Attention'}",
        ""
    ])
    
    return "\n".join(lines)

# Package initialization
def initialize_models_package():
    """Initialize models package with optimal settings."""
    try:
        import warnings
        import logging
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Suppress sklearn warnings
        warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
        
        # Check for optional dependencies
        optional_deps = {}
        
        try:
            import optuna
            optional_deps['optuna'] = True
        except ImportError:
            optional_deps['optuna'] = False
        
        try:
            import xgboost
            optional_deps['xgboost'] = True
        except ImportError:
            optional_deps['xgboost'] = False
        
        try:
            import lightgbm
            optional_deps['lightgbm'] = True
        except ImportError:
            optional_deps['lightgbm'] = False
        
        return True, optional_deps
        
    except Exception as e:
        return False, {'error': str(e)}

# Initialize package on import
_models_initialized, _optional_deps = initialize_models_package()

if not _models_initialized:
    import warnings
    warnings.warn(
        "Models package initialization failed. Some functionality may be limited.",
        ImportWarning
    )

# Package metadata
__all__ = [
    'BioactivityPredictor',
    'ModelConfig',
    'PerformanceMetrics',
    'validate_model_type',
    'validate_training_data',
    'get_model_recommendations',
    'calculate_traditional_knowledge_bias',
    'apply_cultural_bias_correction',
    'calculate_performance_metrics',
    'format_model_report',
    'SUPPORTED_MODEL_TYPES',
    'DEFAULT_CV_FOLDS',
    'DEFAULT_OPTIMIZATION_TRIALS',
    'PERFORMANCE_THRESHOLDS',
    'TRADITIONAL_KNOWLEDGE_FEATURES',
    'OPTIMIZATION_SPACES'
]

