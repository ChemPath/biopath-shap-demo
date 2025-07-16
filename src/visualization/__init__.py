"""
BioPath SHAP Demo - Visualization Package

This package provides comprehensive visualization capabilities for SHAP analysis
of natural compound bioactivity predictions with enhanced biological context
and traditional knowledge integration.

Key Components:
- SHAPVisualization: Advanced SHAP plotting with biological interpretations
- Professional visualization styles for publication-ready plots
- Interactive plotting capabilities with Plotly integration
- Feature grouping and clustering for organized analysis
- Cultural bias visualization and correction displays

Usage:
    from visualization.shap_plots import SHAPVisualization
    
    visualizer = SHAPVisualization(feature_groups=groups)
    fig = visualizer.create_feature_importance_summary(shap_values, feature_names)
"""

__version__ = "2.0.0"
__author__ = "OmniPath Technologies"
__email__ = "biopath@omnipath.ai"

# Import main classes for easy access
from .shap_plots import SHAPVisualization

# Package-level constants
VISUALIZATION_STYLES = ['professional', 'scientific', 'presentation']
DEFAULT_STYLE = 'professional'
MAX_FEATURES_DISPLAY = 25
DEFAULT_FIGURE_SIZE = (12, 8)

# Color palettes for different visualization styles
COLOR_PALETTES = {
    'professional': {
        'positive': '#2E86AB',
        'negative': '#A23B72',
        'neutral': '#F18F01',
        'background': '#F8F9FA'
    },
    'scientific': {
        'positive': '#1f77b4',
        'negative': '#d62728',
        'neutral': '#ff7f0e',
        'background': '#ffffff'
    },
    'presentation': {
        'positive': '#00A651',
        'negative': '#ED1C24',
        'neutral': '#FFB81C',
        'background': '#f5f5f5'
    }
}

# Feature importance thresholds for visualization
IMPORTANCE_THRESHOLDS = {
    'high': 0.1,
    'medium': 0.05,
    'low': 0.01
}

# Traditional knowledge visualization settings
TRADITIONAL_KNOWLEDGE_COLORS = {
    'phenolic_compounds': '#8B4513',
    'alkaloids': '#4B0082',
    'terpenoids': '#228B22',
    'flavonoids': '#FF6347',
    'saponins': '#DAA520'
}

# Validation functions
def validate_shap_data(shap_values, feature_names):
    """
    Validate SHAP data for visualization compatibility.
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        import numpy as np
        
        shap_values = np.asarray(shap_values)
        
        if len(shap_values.shape) != 2:
            return False, "SHAP values must be 2-dimensional (samples x features)"
        
        if shap_values.shape[1] != len(feature_names):
            return False, "Number of features in SHAP values must match feature names"
        
        if len(feature_names) == 0:
            return False, "Feature names cannot be empty"
        
        # Check for valid numeric values
        if np.any(np.isnan(shap_values)) or np.any(np.isinf(shap_values)):
            return False, "SHAP values contain invalid numeric values"
        
        return True, "Validation successful"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def get_optimal_figure_size(n_features: int, plot_type: str = 'horizontal_bar'):
    """
    Get optimal figure size based on number of features and plot type.
    
    Args:
        n_features: Number of features to display
        plot_type: Type of plot ('horizontal_bar', 'beeswarm', 'heatmap')
        
    Returns:
        tuple: (width, height) in inches
    """
    base_sizes = {
        'horizontal_bar': (12, max(6, n_features * 0.3)),
        'beeswarm': (14, max(8, n_features * 0.2)),
        'heatmap': (max(10, n_features * 0.4), max(8, n_features * 0.3))
    }
    
    return base_sizes.get(plot_type, DEFAULT_FIGURE_SIZE)

def configure_plot_style(style: str = DEFAULT_STYLE):
    """
    Configure matplotlib style for consistent plotting.
    
    Args:
        style: Visualization style ('professional', 'scientific', 'presentation')
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if style not in COLOR_PALETTES:
            style = DEFAULT_STYLE
        
        # Set style based on type
        if style == 'professional':
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
        elif style == 'scientific':
            plt.style.use('seaborn-v0_8-paper')
            sns.set_palette("deep")
        elif style == 'presentation':
            plt.style.use('seaborn-v0_8-talk')
            sns.set_palette("bright")
        
        # Set consistent parameters
        plt.rcParams['figure.figsize'] = DEFAULT_FIGURE_SIZE
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        
    except ImportError:
        pass  # Graceful degradation if matplotlib not available

def create_feature_labels(feature_names: list, max_length: int = 25):
    """
    Create readable feature labels for visualization.
    
    Args:
        feature_names: List of feature names
        max_length: Maximum label length
        
    Returns:
        list: Formatted feature labels
    """
    formatted_labels = []
    
    for name in feature_names:
        # Replace underscores with spaces and title case
        formatted = name.replace('_', ' ').title()
        
        # Truncate if too long
        if len(formatted) > max_length:
            formatted = formatted[:max_length-3] + '...'
        
        formatted_labels.append(formatted)
    
    return formatted_labels

def calculate_plot_metrics(shap_values, feature_names):
    """
    Calculate useful metrics for plot optimization.
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        
    Returns:
        dict: Plot metrics and recommendations
    """
    try:
        import numpy as np
        
        shap_values = np.asarray(shap_values)
        
        metrics = {
            'n_samples': shap_values.shape[0],
            'n_features': shap_values.shape[1],
            'mean_importance': np.mean(np.abs(shap_values), axis=0),
            'max_importance': np.max(np.abs(shap_values)),
            'feature_variance': np.var(shap_values, axis=0),
            'recommended_max_features': min(MAX_FEATURES_DISPLAY, shap_values.shape[1])
        }
        
        # Add recommendations
        if metrics['n_features'] > MAX_FEATURES_DISPLAY:
            metrics['recommendation'] = f"Consider showing top {MAX_FEATURES_DISPLAY} features"
        else:
            metrics['recommendation'] = "All features can be displayed"
        
        return metrics
        
    except Exception as e:
        return {'error': str(e)}

# Utility functions for traditional knowledge visualization
def get_traditional_knowledge_features(feature_names: list):
    """
    Identify traditional knowledge-related features.
    
    Args:
        feature_names: List of feature names
        
    Returns:
        dict: Traditional knowledge features by category
    """
    tk_features = {
        'phenolic_compounds': [],
        'alkaloids': [],
        'terpenoids': [],
        'flavonoids': [],
        'saponins': []
    }
    
    # Feature patterns for traditional knowledge categories
    patterns = {
        'phenolic_compounds': ['phenol', 'hydroxyl', 'caffeic', 'gallic', 'protocatechuic'],
        'alkaloids': ['nitrogen', 'caffeine', 'berberine', 'morphine'],
        'terpenoids': ['limonene', 'menthol', 'camphor', 'steroid'],
        'flavonoids': ['quercetin', 'kaempferol', 'catechin', 'anthocyanin'],
        'saponins': ['glycoside', 'sugar', 'triterpene']
    }
    
    for feature in feature_names:
        feature_lower = feature.lower()
        for category, pattern_list in patterns.items():
            if any(pattern in feature_lower for pattern in pattern_list):
                tk_features[category].append(feature)
    
    return tk_features

# Package initialization
def initialize_visualization_package():
    """Initialize visualization package with optimal settings."""
    try:
        # Configure default plotting style
        configure_plot_style(DEFAULT_STYLE)
        
        # Check for required dependencies
        deps_available = {}
        
        try:
            import matplotlib.pyplot as plt
            deps_available['matplotlib'] = True
        except ImportError:
            deps_available['matplotlib'] = False
        
        try:
            import seaborn as sns
            deps_available['seaborn'] = True
        except ImportError:
            deps_available['seaborn'] = False
        
        try:
            import plotly.graph_objects as go
            deps_available['plotly'] = True
        except ImportError:
            deps_available['plotly'] = False
        
        return True, deps_available
        
    except Exception as e:
        return False, {'error': str(e)}

# Initialize package on import
_viz_initialized, _deps_status = initialize_visualization_package()

if not _viz_initialized:
    import warnings
    warnings.warn(
        "Visualization package initialization failed. Some features may be limited.",
        ImportWarning
    )

# Package metadata
__all__ = [
    'SHAPVisualization',
    'validate_shap_data',
    'get_optimal_figure_size',
    'configure_plot_style',
    'create_feature_labels',
    'calculate_plot_metrics',
    'get_traditional_knowledge_features',
    'VISUALIZATION_STYLES',
    'DEFAULT_STYLE',
    'COLOR_PALETTES',
    'IMPORTANCE_THRESHOLDS',
    'TRADITIONAL_KNOWLEDGE_COLORS'
]

