"""
Advanced SHAP Visualization Module for BioPath Demo

This module provides sophisticated visualization tools specifically designed for 
molecular bioactivity SHAP analysis, with emphasis on publication-ready plots
and investor presentation materials.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap

# Configure plotting styles
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)

# Color schemes for molecular feature categories
FEATURE_COLOR_SCHEME = {
    'Basic Properties': '#1f77b4',
    'Drug-likeness': '#ff7f0e', 
    'Structural Complexity': '#2ca02c',
    'Natural Product Features': '#d62728',
    'Functional Groups': '#9467bd',
    'Pharmacophores': '#8c564b',
    'Fingerprint Features': '#e377c2',
    'Ethnobotanical': '#17becf',
    'Other': '#bcbd22'
}


class SHAPVisualization:
    """
    Advanced SHAP visualization toolkit for molecular bioactivity analysis.
    
    This class provides publication-ready and investor-presentation-quality
    visualizations for SHAP analysis results.
    """
    
    def __init__(self, 
                 feature_groups: Optional[Dict[str, List[str]]] = None,
                 color_scheme: Optional[Dict[str, str]] = None,
                 style: str = 'professional'):
        """
        Initialize the SHAP visualization toolkit.
        
        Args:
            feature_groups: Dictionary mapping feature group names to feature lists
            color_scheme: Custom color mapping for feature groups
            style: Visualization style ('professional', 'scientific', 'presentation')
        """
        self.feature_groups = feature_groups or {}
        self.color_scheme = color_scheme or FEATURE_COLOR_SCHEME
        self.style = style
        
        # Configure style settings
        self._configure_style()
        
        logging.info(f"SHAPVisualization initialized with {style} style")
    
    def _configure_style(self):
        """Configure matplotlib and seaborn settings based on style."""
        if self.style == 'professional':
            plt.rcParams.update({
                'font.size': 12,
                'axes.titlesize': 16,
                'axes.labelsize': 14,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'legend.fontsize': 11,
                'figure.titlesize': 18,
                'font.family': 'sans-serif',
                'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans']
            })
        elif self.style == 'presentation':
            plt.rcParams.update({
                'font.size': 14,
                'axes.titlesize': 20,
                'axes.labelsize': 16,
                'xtick.labelsize': 13,
                'ytick.labelsize': 13,
                'legend.fontsize': 13,
                'figure.titlesize': 22,
                'font.weight': 'bold'
            })
    
    def create_feature_importance_summary(self, 
                                        shap_values: np.ndarray,
                                        feature_names: List[str],
                                        max_features: int = 20,
                                        title: str = "Molecular Feature Importance") -> plt.Figure:
        """
        Create a comprehensive feature importance summary plot.
        
        Args:
            shap_values: SHAP values matrix (n_samples x n_features)
            feature_names: List of feature names
            max_features: Maximum features to display
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Get top features
        top_indices = np.argsort(mean_abs_shap)[-max_features:]
        top_features = [feature_names[i] for i in top_indices]
        top_values = mean_abs_shap[top_indices]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Horizontal bar chart
        colors = [self._get_feature_color(feat) for feat in top_features]
        bars = ax1.barh(range(len(top_features)), top_values, color=colors, alpha=0.8)
        
        # Format feature names
        formatted_names = [self._format_feature_name(name) for name in top_features]
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(formatted_names)
        ax1.set_xlabel('Mean |SHAP Value|')
        ax1.set_title('Feature Importance Ranking')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_values)):
            ax1.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontsize=10)
        
        # Right plot: Feature group distribution
        self._plot_feature_group_distribution(ax2, top_features)
        
        # Overall title
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        
        # Add legend for feature groups
        self._add_comprehensive_legend(fig, top_features)
        
        plt.tight_layout()
        return fig
    
    def create_shap_waterfall_interactive(self, 
                                        explanation: Dict[str, Any],
                                        max_features: int = 15) -> go.Figure:
        """
        Create an interactive waterfall plot using Plotly.
        
        Args:
            explanation: SHAP explanation dictionary
            max_features: Maximum features to show
            
        Returns:
            Plotly figure object
        """
        shap_values = explanation['shap_values']
        feature_names = explanation['feature_names']
        feature_values = explanation['feature_values']
        base_value = explanation['base_value']
        
        # Get top contributing features
        feature_importance = list(zip(feature_names, shap_values, feature_values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Prepare data
        top_features = feature_importance[:max_features]
        names = ['Base Value'] + [self._format_feature_name(name) for name, _, _ in top_features] + ['Prediction']
        values = [base_value] + [val for _, val, _ in top_features] + [0]  # Final value calculated below
        
        # Calculate cumulative values
        cumulative = [base_value]
        for _, val, _ in top_features:
            cumulative.append(cumulative[-1] + val)
        final_prediction = cumulative[-1]
        values[-1] = final_prediction  # Set final prediction value
        cumulative.append(final_prediction)
        
        # Create plotly waterfall
        fig = go.Figure()
        
        # Add base value
        fig.add_trace(go.Waterfall(
            name="SHAP Analysis",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(top_features) + ["total"],
            x=names,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        # Add feature value annotations
        annotations = []
        for i, (name, shap_val, feat_val) in enumerate(top_features):
            annotations.append(
                dict(
                    x=i + 1,
                    y=cumulative[i + 1],
                    text=f"Value: {feat_val:.3f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="gray",
                    ax=0,
                    ay=-30,
                    font=dict(size=10)
                )
            )
        
        fig.update_layout(
            title="Bioactivity Prediction Breakdown",
            xaxis_title="Molecular Features",
            yaxis_title="Contribution to Prediction",
            annotations=annotations,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_shap_beeswarm_plot(self, 
                                shap_values: np.ndarray,
                                feature_values: np.ndarray,
                                feature_names: List[str],
                                max_features: int = 20) -> plt.Figure:
        """
        Create a SHAP beeswarm plot showing feature value distributions.
        
        Args:
            shap_values: SHAP values matrix
            feature_values: Feature values matrix
            feature_names: Feature names
            max_features: Maximum features to display
            
        Returns:
            Matplotlib figure
        """
        # Use SHAP's built-in beeswarm plot with customization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create SHAP summary plot
        shap.plots.beeswarm(
            shap.Explanation(
                values=shap_values,
                data=feature_values,
                feature_names=feature_names
            ),
            max_display=max_features,
            show=False
        )
        
        plt.title("SHAP Feature Impact Distribution", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=14)
        plt.tight_layout()
        
        return fig.get_figure() if hasattr(fig, 'get_figure') else plt.gcf()
    
    def create_molecular_heatmap(self, 
                               shap_values: np.ndarray,
                               feature_names: List[str],
                               compound_names: Optional[List[str]] = None,
                               cluster_features: bool = True) -> plt.Figure:
        """
        Create a heatmap showing SHAP values across compounds and features.
        
        Args:
            shap_values: SHAP values matrix (compounds x features)
            feature_names: List of feature names
            compound_names: Optional compound identifiers
            cluster_features: Whether to cluster features by similarity
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        n_compounds, n_features = shap_values.shape
        compound_names = compound_names or [f"Compound_{i+1}" for i in range(n_compounds)]
        
        # Select top features by variance
        feature_variance = np.var(shap_values, axis=0)
        top_feature_indices = np.argsort(feature_variance)[-25:]  # Top 25 most variable features
        
        selected_shap = shap_values[:, top_feature_indices]
        selected_features = [feature_names[i] for i in top_feature_indices]
        selected_formatted = [self._format_feature_name(name) for name in selected_features]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Use clustering if requested
        if cluster_features and len(selected_features) > 5:
            sns.clustermap(
                selected_shap.T,
                xticklabels=compound_names,
                yticklabels=selected_formatted,
                cmap='RdBu_r',
                center=0,
                figsize=(14, 10),
                cbar_kws={'label': 'SHAP Value'}
            )
            return plt.gcf()
        else:
            im = ax.imshow(selected_shap.T, cmap='RdBu_r', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(range(len(compound_names)))
            ax.set_xticklabels(compound_names, rotation=45, ha='right')
            ax.set_yticks(range(len(selected_formatted)))
            ax.set_yticklabels(selected_formatted)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('SHAP Value', rotation=270, labelpad=20)
            
            ax.set_title('SHAP Values Across Compounds and Features', fontsize=16, fontweight='bold')
            ax.set_xlabel('Compounds')
            ax.set_ylabel('Molecular Features')
            
            plt.tight_layout()
            return fig
    
    def create_feature_group_comparison(self, 
                                      shap_values: np.ndarray,
                                      feature_names: List[str]) -> plt.Figure:
        """
        Create a comparison plot of different feature groups' contributions.
        
        Args:
            shap_values: SHAP values matrix
            feature_names: Feature names
            
        Returns:
            Matplotlib figure
        """
        # Calculate group contributions
        group_contributions = {}
        
        for group_name, group_features in self.feature_groups.items():
            group_indices = [i for i, name in enumerate(feature_names) if name in group_features]
            if group_indices:
                group_shap = shap_values[:, group_indices]
                group_contributions[group_name] = {
                    'mean_abs': np.mean(np.abs(group_shap)),
                    'std': np.std(np.abs(group_shap)),
                    'total_impact': np.sum(np.abs(group_shap))
                }
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Mean absolute contribution by group
        groups = list(group_contributions.keys())
        mean_values = [group_contributions[g]['mean_abs'] for g in groups]
        colors = [self.color_scheme.get(g, '#666666') for g in groups]
        
        ax1.bar(groups, mean_values, color=colors, alpha=0.7)
        ax1.set_title('Mean Absolute SHAP by Feature Group')
        ax1.set_ylabel('Mean |SHAP Value|')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Total impact by group
        total_values = [group_contributions[g]['total_impact'] for g in groups]
        ax2.pie(total_values, labels=groups, colors=colors, autopct='%1.1f%%')
        ax2.set_title('Total Impact Distribution')
        
        # Plot 3: Variability by group
        std_values = [group_contributions[g]['std'] for g in groups]
        ax3.bar(groups, std_values, color=colors, alpha=0.7)
        ax3.set_title('Feature Group Variability')
        ax3.set_ylabel('Standard Deviation')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Efficiency (impact per feature)
        efficiency = []
        for group in groups:
            group_features = self.feature_groups[group]
            n_features = len([f for f in group_features if f in feature_names])
            if n_features > 0:
                efficiency.append(group_contributions[group]['total_impact'] / n_features)
            else:
                efficiency.append(0)
        
        ax4.bar(groups, efficiency, color=colors, alpha=0.7)
        ax4.set_title('Feature Group Efficiency (Impact per Feature)')
        ax4.set_ylabel('Impact per Feature')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Molecular Feature Group Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _get_feature_color(self, feature_name: str) -> str:
        """Get color for a feature based on its group."""
        for group_name, group_features in self.feature_groups.items():
            if feature_name in group_features:
                return self.color_scheme.get(group_name, '#666666')
        return self.color_scheme.get('Other', '#666666')
    
    def _format_feature_name(self, name: str) -> str:
        """Format feature names for better readability."""
        formatted = name.replace('_', ' ').title()
        
        # Handle chemical abbreviations
        replacements = {
            'Logp': 'LogP',
            'Tpsa': 'TPSA',
            'Qed': 'QED',
            'Hbd': 'HBD',
            'Hba': 'HBA',
            'Mol ': 'Molecular ',
            'Bertz': 'Bertz',
            'Balaban': 'Balaban'
        }
        
        for old, new in replacements.items():
            formatted = formatted.replace(old, new)
        
        return formatted
    
    def _plot_feature_group_distribution(self, ax: plt.Axes, feature_names: List[str]):
        """Plot distribution of features across groups."""
        # Count features per group
        group_counts = {}
        for feature in feature_names:
            group_found = False
            for group_name, group_features in self.feature_groups.items():
                if feature in group_features:
                    group_counts[group_name] = group_counts.get(group_name, 0) + 1
                    group_found = True
                    break
            if not group_found:
                group_counts['Other'] = group_counts.get('Other', 0) + 1
        
        # Create pie chart
        groups = list(group_counts.keys())
        counts = list(group_counts.values())
        colors = [self.color_scheme.get(g, '#666666') for g in groups]
        
        ax.pie(counts, labels=groups, colors=colors, autopct='%1.0f%%')
        ax.set_title('Feature Group Distribution')
    
    def _add_comprehensive_legend(self, fig: plt.Figure, feature_names: List[str]):
        """Add a comprehensive legend for feature groups."""
        from matplotlib.patches import Patch
        
        # Get unique groups in the plot
        groups_in_plot = set()
        for feature in feature_names:
            for group_name, group_features in self.feature_groups.items():
                if feature in group_features:
                    groups_in_plot.add(group_name)
                    break
        
        if len(groups_in_plot) > 1:
            legend_elements = []
            for group in groups_in_plot:
                color = self.color_scheme.get(group, '#666666')
                legend_elements.append(Patch(facecolor=color, label=group))
            
            fig.legend(handles=legend_elements, 
                      loc='center', 
                      bbox_to_anchor=(0.5, 0.02),
                      ncol=min(len(legend_elements), 4),
                      fontsize=10)
    
    def save_publication_ready_figure(self, 
                                    fig: plt.Figure, 
                                    filename: str,
                                    dpi: int = 300,
                                    formats: List[str] = ['png', 'pdf']) -> List[str]:
        """
        Save figure in publication-ready formats.
        
        Args:
            fig: Matplotlib figure
            filename: Base filename (without extension)
            dpi: Resolution for raster formats
            formats: List of formats to save
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        for fmt in formats:
            filepath = f"{filename}.{fmt}"
            fig.savefig(
                filepath,
                format=fmt,
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            saved_files.append(filepath)
            logging.info(f"Saved publication-ready figure: {filepath}")
        
        return saved_files


# Utility functions for external use
def create_summary_dashboard(shap_values: np.ndarray,
                           feature_values: np.ndarray,
                           feature_names: List[str],
                           feature_groups: Dict[str, List[str]],
                           compound_names: Optional[List[str]] = None) -> plt.Figure:
    """
    Create a comprehensive summary dashboard of SHAP analysis.
    
    Args:
        shap_values: SHAP values matrix
        feature_values: Feature values matrix  
        feature_names: Feature names
        feature_groups: Feature grouping dictionary
        compound_names: Optional compound identifiers
        
    Returns:
        Matplotlib figure with comprehensive dashboard
    """
    visualizer = SHAPVisualization(feature_groups=feature_groups, style='presentation')
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Feature importance summary
    ax1 = plt.subplot(2, 3, 1)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[-15:]
    top_features = [feature_names[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]
    
    colors = [visualizer._get_feature_color(feat) for feat in top_features]
    ax1.barh(range(len(top_features)), top_values, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels([visualizer._format_feature_name(name) for name in top_features])
    ax1.set_title('Top Features by Importance')
    ax1.set_xlabel('Mean |SHAP Value|')
    
    # Feature group distribution  
    ax2 = plt.subplot(2, 3, 2)
    visualizer._plot_feature_group_distribution(ax2, top_features)
    
    # SHAP value distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(shap_values.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('SHAP Value Distribution')
    ax3.set_xlabel('SHAP Value')
    ax3.set_ylabel('Frequency')
    
    # Feature correlation heatmap (subset)
    ax4 = plt.subplot(2, 3, 4)
    if len(top_features) > 5:
        top_feature_indices = [feature_names.index(f) for f in top_features[-10:]]
        correlation_matrix = np.corrcoef(feature_values[:, top_feature_indices].T)
        im = ax4.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(top_features[-10:])))
        ax4.set_yticks(range(len(top_features[-10:])))
        ax4.set_xticklabels([visualizer._format_feature_name(f) for f in top_features[-10:]], rotation=45)
        ax4.set_yticklabels([visualizer._format_feature_name(f) for f in top_features[-10:]])
        ax4.set_title('Feature Correlations')
        plt.colorbar(im, ax=ax4)
    
    # Model prediction scatter
    ax5 = plt.subplot(2, 3, 5)
    prediction_sum = np.sum(shap_values, axis=1)
    ax5.scatter(range(len(prediction_sum)), prediction_sum, alpha=0.6, c='green')
    ax5.set_title('Prediction Values by Compound')
    ax5.set_xlabel('Compound Index')
    ax5.set_ylabel('Prediction Score')
    
    # Feature group impact comparison
    ax6 = plt.subplot(2, 3, 6)
    group_impacts = {}
    for group_name, group_features in feature_groups.items():
        group_indices = [i for i, name in enumerate(feature_names) if name in group_features]
        if group_indices:
            group_impacts[group_name] = np.mean(np.abs(shap_values[:, group_indices]))
    
    if group_impacts:
        groups = list(group_impacts.keys())
        impacts = list(group_impacts.values())
        colors = [visualizer.color_scheme.get(g, '#666666') for g in groups]
        ax6.bar(groups, impacts, color=colors, alpha=0.7)
        ax6.set_title('Feature Group Impact')
        ax6.set_ylabel('Mean |SHAP Value|')
        ax6.tick_params(axis='x', rotation=45)
    
    plt.suptitle('BioPath SHAP Analysis Dashboard', fontsize=24, fontweight='bold')
    plt.tight_layout()
    
    return fig
