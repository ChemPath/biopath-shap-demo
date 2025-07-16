"""
BioPath SHAP Demo - Advanced SHAP Visualization

Comprehensive visualization toolkit for SHAP analysis of natural compound
bioactivity predictions with enhanced biological context and traditional
knowledge integration.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)

class SHAPVisualization:
    """
    Advanced SHAP visualization toolkit with biological interpretations
    and traditional knowledge integration for natural compound analysis.
    """
    
    def __init__(self, 
                 feature_groups: Optional[Dict[str, List[str]]] = None,
                 style: str = 'professional',
                 color_palette: Optional[Dict[str, str]] = None):
        """
        Initialize SHAP visualization toolkit.
        
        Args:
            feature_groups: Dictionary grouping features by chemical relevance
            style: Visualization style ('professional', 'scientific', 'presentation')
            color_palette: Custom color palette for visualizations
        """
        self.feature_groups = feature_groups or {}
        self.style = style
        self.color_palette = color_palette or self._get_default_palette()
        
        # Configure matplotlib style
        self._configure_plot_style()
        
        logging.info(f"SHAPVisualization initialized with {style} style")
    
    def _get_default_palette(self) -> Dict[str, str]:
        """Get default color palette based on style."""
        palettes = {
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
        
        return palettes.get(self.style, palettes['professional'])
    
    def _configure_plot_style(self):
        """Configure matplotlib style for consistent plotting."""
        try:
            if self.style == 'professional':
                plt.style.use('seaborn-v0_8-whitegrid')
                sns.set_palette("husl")
            elif self.style == 'scientific':
                plt.style.use('seaborn-v0_8-paper')
                sns.set_palette("deep")
            elif self.style == 'presentation':
                plt.style.use('seaborn-v0_8-talk')
                sns.set_palette("bright")
            
            # Set consistent parameters
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.titlesize'] = 14
            plt.rcParams['axes.labelsize'] = 12
            plt.rcParams['xtick.labelsize'] = 10
            plt.rcParams['ytick.labelsize'] = 10
            
        except Exception as e:
            logging.warning(f"Error configuring plot style: {e}")
    
    def create_feature_importance_summary(self,
                                        shap_values: np.ndarray,
                                        feature_names: List[str],
                                        max_features: int = 20,
                                        title: str = "SHAP Feature Importance Summary") -> plt.Figure:
        """
        Create comprehensive feature importance summary plot.
        
        Args:
            shap_values: SHAP values matrix (samples x features)
            feature_names: List of feature names
            max_features: Maximum number of features to display
            title: Plot title
            
        Returns:
            matplotlib.Figure: Feature importance plot
        """
        try:
            # Calculate feature importance
            importance = np.mean(np.abs(shap_values), axis=0)
            
            # Get top features
            top_indices = np.argsort(importance)[-max_features:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = importance[top_indices]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, max(8, len(top_features) * 0.4)))
            
            # Format feature names
            formatted_features = [f.replace('_', ' ').title() for f in top_features]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(top_features))
            bars = ax.barh(y_pos, top_importance, 
                          color=self.color_palette['positive'], 
                          alpha=0.8, edgecolor='navy', linewidth=0.5)
            
            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(formatted_features)
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, top_importance)):
                ax.text(value + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', va='center', fontsize=10)
            
            # Add feature group colors if available
            if self.feature_groups:
                self._add_feature_group_colors(ax, top_features, y_pos)
            
            plt.tight_layout()
            logging.info(f"Created feature importance summary with {len(top_features)} features")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating feature importance summary: {e}")
            raise
    
    def create_shap_beeswarm_plot(self,
                                 shap_values: np.ndarray,
                                 feature_values: np.ndarray,
                                 feature_names: List[str],
                                 max_features: int = 15,
                                 title: str = "SHAP Beeswarm Plot") -> plt.Figure:
        """
        Create SHAP beeswarm plot showing feature value distributions.
        
        Args:
            shap_values: SHAP values matrix (samples x features)
            feature_values: Feature values matrix (samples x features)
            feature_names: List of feature names
            max_features: Maximum number of features to display
            title: Plot title
            
        Returns:
            matplotlib.Figure: Beeswarm plot
        """
        try:
            # Calculate feature importance and select top features
            importance = np.mean(np.abs(shap_values), axis=0)
            top_indices = np.argsort(importance)[-max_features:][::-1]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, max(8, len(top_indices) * 0.4)))
            
            # Plot each feature
            for i, idx in enumerate(top_indices):
                feature_name = feature_names[idx]
                shap_vals = shap_values[:, idx]
                feat_vals = feature_values[:, idx]
                
                # Normalize feature values for color mapping
                if np.max(feat_vals) > np.min(feat_vals):
                    norm_vals = (feat_vals - np.min(feat_vals)) / (np.max(feat_vals) - np.min(feat_vals))
                else:
                    norm_vals = np.zeros_like(feat_vals)
                
                # Create scatter plot with jitter
                jitter = np.random.normal(0, 0.1, len(shap_vals))
                y_pos = np.full_like(shap_vals, i) + jitter
                
                scatter = ax.scatter(shap_vals, y_pos, c=norm_vals, 
                                   cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            # Customize plot
            ax.set_yticks(range(len(top_indices)))
            ax.set_yticklabels([feature_names[i].replace('_', ' ').title() for i in top_indices])
            ax.set_xlabel('SHAP Value')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Feature Value (normalized)', rotation=270, labelpad=15)
            
            plt.tight_layout()
            logging.info(f"Created beeswarm plot with {len(top_indices)} features")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating beeswarm plot: {e}")
            raise
    
    def create_feature_group_comparison(self,
                                      shap_values: np.ndarray,
                                      feature_names: List[str],
                                      title: str = "Feature Group Comparison") -> plt.Figure:
        """
        Create comparison of SHAP contributions across feature groups.
        
        Args:
            shap_values: SHAP values matrix (samples x features)
            feature_names: List of feature names
            title: Plot title
            
        Returns:
            matplotlib.Figure: Feature group comparison plot
        """
        try:
            if not self.feature_groups:
                logging.warning("No feature groups defined for comparison")
                return self.create_feature_importance_summary(shap_values, feature_names)
            
            # Calculate group contributions
            group_contributions = {}
            
            for group_name, group_features in self.feature_groups.items():
                group_indices = []
                for feature in group_features:
                    if feature in feature_names:
                        group_indices.append(feature_names.index(feature))
                
                if group_indices:
                    group_shap = shap_values[:, group_indices]
                    group_contributions[group_name] = np.mean(np.abs(group_shap))
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot 1: Group importance bar chart
            groups = list(group_contributions.keys())
            importances = list(group_contributions.values())
            
            bars = ax1.bar(groups, importances, 
                          color=self.color_palette['positive'], 
                          alpha=0.8, edgecolor='navy')
            
            ax1.set_xlabel('Feature Groups')
            ax1.set_ylabel('Mean |SHAP Value|')
            ax1.set_title('Feature Group Importance', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, importances):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # Plot 2: Detailed group breakdown
            all_group_shap = []
            all_group_labels = []
            
            for group_name, group_features in self.feature_groups.items():
                group_indices = []
                for feature in group_features:
                    if feature in feature_names:
                        group_indices.append(feature_names.index(feature))
                
                if group_indices:
                    group_shap = shap_values[:, group_indices]
                    feature_importance = np.mean(np.abs(group_shap), axis=0)
                    
                    all_group_shap.extend(feature_importance)
                    all_group_labels.extend([group_name] * len(feature_importance))
            
            # Create box plot
            if all_group_shap:
                df = pd.DataFrame({
                    'importance': all_group_shap,
                    'group': all_group_labels
                })
                
                sns.boxplot(data=df, x='group', y='importance', ax=ax2)
                ax2.set_xlabel('Feature Groups')
                ax2.set_ylabel('Feature Importance Distribution')
                ax2.set_title('Feature Importance Distribution by Group', fontweight='bold')
                ax2.tick_params(axis='x', rotation=45)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            logging.info(f"Created feature group comparison with {len(groups)} groups")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating feature group comparison: {e}")
            raise
    
    def create_molecular_heatmap(self,
                               shap_values: np.ndarray,
                               feature_names: List[str],
                               compound_names: List[str],
                               max_features: int = 20,
                               title: str = "Molecular SHAP Heatmap") -> plt.Figure:
        """
        Create heatmap of SHAP values across compounds and features.
        
        Args:
            shap_values: SHAP values matrix (samples x features)
            feature_names: List of feature names
            compound_names: List of compound names
            max_features: Maximum number of features to display
            title: Plot title
            
        Returns:
            matplotlib.Figure: Molecular heatmap
        """
        try:
            # Select top features by importance
            importance = np.mean(np.abs(shap_values), axis=0)
            top_indices = np.argsort(importance)[-max_features:][::-1]
            
            # Create data for heatmap
            heatmap_data = shap_values[:, top_indices]
            top_features = [feature_names[i] for i in top_indices]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(max(12, len(top_features) * 0.5), 
                                          max(8, len(compound_names) * 0.3)))
            
            # Create heatmap
            im = ax.imshow(heatmap_data.T, cmap='RdBu_r', aspect='auto')
            
            # Customize plot
            ax.set_xticks(range(len(compound_names)))
            ax.set_xticklabels(compound_names, rotation=45, ha='right')
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels([f.replace('_', ' ').title() for f in top_features])
            ax.set_xlabel('Compounds')
            ax.set_ylabel('Molecular Features')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('SHAP Value', rotation=270, labelpad=15)
            
            # Add text annotations for significant values
            for i in range(len(top_features)):
                for j in range(len(compound_names)):
                    value = heatmap_data[j, i]
                    if abs(value) > 0.1:  # Only annotate significant values
                        ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                               color='white' if abs(value) > 0.2 else 'black',
                               fontsize=8)
            
            plt.tight_layout()
            logging.info(f"Created molecular heatmap with {len(top_features)} features and {len(compound_names)} compounds")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating molecular heatmap: {e}")
            raise
    
    def create_traditional_knowledge_analysis(self,
                                            shap_values: np.ndarray,
                                            feature_names: List[str],
                                            title: str = "Traditional Knowledge Feature Analysis") -> plt.Figure:
        """
        Create visualization highlighting traditional knowledge-related features.
        
        Args:
            shap_values: SHAP values matrix (samples x features)
            feature_names: List of feature names
            title: Plot title
            
        Returns:
            matplotlib.Figure: Traditional knowledge analysis plot
        """
        try:
            # Identify traditional knowledge features
            tk_features = {
                'Phenolic Compounds': [],
                'Alkaloids': [],
                'Terpenoids': [],
                'Flavonoids': [],
                'Functional Groups': []
            }
            
            # Feature patterns
            patterns = {
                'Phenolic Compounds': ['phenol', 'hydroxyl', 'caffeic', 'gallic'],
                'Alkaloids': ['nitrogen', 'caffeine', 'berberine'],
                'Terpenoids': ['limonene', 'menthol', 'steroid'],
                'Flavonoids': ['quercetin', 'kaempferol', 'catechin'],
                'Functional Groups': ['carbonyl', 'ether', 'ester']
            }
            
            for feature in feature_names:
                feature_lower = feature.lower()
                categorized = False
                
                for category, pattern_list in patterns.items():
                    if any(pattern in feature_lower for pattern in pattern_list):
                        tk_features[category].append(feature)
                        categorized = True
                        break
                
                if not categorized and any(term in feature_lower for term in ['traditional', 'ethnobotanical']):
                    tk_features['Functional Groups'].append(feature)
            
            # Calculate importance for each category
            category_importance = {}
            for category, features in tk_features.items():
                if features:
                    indices = [feature_names.index(f) for f in features if f in feature_names]
                    if indices:
                        category_shap = shap_values[:, indices]
                        category_importance[category] = np.mean(np.abs(category_shap))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if category_importance:
                categories = list(category_importance.keys())
                importances = list(category_importance.values())
                
                # Create bar plot with traditional knowledge colors
                colors = ['#8B4513', '#4B0082', '#228B22', '#FF6347', '#DAA520']
                bars = ax.bar(categories, importances, 
                             color=colors[:len(categories)], 
                             alpha=0.8, edgecolor='black')
                
                # Customize plot
                ax.set_xlabel('Traditional Knowledge Categories')
                ax.set_ylabel('Mean |SHAP Value|')
                ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, importances):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.3f}', ha='center', va='bottom')
                
                # Add grid
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No traditional knowledge features identified', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, style='italic')
            
            plt.tight_layout()
            logging.info(f"Created traditional knowledge analysis plot")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating traditional knowledge analysis: {e}")
            raise
    
    def _add_feature_group_colors(self, ax, features: List[str], y_positions: np.ndarray):
        """Add colored bars to indicate feature groups."""
        try:
            if not self.feature_groups:
                return
            
            # Group colors
            group_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
            
            for i, feature in enumerate(features):
                for j, (group_name, group_features) in enumerate(self.feature_groups.items()):
                    if feature in group_features:
                        # Add colored rectangle
                        rect = mpatches.Rectangle(
                            (-0.01, y_positions[i] - 0.4), 0.005, 0.8,
                            color=group_colors[j % len(group_colors)],
                            alpha=0.7
                        )
                        ax.add_patch(rect)
                        break
            
            # Add legend
            legend_elements = []
            for i, group_name in enumerate(self.feature_groups.keys()):
                legend_elements.append(
                    mpatches.Patch(color=group_colors[i % len(group_colors)], 
                                 label=group_name)
                )
            
            ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
            
        except Exception as e:
            logging.warning(f"Error adding feature group colors: {e}")
    
    def save_publication_ready_figure(self, 
                                     fig: plt.Figure,
                                     filename: str,
                                     dpi: int = 300,
                                     formats: List[str] = ['png', 'pdf']):
        """
        Save figure in publication-ready formats.
        
        Args:
            fig: matplotlib Figure object
            filename: Base filename (without extension)
            dpi: Resolution for raster formats
            formats: List of formats to save
        """
        try:
            for fmt in formats:
                full_filename = f"{filename}.{fmt}"
                fig.savefig(full_filename, dpi=dpi, bbox_inches='tight', 
                           format=fmt, facecolor='white', edgecolor='none')
                logging.info(f"Saved figure as {full_filename}")
                
        except Exception as e:
            logging.error(f"Error saving figure: {e}")

