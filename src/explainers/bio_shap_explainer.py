"""
BioPath Custom SHAP Explainer for Natural Compound Bioactivity

This module provides domain-specific SHAP explanations for molecular bioactivity predictions,
with special focus on traditional medicine validation and regulatory compliance.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

# Configure plotting style
plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class BiologicalInterpretation:
    """
    Data class for storing biological interpretations of molecular features.
    """
    feature_name: str
    shap_value: float
    contribution_strength: str
    biological_meaning: str
    traditional_knowledge_link: Optional[str] = None
    regulatory_relevance: Optional[str] = None


class BioPathSHAPExplainer:
    """
    Custom SHAP explainer for natural compound bioactivity prediction with
    domain-specific interpretations and traditional knowledge integration.
    """
    
    def __init__(self, 
                 model: BaseEstimator,
                 feature_names: List[str],
                 feature_groups: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the BioPath SHAP explainer.
        
        Args:
            model: Trained machine learning model
            feature_names: List of feature names
            feature_groups: Dictionary grouping features by chemical relevance
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_groups = feature_groups or {}
        self.explainer = None
        self.base_value = None
        
        # Initialize biological interpretation mappings
        self._init_biological_interpretations()
        
        logging.info(f"BioPathSHAPExplainer initialized with {len(feature_names)} features")
    
    def _init_biological_interpretations(self):
        """Initialize mappings between molecular features and biological meanings."""
        self.biological_meanings = {
            # Basic molecular properties
            'molecular_weight': {
                'positive': 'Higher molecular weight may indicate complex natural products with enhanced bioactivity',
                'negative': 'Lower molecular weight suggests better membrane permeability and bioavailability',
                'traditional_link': 'Large molecules often found in potent plant alkaloids and glycosides'
            },
            'logp': {
                'positive': 'Increased lipophilicity enhances membrane penetration and cellular uptake',
                'negative': 'Reduced lipophilicity improves water solubility and reduces toxicity risk',
                'traditional_link': 'Lipophilic compounds extracted using traditional oil-based preparations'
            },
            'hbd_count': {
                'positive': 'More hydrogen bond donors increase binding affinity to protein targets',
                'negative': 'Fewer hydrogen bond donors improve oral bioavailability',
                'traditional_link': 'Phenolic compounds in medicinal teas provide multiple hydrogen bond donors'
            },
            'phenol_groups': {
                'positive': 'Phenolic groups contribute to antioxidant and anti-inflammatory activities',
                'negative': 'Excess phenolic groups may cause protein precipitation and reduced efficacy',
                'traditional_link': 'Phenolic compounds are key bioactive constituents in traditional herbal medicines'
            },
            'chiral_centers': {
                'positive': 'Stereochemical complexity indicates natural product origin with specific bioactivity',
                'negative': 'Reduced chirality simplifies synthesis and regulatory approval pathway',
                'traditional_link': 'Complex stereochemistry reflects evolutionary optimization in medicinal plants'
            },
            'qed_score': {
                'positive': 'Higher drug-likeness score indicates favorable pharmacological properties',
                'negative': 'Lower QED suggests potential toxicity or poor ADMET characteristics',
                'traditional_link': 'Traditional medicines often contain compounds optimized for oral bioavailability'
            },
            'bertz_complexity': {
                'positive': 'Structural complexity correlates with biological specificity and potency',
                'negative': 'Lower complexity facilitates synthesis and quality control',
                'traditional_link': 'Complex natural products evolved for specific ecological interactions'
            },
            'tpsa': {
                'positive': 'Optimal polar surface area enhances target selectivity',
                'negative': 'Reduced TPSA improves blood-brain barrier penetration',
                'traditional_link': 'Traditional CNS-active plants often contain compounds with low TPSA'
            }
        }
        
        # Regulatory interpretation mappings
        self.regulatory_interpretations = {
            'molecular_weight': 'Molecular weight within Lipinski Rule of Five range (≤500 Da) favors oral bioavailability',
            'logp': 'LogP between 1-3 optimal for drug development (Lipinski compliance)',
            'hbd_count': 'Hydrogen bond donors ≤5 required for Lipinski Rule compliance',
            'hba_count': 'Hydrogen bond acceptors ≤10 required for drug-like properties',
            'rotatable_bonds': 'Rotatable bonds ≤10 important for conformational flexibility and binding',
            'tpsa': 'Topological PSA ≤140 Ų crucial for membrane permeability',
            'qed_score': 'QED score >0.5 indicates favorable drug-like characteristics'
        }
    
    def fit(self, X: np.ndarray, sample_size: int = 100):
        """
        Fit the SHAP explainer to the training data.
        
        Args:
            X: Training feature matrix
            sample_size: Number of background samples for SHAP calculation
        """
        try:
            # Use TreeExplainer for tree-based models, else use KernelExplainer
            if hasattr(self.model, 'predict_proba'):
                # For classification models
                if hasattr(self.model, 'tree_') or 'forest' in str(type(self.model)).lower():
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    # Sample background data for KernelExplainer
                    background_data = shap.sample(X, sample_size)
                    self.explainer = shap.KernelExplainer(self.model.predict_proba, background_data)
            else:
                # For regression models
                if hasattr(self.model, 'tree_') or 'forest' in str(type(self.model)).lower():
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    background_data = shap.sample(X, sample_size)
                    self.explainer = shap.KernelExplainer(self.model.predict, background_data)
            
            # Calculate base value
            self.base_value = self.explainer.expected_value
            if isinstance(self.base_value, np.ndarray):
                self.base_value = self.base_value[1]  # For binary classification
            
            logging.info("SHAP explainer fitted successfully")
            
        except Exception as e:
            logging.error(f"Error fitting SHAP explainer: {e}")
            raise
    
    def explain_instance(self, X_instance: np.ndarray, smiles: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a single compound.
        
        Args:
            X_instance: Feature vector for single instance
            smiles: SMILES string of the compound (optional)
            
        Returns:
            Dictionary containing SHAP values and biological interpretations
        """
        if self.explainer is None:
            raise ValueError("Explainer must be fitted before generating explanations")
        
        try:
            # Calculate SHAP values
            if len(X_instance.shape) == 1:
                X_instance = X_instance.reshape(1, -1)
            
            shap_values = self.explainer.shap_values(X_instance)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary classification
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # Take first instance
            
            # Generate biological interpretations
            interpretations = self._generate_biological_interpretations(
                shap_values, X_instance[0], smiles
            )
            
            # Calculate prediction and confidence
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(X_instance)[0]
                prediction = np.argmax(prediction_proba)
                confidence = np.max(prediction_proba)
            else:
                prediction = self.model.predict(X_instance)[0]
                confidence = None
            
            return {
                'shap_values': shap_values,
                'base_value': self.base_value,
                'feature_values': X_instance[0],
                'feature_names': self.feature_names,
                'prediction': prediction,
                'confidence': confidence,
                'biological_interpretations': interpretations,
                'smiles': smiles
            }
            
        except Exception as e:
            logging.error(f"Error explaining instance: {e}")
            raise
    
    def _generate_biological_interpretations(self, 
                                           shap_values: np.ndarray, 
                                           feature_values: np.ndarray,
                                           smiles: Optional[str] = None) -> List[BiologicalInterpretation]:
        """
        Generate biological interpretations for SHAP values.
        
        Args:
            shap_values: SHAP values for the instance
            feature_values: Feature values for the instance
            smiles: SMILES string (optional)
            
        Returns:
            List of biological interpretations
        """
        interpretations = []
        
        # Get top contributing features
        feature_importance = list(zip(self.feature_names, shap_values, feature_values))
        feature_importance.sort(key=lambda x: abs(np.asarray(x[1]).flatten()[0]), reverse=True)
        
        # Generate interpretations for top 10 features
        for feature_name, shap_val, feature_val in feature_importance[:10]:
            if abs(shap_val) < 0.001:  # Skip very small contributions
                continue
            
            # Determine contribution strength
            abs_shap = abs(shap_val)
            if abs_shap > 0.1:
                strength = "Strong"
            elif abs_shap > 0.05:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            # Get biological meaning
            meaning_data = self.biological_meanings.get(feature_name, {})
            if shap_val > 0:
                biological_meaning = meaning_data.get('positive', 
                    f'Higher {feature_name} contributes positively to bioactivity prediction')
            else:
                biological_meaning = meaning_data.get('negative',
                    f'Lower {feature_name} contributes negatively to bioactivity prediction')
            
            # Traditional knowledge link
            traditional_link = meaning_data.get('traditional_link')
            
            # Regulatory relevance
            regulatory_relevance = self.regulatory_interpretations.get(feature_name)
            
            interpretation = BiologicalInterpretation(
                feature_name=feature_name,
                shap_value=shap_val,
                contribution_strength=strength,
                biological_meaning=biological_meaning,
                traditional_knowledge_link=traditional_link,
                regulatory_relevance=regulatory_relevance
            )
            
            interpretations.append(interpretation)
        
        return interpretations
    
    def plot_feature_importance(self, 
                               shap_values: np.ndarray, 
                               feature_names: Optional[List[str]] = None,
                               max_features: int = 15,
                               title: str = "Molecular Feature Importance") -> plt.Figure:
        """
        Create a customized feature importance plot.
        
        Args:
            shap_values: SHAP values matrix
            feature_names: Feature names (optional)
            max_features: Maximum number of features to display
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        feature_names = feature_names or self.feature_names
        
        # Calculate mean absolute SHAP values
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Get top features
        top_indices = np.argsort(mean_shap)[-max_features:]
        top_features = [feature_names[i] for i in top_indices]
        top_values = mean_shap[top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color mapping based on feature groups
        colors = self._get_feature_colors(top_features)
        
        bars = ax.barh(range(len(top_features)), top_values, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels([self._format_feature_name(name) for name in top_features])
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_values)):
            ax.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', va='center', fontsize=10)
        
        # Add legend for feature groups
        self._add_feature_group_legend(ax, top_features)
        
        plt.tight_layout()
        return fig
    
    def _get_feature_colors(self, feature_names: List[str]) -> List[str]:
        """Get colors for features based on their groups."""
        color_map = {
            'Basic Properties': '#1f77b4',
            'Drug-likeness': '#ff7f0e', 
            'Structural Complexity': '#2ca02c',
            'Natural Product Features': '#d62728',
            'Functional Groups': '#9467bd',
            'Pharmacophores': '#8c564b',
            'Fingerprint Features': '#e377c2'
        }
        
        colors = []
        for feature in feature_names:
            # Find which group this feature belongs to
            feature_color = '#7f7f7f'  # Default gray
            for group_name, group_features in self.feature_groups.items():
                if feature in group_features:
                    feature_color = color_map.get(group_name, '#7f7f7f')
                    break
            colors.append(feature_color)
        
        return colors
    
    def _format_feature_name(self, name: str) -> str:
        """Format feature names for better readability."""
        # Replace underscores with spaces and capitalize
        formatted = name.replace('_', ' ').title()
        
        # Handle special cases
        replacements = {
            'Logp': 'LogP',
            'Tpsa': 'TPSA', 
            'Qed': 'QED',
            'Hbd': 'HBD',
            'Hba': 'HBA'
        }
        
        for old, new in replacements.items():
            formatted = formatted.replace(old, new)
        
        return formatted
    
    def _add_feature_group_legend(self, ax: plt.Axes, feature_names: List[str]):
        """Add legend showing feature groups."""
        from matplotlib.patches import Patch
        
        # Get unique groups represented in the plot
        groups_in_plot = set()
        for feature in feature_names:
            for group_name, group_features in self.feature_groups.items():
                if feature in group_features:
                    groups_in_plot.add(group_name)
                    break
        
        if len(groups_in_plot) > 1:
            color_map = {
                'Basic Properties': '#1f77b4',
                'Drug-likeness': '#ff7f0e',
                'Structural Complexity': '#2ca02c', 
                'Natural Product Features': '#d62728',
                'Functional Groups': '#9467bd',
                'Pharmacophores': '#8c564b',
                'Fingerprint Features': '#e377c2'
            }
            
            legend_elements = [Patch(facecolor=color_map.get(group, '#7f7f7f'), 
                                   label=group) for group in groups_in_plot]
            
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    def generate_report(self, explanations: List[Dict[str, Any]], 
                       output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive interpretation report.
        
        Args:
            explanations: List of explanation dictionaries
            output_file: Optional file path to save the report
            
        Returns:
            Report as a string
        """
        report_lines = []
        report_lines.append("# BioPath Molecular Bioactivity Prediction Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Summary statistics
        predictions = [exp['prediction'] for exp in explanations]
        confidences = [exp['confidence'] for exp in explanations if exp['confidence'] is not None]
        
        report_lines.append("## Summary")
        report_lines.append(f"Total compounds analyzed: {len(explanations)}")
        if confidences:
            report_lines.append(f"Average prediction confidence: {np.mean(confidences):.3f}")
        report_lines.append("")
        
        # Individual compound analyses
        for i, exp in enumerate(explanations):
            report_lines.append(f"## Compound {i+1}")
            if exp['smiles']:
                report_lines.append(f"SMILES: {exp['smiles']}")
            report_lines.append(f"Prediction: {exp['prediction']}")
            if exp['confidence']:
                report_lines.append(f"Confidence: {exp['confidence']:.3f}")
            report_lines.append("")
            
            # Top biological interpretations
            report_lines.append("### Key Molecular Factors:")
            for interp in exp['biological_interpretations'][:5]:
                report_lines.append(f"- **{interp.feature_name}** ({interp.contribution_strength} impact)")
                report_lines.append(f"  {interp.biological_meaning}")
                if interp.traditional_knowledge_link:
                    report_lines.append(f"  *Traditional context: {interp.traditional_knowledge_link}*")
                if interp.regulatory_relevance:
                    report_lines.append(f"  *Regulatory note: {interp.regulatory_relevance}*")
                report_lines.append("")
            
            report_lines.append("---")
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logging.info(f"Report saved to {output_file}")
        
        return report


def create_molecular_waterfall_plot(explanation: Dict[str, Any], 
                                   max_features: int = 10,
                                   title: str = "Bioactivity Prediction Breakdown") -> plt.Figure:
    """
    Create a waterfall plot showing how features contribute to the prediction.
    
    Args:
        explanation: Explanation dictionary from explain_instance
        max_features: Maximum number of features to show
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    shap_values = explanation['shap_values']
    feature_names = explanation['feature_names'] 
    base_value = explanation['base_value']
    
    # Get top contributing features
    feature_importance = list(zip(feature_names, shap_values))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Prepare data for waterfall plot
    top_features = feature_importance[:max_features]
    names = [name.replace('_', ' ').title() for name, _ in top_features]
    values = [val for _, val in top_features]
    
    # Create cumulative values for waterfall effect
    cumulative = [base_value]
    for val in values:
        cumulative.append(cumulative[-1] + val)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars
    colors = ['green' if val > 0 else 'red' for val in values]
    x_pos = range(len(names))
    
    for i, (name, val, cum) in enumerate(zip(names, values, cumulative[1:])):
        bottom = cumulative[i] if val > 0 else cum
        height = abs(val)
        ax.bar(i, height, bottom=bottom, color=colors[i], alpha=0.7, 
               label=f'{name}: {val:+.3f}')
    
    # Add base value bar
    ax.axhline(y=base_value, color='black', linestyle='--', alpha=0.5, 
               label=f'Base value: {base_value:.3f}')
    
    # Final prediction line
    final_pred = cumulative[-1]
    ax.axhline(y=final_pred, color='blue', linestyle='-', linewidth=2,
               label=f'Final prediction: {final_pred:.3f}')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Contribution to Prediction')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig
