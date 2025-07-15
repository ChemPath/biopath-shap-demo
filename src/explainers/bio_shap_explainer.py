"""
Modern BioPath SHAP Explainer with Enhanced Interpretability
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class BiologicalInterpretation:
    """Enhanced biological interpretation with confidence scores."""
    feature_name: str
    shap_value: float
    contribution_strength: str
    biological_meaning: str
    confidence_score: float
    traditional_knowledge_link: Optional[str] = None
    regulatory_relevance: Optional[str] = None

class ModernBioPathSHAPExplainer:
    """
    Modern SHAP explainer with enhanced biological interpretations
    and improved stability for molecular bioactivity prediction.
    """
    
    def __init__(self, 
                 model: BaseEstimator,
                 feature_names: List[str],
                 feature_groups: Optional[Dict[str, List[str]]] = None):
        """
        Initialize modern SHAP explainer with enhanced capabilities.
        
        Args:
            model: Trained scikit-learn model
            feature_names: List of molecular feature names
            feature_groups: Dictionary grouping features by chemical relevance
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_groups = feature_groups or {}
        self.explainer = None
        self.base_value = None
        
        # Initialize enhanced biological interpretation database
        self._init_enhanced_biological_meanings()
        
        logging.info(f"ModernBioPathSHAPExplainer initialized with {len(feature_names)} features")
    
    def _init_enhanced_biological_meanings(self):
        """Initialize comprehensive biological interpretation database."""
        self.biological_meanings = {
            'molecular_weight': {
                'positive': 'Higher molecular weight indicates complex natural products with potential multi-target activity',
                'negative': 'Lower molecular weight favors better bioavailability and cellular penetration',
                'confidence': 0.85,
                'traditional_link': 'Heavy molecules often correspond to potent alkaloids in traditional preparations'
            },
            'logp': {
                'positive': 'Increased lipophilicity enhances membrane penetration and protein binding',
                'negative': 'Lower lipophilicity improves water solubility and reduces toxicity potential',
                'confidence': 0.92,
                'traditional_link': 'Lipophilic compounds extracted through traditional oil-based methods'
            },
            'phenol_groups': {
                'positive': 'Multiple phenolic groups strongly correlate with antioxidant and anti-inflammatory activity',
                'negative': 'Excessive phenolic content may cause protein precipitation and reduced bioavailability',
                'confidence': 0.94,
                'traditional_link': 'Phenolic compounds are primary bioactive constituents in medicinal teas and tinctures'
            },
            'qed_score': {
                'positive': 'High drug-likeness score indicates favorable pharmacokinetic properties',
                'negative': 'Low QED suggests potential ADMET liabilities requiring optimization',
                'confidence': 0.88,
                'traditional_link': 'Traditional medicines often contain naturally drug-like compounds'
            }
        }
    
    def fit(self, X: np.ndarray, sample_size: int = 100):
        """
        Fit SHAP explainer with automatic model type detection.
        
        Args:
            X: Training feature matrix
            sample_size: Background sample size for KernelExplainer
        """
        try:
            # Automatic explainer selection based on model type
            model_type = str(type(self.model)).lower()
            
            if any(tree_type in model_type for tree_type in ['forest', 'tree', 'gradient', 'xgb', 'lgb']):
                # Use TreeExplainer for tree-based models
                self.explainer = shap.TreeExplainer(self.model)
                logging.info("Using TreeExplainer for tree-based model")
            else:
                # Use KernelExplainer for other models
                background_data = shap.sample(X, min(sample_size, len(X)))
                
                if hasattr(self.model, 'predict_proba'):
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba, 
                        background_data
                    )
                    logging.info("Using KernelExplainer with predict_proba")
                else:
                    self.explainer = shap.KernelExplainer(
                        self.model.predict, 
                        background_data
                    )
                    logging.info("Using KernelExplainer with predict")
            
            # Calculate base value
            self.base_value = self.explainer.expected_value
            if isinstance(self.base_value, np.ndarray):
                self.base_value = self.base_value[1]  # Binary classification
            
            logging.info("SHAP explainer fitted successfully")
            
        except Exception as e:
            logging.error(f"Error fitting SHAP explainer: {e}")
            raise
    
    def explain_instance(self, 
                        X_instance: np.ndarray, 
                        compound_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a single compound.
        
        Args:
            X_instance: Feature vector for single compound
            compound_id: Optional compound identifier
            
        Returns:
            Dictionary with SHAP values and biological interpretations
        """
        if self.explainer is None:
            raise ValueError("Explainer must be fitted before generating explanations")
        
        try:
            # Ensure correct shape
            if len(X_instance.shape) == 1:
                X_instance = X_instance.reshape(1, -1)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_instance)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class for binary
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # First instance
            
            # Generate biological interpretations
            interpretations = self._generate_enhanced_interpretations(
                shap_values, X_instance[0], compound_id
            )
            
            # Get model prediction
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
                'compound_id': compound_id
            }
            
        except Exception as e:
            logging.error(f"Error explaining instance: {e}")
            raise
    
    def _generate_enhanced_interpretations(self,
                                         shap_values: np.ndarray,
                                         feature_values: np.ndarray,
                                         compound_id: Optional[str] = None) -> List[BiologicalInterpretation]:
        """Generate enhanced biological interpretations with confidence scores."""
        interpretations = []
        
        # Process features by importance
        feature_importance = []
        for i, (fname, sval, fval) in enumerate(zip(self.feature_names, shap_values, feature_values)):
            try:
                scalar_val = float(np.asarray(sval).flatten()[0])
                feature_importance.append((fname, scalar_val, fval))
            except Exception as e:
                logging.warning(f"Error processing feature {fname}: {e}")
                continue
        
        # Sort by absolute SHAP value
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Generate interpretations for top features
        for feature_name, shap_val, feature_val in feature_importance[:15]:
            if abs(shap_val) < 0.001:  # Skip negligible contributions
                continue
            
            # Determine contribution strength
            abs_shap = abs(shap_val)
            if abs_shap > 0.1:
                strength = "Strong"
            elif abs_shap > 0.05:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            # Get biological meaning with confidence
            meaning_data = self.biological_meanings.get(feature_name, {})
            
            if shap_val > 0:
                biological_meaning = meaning_data.get('positive',
                    f'Higher {feature_name} (value: {feature_val:.3f}) contributes positively to predicted bioactivity')
            else:
                biological_meaning = meaning_data.get('negative',
                    f'Lower {feature_name} (value: {feature_val:.3f}) contributes negatively to predicted bioactivity')
            
            confidence_score = meaning_data.get('confidence', 0.5)
            traditional_link = meaning_data.get('traditional_link')
            
            # Create interpretation object
            interpretation = BiologicalInterpretation(
                feature_name=feature_name,
                shap_value=shap_val,
                contribution_strength=strength,
                biological_meaning=biological_meaning,
                confidence_score=confidence_score,
                traditional_knowledge_link=traditional_link
            )
            
            interpretations.append(interpretation)
        
        return interpretations
    
    def generate_summary_report(self, 
                               explanations: List[Dict[str, Any]],
                               output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive summary report for multiple compounds.
        
        Args:
            explanations: List of explanation dictionaries
            output_file: Optional output file path
            
        Returns:
            Formatted report string
        """
        report_lines = []
        
        # Header
        report_lines.extend([
            "# BioPath SHAP Analysis Report",
            "## Explainable AI for Natural Compound Bioactivity",
            "=" * 60,
            ""
        ])
        
        # Executive summary
        predictions = [exp['prediction'] for exp in explanations]
        confidences = [exp['confidence'] for exp in explanations if exp['confidence'] is not None]
        
        report_lines.extend([
            "## Executive Summary",
            f"- **Compounds Analyzed**: {len(explanations)}",
            f"- **Active Predictions**: {sum(predictions)} ({100*sum(predictions)/len(predictions):.1f}%)",
        ])
        
        if confidences:
            report_lines.append(f"- **Average Confidence**: {np.mean(confidences):.1%}")
        
        report_lines.extend(["", "## Key Findings", ""])
        
        # Aggregate feature importance
        all_features = {}
        for exp in explanations:
            for interp in exp['biological_interpretations']:
                fname = interp.feature_name
                if fname not in all_features:
                    all_features[fname] = []
                all_features[fname].append(abs(interp.shap_value))
        
        # Top features across all compounds
        feature_importance = [(fname, np.mean(values)) for fname, values in all_features.items()]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        report_lines.append("### Most Important Molecular Features:")
        for fname, importance in feature_importance[:10]:
            report_lines.append(f"- **{fname}**: {importance:.4f} average importance")
        
        report_lines.extend(["", "## Individual Compound Analysis", ""])
        
        # Individual compound details
        for i, exp in enumerate(explanations):
            compound_id = exp.get('compound_id', f'Compound_{i+1}')
            report_lines.extend([
                f"### {compound_id}",
                f"- **Prediction**: {'Active' if exp['prediction'] else 'Inactive'}",
            ])
            
            if exp['confidence']:
                report_lines.append(f"- **Confidence**: {exp['confidence']:.1%}")
            
            report_lines.append("- **Key Contributing Features**:")
            for interp in exp['biological_interpretations'][:5]:
                report_lines.append(f"  - {interp.feature_name}: {interp.biological_meaning}")
            
            report_lines.append("")
        
        # Generate report
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logging.info(f"Report saved to {output_file}")
        
        return report

