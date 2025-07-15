#!/usr/bin/env python3
"""
Modern BioPath SHAP Demo Script
Demonstrates explainable AI for natural compound bioactivity prediction
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_preprocessing.molecular_features import ModernMolecularFeatureCalculator
from explainers.bio_shap_explainer import ModernBioPathSHAPExplainer
from visualization.shap_plots import create_modern_shap_plots

def main():
    """Run the modern BioPath SHAP demonstration."""
    
    print("🧬 BioPath SHAP Demo - Modern Implementation")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample natural product SMILES for demonstration
    sample_compounds = [
        'c1cc(ccc1c2cc(=O)c3c(cc(cc3o2)O)O)O',  # Quercetin
        'CN1CCc2cc3c(cc2C1)OCO3',  # Berberine
        'COc1cc(cc(c1O)OC)c2cc(=O)c3c(o2)cc(cc3O)O',  # Chrysin
        'c1cc(c(cc1CC(C(=O)O)N)O)O',  # L-DOPA
        'c1cc(ccc1C=CC(=O)O)O',  # p-Coumaric acid
    ]
    
    # Generate synthetic bioactivity labels
    np.random.seed(42)
    bioactivity_labels = np.random.choice([0, 1], size=len(sample_compounds), p=[0.4, 0.6])
    
    print(f"📊 Processing {len(sample_compounds)} natural compounds...")
    
    # Calculate molecular features
    feature_calculator = ModernMolecularFeatureCalculator(
        include_fingerprints=True,
        fingerprint_radius=2
    )
    
    features_df = feature_calculator.process_batch(sample_compounds)
    
    if features_df.empty:
        print("❌ No valid features calculated")
        return
    
    print(f"✅ Calculated {len(features_df.columns)-1} molecular features")
    
    # Prepare data for machine learning
    feature_columns = [col for col in features_df.columns if col != 'smiles']
    X = features_df[feature_columns].values
    y = bioactivity_labels[:len(X)]
    
    # Train model
    print("🤖 Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Setup SHAP explainer
    print("🔍 Setting up SHAP explainer...")
    feature_groups = feature_calculator.get_feature_groups()
    
    explainer = ModernBioPathSHAPExplainer(
        model=model,
        feature_names=feature_columns,
        feature_groups=feature_groups
    )
    
    explainer.fit(X, sample_size=len(X))
    
    # Generate explanations
    print("💡 Generating SHAP explanations...")
    explanations = []
    
    for i in range(len(X)):
        explanation = explainer.explain_instance(
            X[i], 
            compound_id=f"Compound_{i+1}"
        )
        explanations.append(explanation)
    
    # Generate report
    print("📄 Generating analysis report...")
    report = explainer.generate_summary_report(
        explanations, 
        output_file="biopath_shap_report.md"
    )
    
    print("✅ Demo completed successfully!")
    print(f"📁 Report saved to: biopath_shap_report.md")
    print("\n" + "="*50)
    print("🚀 BioPath SHAP Demo showcases:")
    print("- Modern molecular feature calculation")
    print("- Explainable AI for bioactivity prediction")
    print("- Biological interpretation of SHAP values")
    print("- Integration with traditional knowledge")
    print("- Regulatory-compliant explanations")

if __name__ == "__main__":
    main()

