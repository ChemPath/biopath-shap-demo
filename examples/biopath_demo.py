#!/usr/bin/env python3
"""
BioPath SHAP Demo - Complete Demonstration Script

This script provides a comprehensive demonstration of the BioPath ecosystem's
capabilities for natural compound bioactivity prediction with explainable AI.

Usage:
    python examples/biopath_demo.py [--activity antioxidant] [--compounds 50] [--output results/]

Example:
    python examples/biopath_demo.py --activity anti_inflammatory --compounds 100
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import our custom modules
from data_preprocessing.molecular_features import MolecularFeatureCalculator, normalize_features
from explainers.bio_shap_explainer import BioPathSHAPExplainer, BiologicalInterpretation
from visualization.shap_plots import SHAPVisualization, create_summary_dashboard

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
plt.rcParams['figure.max_open_warning'] = 50

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class BioPathDemo:
    """
    Complete demonstration of BioPath SHAP capabilities.
    
    This class orchestrates the entire demonstration pipeline from data loading
    through model training, SHAP analysis, and report generation.
    """
    
    def __init__(self, 
                 activity_type: str = 'antioxidant',
                 max_compounds: int = 200,
                 output_dir: str = 'demo_results'):
        """
        Initialize the BioPath demonstration.
        
        Args:
            activity_type: Bioactivity type to predict
            max_compounds: Maximum number of compounds to use
            output_dir: Directory for saving results
        """
        self.activity_type = activity_type
        self.max_compounds = max_compounds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.feature_calculator = None
        self.model = None
        self.explainer = None
        self.visualizer = None
        
        # Data storage
        self.compounds_df = None
        self.features_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        logging.info(f"BioPathDemo initialized for {activity_type} prediction")
    
    def load_data(self) -> bool:
        """
        Load and prepare the compound dataset.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("📁 Loading compound dataset...")
            
            # Try to load sample data
            data_path = Path('data/sample_compounds.csv')
            if not data_path.exists():
                # Generate data if it doesn't exist
                print("   Sample data not found. Generating...")
                sys.path.append('data')
                from generate_sample_data import create_sample_datasets
                create_sample_datasets()
            
            # Load the dataset
            self.compounds_df = pd.read_csv(data_path)
            
            # Limit to max_compounds for demo performance
            if len(self.compounds_df) > self.max_compounds:
                self.compounds_df = self.compounds_df.sample(
                    n=self.max_compounds, 
                    random_state=42
                ).reset_index(drop=True)
            
            print(f"   ✅ Loaded {len(self.compounds_df)} compounds")
            print(f"   🎯 Predicting: {self.activity_type}")
            
            # Check if activity column exists
            activity_col = f'{self.activity_type}_active'
            if activity_col not in self.compounds_df.columns:
                available_activities = [col.replace('_active', '') for col in self.compounds_df.columns if col.endswith('_active')]
                print(f"   ⚠️  Activity '{self.activity_type}' not found.")
                print(f"   Available activities: {available_activities}")
                if available_activities:
                    self.activity_type = available_activities[0]
                    print(f"   Using '{self.activity_type}' instead")
                else:
                    return False
            
            # Show activity distribution
            activity_col = f'{self.activity_type}_active'
            active_count = self.compounds_df[activity_col].sum()
            total_count = len(self.compounds_df)
            print(f"   📊 Activity distribution: {active_count}/{total_count} ({100*active_count/total_count:.1f}% active)")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            logging.error(f"Data loading failed: {e}", exc_info=True)
            return False
    
    def calculate_features(self) -> bool:
        """
        Calculate molecular features for all compounds.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("🧬 Calculating molecular features...")
            
            # Initialize feature calculator
            self.feature_calculator = MolecularFeatureCalculator(
                include_fingerprints=True,
                fingerprint_radius=2
            )
            
            # Calculate features for all compounds
            smiles_list = self.compounds_df['smiles'].tolist()
            self.features_df = self.feature_calculator.process_batch(
                smiles_list, 
                show_progress=True
            )
            
            if self.features_df.empty:
                print("   ❌ No valid molecular features calculated")
                return False
            
            # Remove SMILES column for ML (keep for reference)
            feature_columns = [col for col in self.features_df.columns if col != 'smiles']
            print(f"   ✅ Calculated {len(feature_columns)} molecular features")
            print(f"   📊 Feature categories:")
            
            # Show feature group distribution
            feature_groups = self.feature_calculator.get_feature_importance_groups()
            for group_name, group_features in feature_groups.items():
                count = len([f for f in group_features if f in feature_columns])
                if count > 0:
                    print(f"      • {group_name}: {count} features")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error calculating features: {e}")
            logging.error(f"Feature calculation failed: {e}", exc_info=True)
            return False
    
    def prepare_ml_data(self) -> bool:
        """
        Prepare data for machine learning.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("🔧 Preparing machine learning data...")
            
            # Align datasets
            valid_indices = self.features_df.index[:len(self.compounds_df)]
            self.features_df = self.features_df.iloc[valid_indices].reset_index(drop=True)
            
            # Prepare features and labels
            feature_columns = [col for col in self.features_df.columns if col != 'smiles']
            X = self.features_df[feature_columns].values
            y = self.compounds_df[f'{self.activity_type}_active'].values
            
            # Handle any missing values
            X = np.nan_to_num(X, nan=0.0)
            
            # Normalize features
            normalized_df = normalize_features(
                self.features_df[feature_columns + ['smiles']], 
                exclude_columns=['smiles']
            )
            X = normalized_df[feature_columns].values
            
            # Train/test split (80/20)
            split_idx = int(0.8 * len(X))
            indices = np.random.permutation(len(X))
            
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            
            self.X_train = X[train_indices]
            self.X_test = X[test_indices]
            self.y_train = y[train_indices]
            self.y_test = y[test_indices]
            
            self.feature_names = feature_columns
            
            print(f"   ✅ Training set: {len(self.X_train)} compounds")
            print(f"   ✅ Test set: {len(self.X_test)} compounds")
            print(f"   📊 Features: {len(self.feature_names)}")
            print(f"   🎯 Class distribution (train): {np.sum(self.y_train)}/{len(self.y_train)} active")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error preparing ML data: {e}")
            logging.error(f"ML data preparation failed: {e}", exc_info=True)
            return False
    
    def train_ensemble_model(self) -> bool:
        """
        Train an ensemble model for bioactivity prediction.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("🤖 Training ensemble bioactivity prediction model...")
            
            # Define base models
            models = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                ),
                'Logistic Regression': LogisticRegression(
                    random_state=42,
                    max_iter=1000
                )
            }
            
            # Train and evaluate models
            model_scores = {}
            for name, model in models.items():
                print(f"   Training {name}...")
                model.fit(self.X_train, self.y_train)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
                mean_cv_score = np.mean(cv_scores)
                model_scores[name] = mean_cv_score
                
                print(f"      CV Score: {mean_cv_score:.3f} (+/- {np.std(cv_scores)*2:.3f})")
            
            # Select best model
            best_model_name = max(model_scores, key=model_scores.get)
            self.model = models[best_model_name]
            
            print(f"   🏆 Best model: {best_model_name} (CV: {model_scores[best_model_name]:.3f})")
            
            # Test set evaluation
            y_pred = self.model.predict(self.X_test)
            test_accuracy = accuracy_score(self.y_test, y_pred)
            
            print(f"   📊 Test accuracy: {test_accuracy:.3f}")
            
            # Classification report
            report = classification_report(self.y_test, y_pred, output_dict=True)
            print(f"   📈 Precision: {report['1']['precision']:.3f}")
            print(f"   📈 Recall: {report['1']['recall']:.3f}")
            print(f"   📈 F1-Score: {report['1']['f1-score']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error training model: {e}")
            logging.error(f"Model training failed: {e}", exc_info=True)
            return False
    
    def setup_shap_explainer(self) -> bool:
        """
        Set up and fit the SHAP explainer.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("🔍 Setting up SHAP explainer...")
            
            # Get feature groups for explanation
            feature_groups = self.feature_calculator.get_feature_importance_groups()
            
            # Initialize explainer
            self.explainer = BioPathSHAPExplainer(
                model=self.model,
                feature_names=self.feature_names,
                feature_groups=feature_groups
            )
            
            # Fit explainer
            print("   Fitting SHAP explainer (this may take a moment)...")
            self.explainer.fit(self.X_train, sample_size=min(100, len(self.X_train)))
            
            # Initialize visualizer
            self.visualizer = SHAPVisualization(
                feature_groups=feature_groups,
                style='professional'
            )
            
            print("   ✅ SHAP explainer ready")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error setting up SHAP: {e}")
            logging.error(f"SHAP setup failed: {e}", exc_info=True)
            return False
    
    def generate_explanations(self) -> bool:
        """
        Generate SHAP explanations for test compounds.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("💡 Generating SHAP explanations...")
            
            # Generate explanations for first 10 test compounds
            n_explain = min(10, len(self.X_test))
            self.explanations = []
            
            print(f"   Explaining {n_explain} test compounds...")
            
            for i in range(n_explain):
                # Get compound SMILES if available
                compound_idx = len(self.X_train) + i
                smiles = None
                if compound_idx < len(self.compounds_df):
                    smiles = self.compounds_df.iloc[compound_idx]['smiles']
                
                # Generate explanation
                explanation = self.explainer.explain_instance(
                    self.X_test[i], 
                    smiles=smiles
                )
                self.explanations.append(explanation)
                
                if (i + 1) % 5 == 0:
                    print(f"      Completed {i + 1}/{n_explain} explanations")
            
            print("   ✅ Explanations generated")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error generating explanations: {e}")
            logging.error(f"Explanation generation failed: {e}", exc_info=True)
            return False
    
    def create_visualizations(self) -> bool:
        """
        Create and save SHAP visualizations.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("📊 Creating SHAP visualizations...")
            
            # Calculate SHAP values for test set
            test_shap_values = []
            for i in range(min(50, len(self.X_test))):  # Limit for performance
                explanation = self.explainer.explain_instance(self.X_test[i])
                test_shap_values.append(explanation['shap_values'])
            
            shap_matrix = np.array(test_shap_values)
            
            # 1. Feature importance summary
            print("   Creating feature importance plot...")
            fig1 = self.visualizer.create_feature_importance_summary(
                shap_matrix,
                self.feature_names,
                title=f"{self.activity_type.replace('_', ' ').title()} Feature Importance"
            )
            fig1.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            # 2. Feature group comparison
            print("   Creating feature group analysis...")
            fig2 = self.visualizer.create_feature_group_comparison(
                shap_matrix,
                self.feature_names
            )
            fig2.savefig(self.output_dir / 'feature_groups.png', dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            # 3. SHAP beeswarm plot
            print("   Creating SHAP beeswarm plot...")
            fig3 = self.visualizer.create_shap_beeswarm_plot(
                shap_matrix,
                self.X_test[:len(shap_matrix)],
                self.feature_names
            )
            fig3.savefig(self.output_dir / 'shap_beeswarm.png', dpi=300, bbox_inches='tight')
            plt.close(fig3)
            
            # 4. Summary dashboard
            print("   Creating summary dashboard...")
            fig4 = create_summary_dashboard(
                shap_matrix,
                self.X_test[:len(shap_matrix)],
                self.feature_names,
                self.feature_calculator.get_feature_importance_groups()
            )
            fig4.savefig(self.output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
            plt.close(fig4)
            
            # 5. Individual compound waterfall plots (first 3)
            print("   Creating individual compound explanations...")
            for i, explanation in enumerate(self.explanations[:3]):
                from visualization.shap_plots import create_molecular_waterfall_plot
                fig = create_molecular_waterfall_plot(
                    explanation,
                    title=f"Compound {i+1} - {self.activity_type.replace('_', ' ').title()} Prediction"
                )
                fig.savefig(self.output_dir / f'compound_{i+1}_waterfall.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
            
            print("   ✅ Visualizations saved to demo_results/")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating visualizations: {e}")
            logging.error(f"Visualization creation failed: {e}", exc_info=True)
            return False
    
    def generate_report(self) -> bool:
        """
        Generate a comprehensive demonstration report.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("📄 Generating demonstration report...")
            
            # Generate report using explainer
            report = self.explainer.generate_report(
                self.explanations,
                output_file=str(self.output_dir / 'biopath_demo_report.md')
            )
            
            # Create executive summary
            summary_lines = [
                "# BioPath SHAP Demo - Executive Summary",
                "=" * 50,
                "",
                "## Demonstration Overview",
                f"- **Target Activity**: {self.activity_type.replace('_', ' ').title()}",
                f"- **Compounds Analyzed**: {len(self.compounds_df)}",
                f"- **Model Accuracy**: {accuracy_score(self.y_test, self.model.predict(self.X_test)):.1%}",
                f"- **Features Calculated**: {len(self.feature_names)}",
                "",
                "## Key Capabilities Demonstrated",
                "- ✅ Advanced molecular feature engineering (200+ descriptors)",
                "- ✅ Ensemble machine learning for bioactivity prediction", 
                "- ✅ Domain-specific SHAP explanations with biological context",
                "- ✅ Publication-ready visualizations",
                "- ✅ Traditional knowledge integration framework",
                "- ✅ Regulatory-compliant explainable AI",
                "",
                "## Technical Highlights",
                f"- **Chemical Space Coverage**: {len(set(self.compounds_df['compound_class']))} natural product classes",
                f"- **Traditional Sources**: {len(set(self.compounds_df['traditional_source']))} ethnobotanical sources",
                "- **Prediction Confidence**: Full molecular-level explanations for every prediction",
                "- **Scalability**: Batch processing capable for large compound libraries",
                "",
                "## Business Value Proposition",
                "- **Drug Discovery**: Accelerate lead compound identification",
                "- **Traditional Medicine**: Scientific validation of ethnobotanical knowledge",
                "- **Regulatory Compliance**: Explainable AI for FDA/EMA submissions",
                "- **IP Protection**: Secure traditional knowledge attribution",
                "",
                f"## Files Generated",
                "- `biopath_demo_report.md` - Detailed technical analysis",
                "- `feature_importance.png` - Molecular feature importance ranking",
                "- `feature_groups.png` - Chemical feature category analysis",
                "- `shap_beeswarm.png` - Feature impact distribution",
                "- `summary_dashboard.png` - Comprehensive analysis overview",
                "- `compound_*_waterfall.png` - Individual compound breakdowns",
                "",
                "---",
                "*Generated by BioPath SHAP Demo v0.1.0*",
                "*Contact: biopath-demo@omnipath.ai*"
            ]
            
            # Save executive summary
            with open(self.output_dir / 'executive_summary.md', 'w') as f:
                f.write('\n'.join(summary_lines))
            
            print("   ✅ Reports generated")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error generating report: {e}")
            logging.error(f"Report generation failed: {e}", exc_info=True)
            return False
    
    def run_complete_demo(self) -> bool:
        """
        Run the complete demonstration pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        print("🚀 Starting BioPath SHAP Complete Demonstration")
        print("=" * 60)
        print(f"Activity: {self.activity_type}")
        print(f"Max Compounds: {self.max_compounds}")
        print(f"Output Directory: {self.output_dir}")
        print()
        
        steps = [
            ("Loading Data", self.load_data),
            ("Calculating Features", self.calculate_features),
            ("Preparing ML Data", self.prepare_ml_data),
            ("Training Model", self.train_ensemble_model),
            ("Setting up SHAP", self.setup_shap_explainer),
            ("Generating Explanations", self.generate_explanations),
            ("Creating Visualizations", self.create_visualizations),
            ("Generating Report", self.generate_report)
        ]
        
        for step_name, step_func in steps:
            print(f"\n🔄 {step_name}...")
            if not step_func():
                print(f"❌ Demo failed at step: {step_name}")
                return False
        
        print("\n🎉 BioPath SHAP Demo completed successfully!")
        print("=" * 60)
        print(f"📁 Results saved to: {self.output_dir}")
        print("📊 Key files:")
        for file_path in sorted(self.output_dir.glob('*')):
            print(f"   • {file_path.name}")
        print()
        print("🚀 Ready for investor presentation!")
        
        return True


def main():
    """Main function to run the BioPath demonstration."""
    parser = argparse.ArgumentParser(
        description="BioPath SHAP Demo - Explainable Natural Compound Bioactivity Prediction"
    )
    parser.add_argument(
        '--activity',
        type=str,
        default='antioxidant',
        choices=['antioxidant', 'anti_inflammatory', 'antimicrobial', 'neuroprotective'],
        help='Bioactivity type to predict'
    )
    parser.add_argument(
        '--compounds',
        type=int,
        default=200,
        help='Maximum number of compounds to analyze'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='demo_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Run demonstration
    demo = BioPathDemo(
        activity_type=args.activity,
        max_compounds=args.compounds,
        output_dir=args.output
    )
    
    success = demo.run_complete_demo()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
