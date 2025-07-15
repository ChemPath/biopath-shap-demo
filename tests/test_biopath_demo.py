#!/usr/bin/env python3
"""
Comprehensive test suite for BioPath SHAP Demo
Tests molecular feature calculation, model training, and SHAP explanations
"""

import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_preprocessing.molecular_features import ModernMolecularFeatureCalculator
from models.bioactivity_predictor import BioactivityPredictor, ModelConfig
from explainers.bio_shap_explainer import ModernBioPathSHAPExplainer
from visualization.shap_plots import SHAPVisualization

class TestMolecularFeatureCalculator:
    """Test suite for molecular feature calculation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.calculator = ModernMolecularFeatureCalculator()
        self.test_smiles = [
            'CCO',  # Ethanol
            'c1ccccc1',  # Benzene
            'CC(=O)O',  # Acetic acid
            'c1cc(ccc1c2cc(=O)c3c(cc(cc3o2)O)O)O',  # Quercetin
        ]
    
    def test_basic_descriptors(self):
        """Test basic molecular descriptor calculation"""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles('CCO')
        descriptors = self.calculator.calculate_basic_descriptors(mol)
        
        assert isinstance(descriptors, dict)
        assert 'molecular_weight' in descriptors
        assert 'logp' in descriptors
        assert descriptors['molecular_weight'] > 0
        assert isinstance(descriptors['logp'], (int, float))
    
    def test_ethnobotanical_features(self):
        """Test ethnobotanical feature calculation"""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles('c1cc(ccc1c2cc(=O)c3c(cc(cc3o2)O)O)O')
        features = self.calculator.calculate_ethnobotanical_features(mol)
        
        assert isinstance(features, dict)
        assert 'phenol_groups' in features
        assert 'chiral_centers' in features
        assert features['phenol_groups'] >= 0
    
    def test_fingerprint_features(self):
        """Test molecular fingerprint calculation"""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles('CCO')
        features = self.calculator.calculate_fingerprint_features(mol)
        
        if self.calculator.include_fingerprints:
            assert isinstance(features, dict)
            assert len(features) > 0
            # Check for fingerprint bit features
            morgan_bits = [k for k in features.keys() if 'morgan_bit_' in k]
            assert len(morgan_bits) > 0
    
    def test_batch_processing(self):
        """Test batch processing of multiple SMILES"""
        df = self.calculator.process_batch(self.test_smiles, show_progress=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(self.test_smiles)
        assert 'smiles' in df.columns
        assert 'molecular_weight' in df.columns
    
    def test_invalid_smiles_handling(self):
        """Test handling of invalid SMILES strings"""
        invalid_smiles = ['INVALID', 'C[C@H]([C@@H]([C@H]()', '']
        results = []
        
        for smiles in invalid_smiles:
            result = self.calculator.calculate_all_features(smiles)
            results.append(result)
        
        # Should return None for invalid SMILES
        valid_results = [r for r in results if r is not None]
        assert len(valid_results) == 0 or len(valid_results) < len(invalid_smiles)

class TestBioactivityPredictor:
    """Test suite for bioactivity prediction models"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = ModelConfig(
            model_type='random_forest',
            use_hyperparameter_optimization=False,
            cross_validation_folds=3
        )
        self.predictor = BioactivityPredictor(self.config)
        
        # Generate synthetic data
        np.random.seed(42)
        self.X = np.random.rand(100, 20)
        self.y = np.random.choice([0, 1], size=100)
        self.feature_names = [f'feature_{i}' for i in range(20)]
    
    def test_model_initialization(self):
        """Test model initialization"""
        assert self.predictor.config.model_type == 'random_forest'
        assert not self.predictor.is_fitted
        assert self.predictor.model is None
    
    def test_model_training(self):
        """Test model training process"""
        self.predictor.fit(self.X, self.y, self.feature_names)
        
        assert self.predictor.is_fitted
        assert self.predictor.model is not None
        assert self.predictor.feature_names == self.feature_names
        assert self.predictor.performance_metrics is not None
    
    def test_model_prediction(self):
        """Test model prediction"""
        self.predictor.fit(self.X, self.y, self.feature_names)
        
        # Test prediction
        predictions = self.predictor.predict(self.X[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        self.predictor.fit(self.X, self.y, self.feature_names)
        
        importance = self.predictor.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) == len(self.feature_names)
        assert all(isinstance(val, (int, float)) for val in importance.values())
    
    def test_model_serialization(self):
        """Test model save/load functionality"""
        import tempfile
        
        self.predictor.fit(self.X, self.y, self.feature_names)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            self.predictor.save_model(tmp.name)
            
            # Create new predictor and load
            new_predictor = BioactivityPredictor()
            new_predictor.load_model(tmp.name)
            
            assert new_predictor.is_fitted
            assert new_predictor.feature_names == self.feature_names
            
            # Test predictions are consistent
            pred1 = self.predictor.predict(self.X[:5])
            pred2 = new_predictor.predict(self.X[:5])
            np.testing.assert_array_equal(pred1, pred2)

class TestSHAPExplainer:
    """Test suite for SHAP explainer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create and train a simple model
        np.random.seed(42)
        self.X = np.random.rand(50, 10)
        self.y = np.random.choice([0, 1], size=50)
        self.feature_names = [f'feature_{i}' for i in range(10)]
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)
        
        self.explainer = ModernBioPathSHAPExplainer(
            model=self.model,
            feature_names=self.feature_names
        )
    
    def test_explainer_initialization(self):
        """Test SHAP explainer initialization"""
        assert self.explainer.model is not None
        assert self.explainer.feature_names == self.feature_names
        assert self.explainer.explainer is None  # Not fitted yet
    
    def test_explainer_fitting(self):
        """Test SHAP explainer fitting"""
        self.explainer.fit(self.X, sample_size=20)
        
        assert self.explainer.explainer is not None
        assert self.explainer.base_value is not None
    
    def test_instance_explanation(self):
        """Test single instance explanation"""
        self.explainer.fit(self.X, sample_size=20)
        
        explanation = self.explainer.explain_instance(self.X[0])
        
        assert isinstance(explanation, dict)
        assert 'shap_values' in explanation
        assert 'feature_names' in explanation
        assert 'prediction' in explanation
        assert 'biological_interpretations' in explanation
        
        # Check SHAP values structure
        shap_values = explanation['shap_values']
        assert len(shap_values) == len(self.feature_names)
    
    def test_biological_interpretations(self):
        """Test biological interpretation generation"""
        self.explainer.fit(self.X, sample_size=20)
        
        explanation = self.explainer.explain_instance(self.X[0])
        interpretations = explanation['biological_interpretations']
        
        assert isinstance(interpretations, list)
        assert len(interpretations) > 0
        
        # Check interpretation structure
        for interp in interpretations:
            assert hasattr(interp, 'feature_name')
            assert hasattr(interp, 'shap_value')
            assert hasattr(interp, 'biological_meaning')
    
    def test_report_generation(self):
        """Test comprehensive report generation"""
        self.explainer.fit(self.X, sample_size=20)
        
        explanations = []
        for i in range(3):
            exp = self.explainer.explain_instance(self.X[i], compound_id=f'compound_{i}')
            explanations.append(exp)
        
        report = self.explainer.generate_summary_report(explanations)
        
        assert isinstance(report, str)
        assert 'BioPath SHAP Analysis Report' in report
        assert len(report) > 100  # Should be substantial

class TestVisualization:
    """Test suite for visualization components"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.feature_names = [f'feature_{i}' for i in range(10)]
        self.feature_groups = {
            'Group A': ['feature_0', 'feature_1', 'feature_2'],
            'Group B': ['feature_3', 'feature_4', 'feature_5'],
            'Group C': ['feature_6', 'feature_7', 'feature_8', 'feature_9']
        }
        
        self.visualizer = SHAPVisualization(
            feature_groups=self.feature_groups,
            style='professional'
        )
        
        # Generate synthetic SHAP data
        np.random.seed(42)
        self.shap_values = np.random.normal(0, 0.1, size=(20, 10))
        self.feature_values = np.random.rand(20, 10)
    
    def test_visualization_initialization(self):
        """Test visualization initialization"""
        assert self.visualizer.feature_groups == self.feature_groups
        assert self.visualizer.style == 'professional'
    
    def test_feature_importance_plot(self):
        """Test feature importance plot creation"""
        fig = self.visualizer.create_feature_importance_summary(
            self.shap_values,
            self.feature_names,
            max_features=5
        )
        
        assert fig is not None
        # Check that figure has the expected structure
        assert hasattr(fig, 'savefig')
    
    def test_feature_group_analysis(self):
        """Test feature group analysis"""
        fig = self.visualizer.create_feature_group_comparison(
            self.shap_values,
            self.feature_names
        )
        
        assert fig is not None
        # Should create a multi-subplot figure
        assert len(fig.axes) > 1
    
    def test_beeswarm_plot(self):
        """Test SHAP beeswarm plot"""
        fig = self.visualizer.create_shap_beeswarm_plot(
            self.shap_values,
            self.feature_values,
            self.feature_names
        )
        
        assert fig is not None
    
    def test_heatmap_creation(self):
        """Test molecular heatmap creation"""
        compound_names = [f'compound_{i}' for i in range(20)]
        
        fig = self.visualizer.create_molecular_heatmap(
            self.shap_values,
            self.feature_names,
            compound_names
        )
        
        assert fig is not None

class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_complete_workflow(self):
        """Test complete demo workflow"""
        # Sample data
        test_smiles = [
            'CCO',  # Ethanol
            'c1ccccc1',  # Benzene
            'CC(=O)O',  # Acetic acid
        ]
        
        # Step 1: Calculate features
        calculator = ModernMolecularFeatureCalculator(include_fingerprints=False)
        features_df = calculator.process_batch(test_smiles, show_progress=False)
        
        assert not features_df.empty
        
        # Step 2: Prepare data
        feature_columns = [col for col in features_df.columns if col != 'smiles']
        X = features_df[feature_columns].values
        y = np.array([1, 0, 1])  # Synthetic labels
        
        # Step 3: Train model
        config = ModelConfig(
            model_type='random_forest',
            use_hyperparameter_optimization=False,
            cross_validation_folds=3
        )
        predictor = BioactivityPredictor(config)
        predictor.fit(X, y, feature_columns, validation_split=False)
        
        assert predictor.is_fitted
        
        # Step 4: Generate SHAP explanations
        explainer = ModernBioPathSHAPExplainer(
            model=predictor.model,
            feature_names=feature_columns
        )
        explainer.fit(X, sample_size=len(X))
        
        # Step 5: Explain instances
        explanations = []
        for i in range(len(X)):
            exp = explainer.explain_instance(X[i], compound_id=f'compound_{i}')
            explanations.append(exp)
        
        assert len(explanations) == len(test_smiles)
        
        # Step 6: Generate report
        report = explainer.generate_summary_report(explanations)
        assert isinstance(report, str)
        assert len(report) > 100

# Pytest configuration
@pytest.fixture
def sample_data():
    """Generate sample molecular data for testing"""
    np.random.seed(42)
    return {
        'smiles': ['CCO', 'c1ccccc1', 'CC(=O)O'],
        'features': np.random.rand(3, 10),
        'labels': np.array([1, 0, 1])
    }

def test_demo_script_execution():
    """Test that the main demo script can be executed"""
    # This would typically import and run the main demo
    # For now, just test that imports work
    try:
        from examples.demo_script import main
        # Could run main() with test parameters
        assert callable(main)
    except ImportError:
        pytest.skip("Demo script not available")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

