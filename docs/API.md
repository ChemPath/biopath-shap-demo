# BioPath SHAP Demo API Documentation

## Overview

The BioPath SHAP Demo provides a comprehensive API for molecular feature calculation, bioactivity prediction, and explainable AI analysis of natural compounds. This documentation covers all major components and their usage.

## Core Components

### 1. Molecular Feature Calculator

#### `ModernMolecularFeatureCalculator`

Main class for calculating molecular descriptors and fingerprints from SMILES strings.


from data_preprocessing.molecular_features import ModernMolecularFeatureCalculator

Initialize calculator
calculator = ModernMolecularFeatureCalculator(
include_fingerprints=True,
fingerprint_radius=2
)

Calculate features for a single compound
features = calculator.calculate_all_features('CCO')

Process multiple compounds
smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O']
features_df = calculator.process_batch(smiles_list)



**Key Methods:**

- `calculate_all_features(smiles: str) -> Dict[str, Union[float, int]]`
  - Calculate comprehensive molecular features for a SMILES string
  - Returns dictionary of feature names and values
  - Returns None for invalid SMILES

- `process_batch(smiles_list: List[str], show_progress: bool = True) -> pd.DataFrame`
  - Process multiple SMILES strings efficiently
  - Returns DataFrame with all calculated features
  - Includes progress tracking for large datasets

- `get_feature_groups() -> Dict[str, List[str]]`
  - Returns feature groupings for SHAP analysis
  - Groups features by chemical relevance

**Feature Categories:**
- **Basic Properties**: molecular weight, atom counts, ring counts
- **Drug-likeness**: LogP, TPSA, HBD/HBA counts, QED score
- **Structural Complexity**: Bertz complexity, Balaban index
- **Natural Product Features**: sp3 fraction, chiral centers, stereocenters
- **Functional Groups**: phenol, carbonyl, hydroxyl, ether groups
- **Pharmacophore Features**: hydrophobic, HBD/HBA regions
- **Fingerprint Features**: Morgan, atom pair, topological torsion bits

### 2. Bioactivity Prediction Models

#### `BioactivityPredictor`

Enterprise-grade ensemble model for natural compound bioactivity prediction.

from models.bioactivity_predictor import BioactivityPredictor, ModelConfig

Configure model
config = ModelConfig(
model_type='ensemb
e', use_hyperparameter_optimizati
n=True, cross_valida
Initialize predictor
predictor = BioactivityPredictor(config)

Train model
predictor.fit(X_train, y_train, feature_names)

Make predictions
predictions = predictor.predict(X_test)
probabi


**Key Methods:**

- `fit(X, y, feature_names, validation_split=True) -> BioactivityPredictor`
  - Train the bioactivity prediction model
  - Supports validation splitting and cross-validation
  - Returns self for method chaining

- `predict(X: np.ndarray) -> np.ndarray`
  - Make binary predictions on new data
  - Returns predicted class labels (0 or 1)

- `predict_proba(X: np.ndarray) -> np.ndarray`
  - Predict class probabilities
  - Returns probability distributions for each class

- `get_feature_importance() -> Dict[str, float]`
  - Extract feature importance scores
  - Returns dictionary mapping feature names to importance values

- `save_model(filepath: str)` / `load_model(filepath: str)`
  - Serialize and deserialize trained models
  - Includes all model components and metadata

**Configuration Options:**
- `model_type`: 'ensemble', 'random_forest', 'gradient_boosting', 'optimized'
- `use_hyperparameter_optimization`: Enable Optuna-based optimization
- `cross_validation_folds`: Number of CV folds for validation
- `feature_selection`: Enable automatic feature selection

### 3. SHAP Explainable AI

#### `ModernBioPathSHAPExplainer`

Custom SHAP explainer with biological interpretations for molecular bioactivity.

from explainers.bio_shap_explainer import ModernBioPathSHAPExplainer

Initialize explainer
explainer = ModernBioPathSHAPExplainer(
model=trained_mo
el, feature_names=featur
_names, feature_groups=fe
Fit explainer
explainer.fit(X_train, sample_size=100)

Generate explanation
explanation = explainer.explain_instance(X_test, compound_id='compound_1')


**Key Methods:**

- `fit(X: np.ndarray, sample_size: int = 100)`
  - Fit SHAP explainer to training data
  - Automatic model type detection for optimal explainer selection

- `explain_instance(X_instance, compound_id=None) -> Dict[str, Any]`
  - Generate explanation for single compound
  - Returns SHAP values and biological interpretations

- `generate_summary_report(explanations, output_file=None) -> str`
  - Create comprehensive report for multiple compounds
  - Includes executive summary and detailed analysis

**Explanation Output:**
- `shap_values`: Feature contribution values
- `base_value`: Model's expected output
- `feature_values`: Original feature values
- `prediction`: Model prediction
- `confidence`: Prediction confidence (if available)
- `biological_interpretations`: Domain-specific explanations

#### `BiologicalInterpretation`

Data class for storing biological meanings of molecular features.

@dataclass
class BiologicalInterpretation:
feature_name: str
shap_value: float
contribution_strength: str # "Strong", "Moderate", "Weak"
biological_meaning: str
confidence_score: float
traditional_knowledge_link: Optional[str] = None
regulatory_relevance: Optional[str] = None


### 4. Visualization Components

#### `SHAPVisualization`

Advanced visualization toolkit for molecular SHAP analysis.

from visualization.shap_plots import SHAPVisualization

Initialize visualizer
visualizer = SHAPVisualization(
feature_groups=feature_groups,
style='professional'
)

Create feature importance plot
fig = visualizer.create_feature_importance_summary(
shap_values_matrix,
feature_names,
title="Molecular Feature Importance"
)

Create beeswarm plot
fig = visualizer.create_shap_beeswarm_plot(
shap_values_matrix,
feature_values_matrix,
feature_names
)


**Key Methods:**

- `create_feature_importance_summary(shap_values, feature_names, max_features=20)`
  - Generate comprehensive feature importance visualization
  - Includes feature grouping and biological context

- `create_shap_beeswarm_plot(shap_values, feature_values, feature_names)`
  - Create SHAP beeswarm plot showing feature distributions
  - Shows relationship between feature values and SHAP contributions

- `create_feature_group_comparison(shap_values, feature_names)`
  - Compare contributions across different feature groups
  - Useful for understanding chemical property importance

- `create_molecular_heatmap(shap_values, feature_names, compound_names)`
  - Generate heatmap of SHAP values across compounds
  - Optional clustering for pattern identification

**Visualization Styles:**
- `professional`: Clean, publication-ready plots
- `scientific`: Academic journal formatting
- `presentation`: Bold, high-contrast for presentations

### 5. Sample Data Generation

#### `NaturalCompoundGenerator`

Enhanced generator for realistic natural compound datasets.

from data.generate_sample_data import NaturalCompoundGenerator

Initialize generator
generator = NaturalCompoundGenerator(random_state=42)

Generate single compound
smiles, compound_class, source = generator.generate_compound()

Generate complete dataset
train_df, test_df = generator.generate_dataset(
n_compounds=500,
activity_types=['antioxidant', 'anti_inflammatory']
)


**Key Methods:**

- `generate_compound(compound_class=None) -> Tuple[str, str, str]`
  - Generate single natural compound with metadata
  - Returns SMILES, compound class, and traditional source

- `generate_dataset(n_compounds, activity_types, test_size=0.2)`
  - Generate complete training and test datasets
  - Includes bioactivity labels and comprehensive metadata

- `calculate_bioactivity_probability(smiles, activity_type) -> float`
  - Calculate realistic bioactivity probabilities
  - Based on structure-activity relationships

**Compound Classes:**
- `flavonoids`: Quercetin-like compounds with antioxidant activity
- `alkaloids`: Berberine-like compounds with diverse activities
- `terpenoids`: Limonene-like compounds with anti-inflammatory properties
- `phenolic_acids`: Coumaric acid-like compounds with antioxidant properties
- `saponins`: Glycoside-like compounds with immunomodulatory effects

## Usage Examples

### Complete Workflow Example

1. Generate sample data
from data.generate_sample_data import create_enhanced_sample_datasets
main_df, demo_df = create_enhanced_sample_datasets()

2. Calculate molecular features
from data_preprocessing.molecular_features import ModernMolecularFeatureCalculator
calculator = ModernMolecularFeatureCalculator()
features_df = calculator.process_batch(demo_df['smiles'].tolist())

3. Prepare ML data
feature_columns = [col for col in features_df.columns if col != 'smiles']
X = features_df[feature_columns].values
y = demo_df['antioxidant_active'].values

4. Train model
from models.bioactivity_predictor import BioactivityPredictor, ModelConfig
config = ModelConfig(model_type='ensemble')
predictor = BioactivityPredictor(config)
predictor.fit(X, y, feature_columns)

5. Setup SHAP explainer
from explainers.bio_shap_explainer import ModernBioPathSHAPExplainer
explainer = ModernBioPathSHAPExplainer(
model=predictor.model,
feature_names=feature_columns,
feature_groups=calculator.get_feature_groups()
)
explainer.fit(X)

6. Generate explanations
explanations = []
for i in range(5): # First 5 compounds
explanation = explainer.explain_instance(X[i], compound_id=f'compound_{i}')
explanations.append(explanation)

7. Create visualizations
from visualization.shap_plots import SHAPVisualization
visualizer = SHAPVisualization(feature_groups=calculator.get_feature_groups())
shap_matrix = np.array([exp['shap_values'] for exp in explanations])
fig = visualizer.create_feature_importance_summary(shap_matrix, feature_columns)

8. Generate report
report = explainer.generate_summary_report(explanations, 'biopath_report.md')


### CLI Usage

The package includes command-line interfaces for common operations:

Predict bioactivity
biopath-predict --input compounds.csv --output predictions.csv --model trained_model.pkl

Generate SHAP explanations
biopath-explain --input compounds.csv --model trained_model.pkl --output explanations.json

Calculate molecular features
biopath-features --input smiles.txt --output features.csv --fingerprints


## Error Handling

All API functions include comprehensive error handling:

try:
features = calculator.calculate_all_features(smiles)
if features is None:
print(f"Invalid SMILES: {smiles}")
except Exception as e:
print(f"Error processing compound: {e}")


## Performance Considerations

### Batch Processing
- Use `process_batch()` for multiple compounds
- Implement progress tracking for large datasets
- Consider memory usage with large feature matrices

### Model Training
- Feature selection reduces overfitting
- Cross-validation provides reliable performance estimates
- Hyperparameter optimization improves accuracy

### SHAP Calculations
- TreeExplainer is faster for tree-based models
- KernelExplainer works with any model type
- Sample size affects explanation quality vs. speed

## Integration Guidelines

### Custom Feature Calculators
class CustomFeatureCalculator(ModernMolecularFeatureCalculator):
def calculate_custom_features(self, mol):
# Add custom molecular features
return custom_features


### Custom SHAP Interpretations

Extend biological meanings
explainer.biological_meanings.update({
'custom_feature': {
'positive': 'High values indicate...',
'negative': 'Low values suggest...',
'confidence': 0.85
}
})


### Custom Visualizations
Add custom plotting functions
def create_custom_plot(shap_values, feature_names):
# Custom visualization logic
return fig

## Testing and Validation

### Unit Tests
Run all tests
pytest tests/ -v

Run specific test module
pytest tests/test_molecular_features.py -v

Run with coverage
pytest tests/ --cov=src/

### Integration Tests
Test complete workflow
python examples/biopath_demo.py

Test with sample data
python examples/demo_script.py

## API Reference Summary

| Component | Main Class | Key Methods | Purpose |
|-----------|------------|-------------|---------|
| Features | `ModernMolecularFeatureCalculator` | `calculate_all_features()`, `process_batch()` | Molecular descriptor calculation |
| Models | `BioactivityPredictor` | `fit()`, `predict()`, `predict_proba()` | Bioactivity prediction |
| Explainers | `ModernBioPathSHAPExplainer` | `fit()`, `explain_instance()` | SHAP-based explanations |
| Visualization | `SHAPVisualization` | `create_feature_importance_summary()` | SHAP visualizations |
| Data Generation | `NaturalCompoundGenerator` | `generate_dataset()` | Sample data creation |

## Support and Documentation

- **GitHub Repository**: [github.com/Omnipath2025/biopath-shap-demo](https://github.com/Omnipath2025/biopath-shap-demo)
- **Issues and Bug Reports**: Use GitHub Issues
- **Documentation**: Full documentation available in `/docs` directory
- **Examples**: Working examples in `/examples` directory

## Version Information

- **Current Version**: 2.0.0
- **Python Requirements**: >= 3.8
- **Key Dependencies**: RDKit >= 2023.9.1, SHAP >= 0.42.0, scikit-learn >= 1.3.0

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

*This API documentation covers the complete BioPath SHAP Demo functionality. For additional examples and tutorials, see the examples directory and Jupyter notebooks.*
