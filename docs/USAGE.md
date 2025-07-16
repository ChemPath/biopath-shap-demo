# BioPath SHAP Demo Usage Guide

## Overview

This guide provides step-by-step instructions for using the BioPath SHAP Demo to analyze natural compound bioactivity with explainable AI. The demo showcases molecular feature calculation, bioactivity prediction, and SHAP-based explanations for natural products with traditional knowledge integration.

## Prerequisites

### System Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space
- Internet connection for package installation

### Required Dependencies

pip install -r requirements.txt


## Quick Start

### 1. Basic Demo Execution

Run the complete demonstration with default settings:

python examples/demo_script.py


This will:
- Load sample natural compounds from `data/sample_compounds.csv`
- Calculate comprehensive molecular features using RDKit
- Train ensemble machine learning models for bioactivity prediction
- Generate SHAP explanations with biological interpretations
- Create publication-ready visualizations
- Produce comprehensive analysis reports

### 2. Interactive Jupyter Notebook

Launch the interactive analysis notebook:

jupyter notebook examples/demo_notebook.ipynb


This provides a step-by-step walkthrough of the entire BioPath SHAP pipeline with detailed explanations and visualizations.

### 3. Custom Activity Analysis

Specify different bioactivity targets:

python examples/demo_script.py --activity anti_inflammatory --compounds 100


Available activities:
- `antioxidant` (default)
- `anti_inflammatory`
- `antimicrobial`
- `neuroprotective`

## Detailed Usage Instructions

### Data Preparation

#### Using Sample Data
The demo includes 50 pre-generated natural compounds with bioactivity labels:

import pandas as pd
compounds_df = pd.read_csv('data/sample_compounds.csv')
print(f"Loaded {len(compounds_df)} compounds")


#### Custom Compound Analysis
Analyze your own compounds with SMILES strings:

from src.data_preprocessing.molecular_features import ModernMolecularFeatureCalculator

calculator = ModernMolecularFeatureCalculator()
custom_smiles = [
'c1cc(ccc1c2cc(=O)c3c(cc(cc3o2)O)O)O', # Quercetin
'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' # Caffeine
]

features_df = calculator.process_batch(custom_smiles)


### Molecular Feature Calculation

#### Basic Feature Calculation

from src.data_preprocessing.molecular_features import ModernMolecularFeatureCalculator

Initialize calculator
calculator = ModernMolecularFeatureCalculator(
include_fingerprints=True,
fingerprint_radius=2
)

Calculate features for a single compound
features = calculator.calculate_all_features('CCO') # Ethanol
print(f"Calculated {len(features)} features")

Process multiple compounds
smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O']
features_df = calculator.process_batch(smiles_list)


#### Available Feature Categories
- **Basic Properties**: molecular weight, atom counts, ring counts
- **Drug-likeness**: LogP, TPSA, HBD/HBA counts, QED score
- **Structural Complexity**: Bertz complexity, Balaban index
- **Natural Product Features**: sp3 fraction, chiral centers, stereocenters
- **Functional Groups**: phenol, carbonyl, hydroxyl, ether groups
- **Pharmacophore Features**: hydrophobic, HBD/HBA regions
- **Fingerprint Features**: Morgan, atom pair, topological torsion bits

### Bioactivity Prediction

#### Model Training

from src.models.bioactivity_predictor import BioactivityPredictor, ModelConfig

Configure model
config = ModelConfig(
model_type='ensemble',
use_hyperparameter_optimization=True,
cross_validation_folds=5
)

Train model
predictor = BioactivityPredictor(config)
predictor.fit(X_train, y_train, feature_names)

Make predictions
predictions = predictor.predict(X_test)
probabilities = predictor.predict_proba(X_test)


#### Available Model Types
- `ensemble`: Voting classifier with multiple algorithms
- `random_forest`: Random Forest classifier
- `gradient_boosting`: Gradient Boosting classifier
- `optimized`: Hyperparameter-optimized model

### SHAP Explainability Analysis

#### Generate Explanations

from src.explainers.bio_shap_explainer import ModernBioPathSHAPExplainer

Setup explainer
explainer = ModernBioPathSHAPExplainer(
model=trained_model,
feature_names=feature_names,
feature_groups=feature_groups
)

Fit explainer
explainer.fit(X_train, sample_size=100)

Explain individual compounds
explanation = explainer.explain_instance(
X_test,
compound_id='quercetin'
)


#### Understanding Explanation Output

Extract key information
shap_values = explanation['shap_values']
prediction = explanation['prediction']
confidence = explanation['confidence']
biological_interpretations = explanation['biological_interpretations']

View top contributing features
for interp in biological_interpretations[:5]:
print(f"Feature: {interp.feature_name}")
print(f"Impact: {interp.contribution_strength}")
print(f"Meaning: {interp.biological_meaning}")
print("---")


### Visualization Creation

#### Feature Importance Plot

from src.visualization.shap_plots import SHAPVisualization

visualizer = SHAPVisualization(
feature_groups=feature_groups,
style='professional'
)

Create feature importance summary
fig = visualizer.create_feature_importance_summary(
shap_values_matrix,
feature_names,
title="Natural Compound Feature Importance"
)


#### SHAP Beeswarm Plot

Show feature value distributions
fig = visualizer.create_shap_beeswarm_plot(
shap_values_matrix,
feature_values_matrix,
feature_names
)


## Advanced Usage

### Custom Feature Engineering

#### Add Custom Molecular Descriptors

from src.data_preprocessing.molecular_features import ModernMolecularFeatureCalculator

class CustomFeatureCalculator(ModernMolecularFeatureCalculator):
def calculate_custom_features(self, mol):
custom_features = {}
# Add your custom calculations here
custom_features['my_descriptor'] = my_calculation(mol)
return custom_features


### Batch Processing

#### Process Large Datasets

import pandas as pd
from src.data_preprocessing.molecular_features import ModernMolecularFeatureCalculator

Load large dataset
compounds_df = pd.read_csv('large_compound_dataset.csv')

Process in batches
calculator = ModernMolecularFeatureCalculator()
batch_size = 1000

results = []
for i in range(0, len(compounds_df), batch_size):
batch = compounds_df.iloc[i:i+batch_size]
batch_features = calculator.process_batch(
batch['smiles'].tolist(),
show_progress=True
)
results.append(batch_features)

Combine results
final_features = pd.concat(results, ignore_index=True)


### Model Optimization

#### Hyperparameter Tuning

from src.models.bioactivity_predictor import BioactivityPredictor, ModelConfig

Advanced configuration
config = ModelConfig(
model_type='optimized',
use_hyperparameter_optimization=True,
optimization_trials=200,
cross_validation_folds=10,
feature_selection=True
)

predictor = BioactivityPredictor(config)
predictor.fit(X_train, y_train, feature_names)


### Report Generation

#### Comprehensive Analysis Report

Generate detailed report
explanations = []
for i in range(len(X_test)):
explanation = explainer.explain_instance(
X_test[i],
compound_id=f'compound_{i+1}'
)
explanations.append(explanation)

Create report
report = explainer.generate_summary_report(
explanations,
output_file='bioactivity_analysis_report.md'
)


## Command Line Interface

### Available Commands

#### Predict Bioactivity

biopath-predict --input compounds.csv --output predictions.csv --model trained_model.pkl


#### Calculate Features

biopath-features --input smiles.txt --output features.csv --fingerprints


## Troubleshooting

### Common Issues

#### RDKit Installation Problems

Install RDKit via conda
conda install -c conda-forge rdkit

Or via pip
pip install rdkit-pypi


#### Memory Issues with Large Datasets

Use batch processing
batch_size = 500 # Reduce if needed
calculator = ModernMolecularFeatureCalculator(
include_fingerprints=False # Reduce memory usage
)


#### Invalid SMILES Handling

Check SMILES validity
from rdkit import Chem

def validate_smiles(smiles):
return Chem.MolFromSmiles(smiles) is not None

Filter invalid SMILES
valid_smiles = [s for s in smiles_list if validate_smiles(s)]


### Performance Optimization

#### Speed Up Feature Calculation

Disable fingerprints for faster processing
calculator = ModernMolecularFeatureCalculator(
include_fingerprints=False
)

Use parallel processing
from multiprocessing import Pool

def process_smiles_batch(smiles_batch):
calc = ModernMolecularFeatureCalculator()
return calc.process_batch(smiles_batch)

Split into chunks and process in parallel
chunks = [smiles_list[i:i+100] for i in range(0, len(smiles_list), 100)]
with Pool() as pool:
results = pool.map(process_smiles_batch, chunks)


## Output Files

### Generated Files
- `biopath_shap_analysis_report.md`: Main analysis report
- `feature_importance.png`: Feature importance visualization
- `shap_beeswarm.png`: SHAP beeswarm plot
- `compound_*_waterfall.png`: Individual compound explanations

### File Formats
- **CSV**: Tabular data (features, predictions)
- **JSON**: Structured explanation data
- **PNG**: High-resolution visualizations
- **HTML**: Interactive reports
- **MD**: Markdown reports

## Best Practices

### Data Quality
- Validate SMILES strings before processing
- Remove duplicates and invalid structures
- Check for reasonable molecular weight ranges
- Verify activity labels are balanced

### Model Training
- Use stratified cross-validation
- Monitor for overfitting
- Validate on external test sets
- Document model parameters

### SHAP Analysis
- Use appropriate sample sizes
- Validate explanations make chemical sense
- Consider feature interactions
- Document interpretation methods

## Example Workflows

### Complete Analysis Pipeline

1. Load and prepare data
compounds_df = pd.read_csv('data/sample_compounds.csv')

2. Calculate features
calculator = ModernMolecularFeatureCalculator()
features_df = calculator.process_batch(compounds_df['smiles'].tolist())

3. Prepare ML data
X = features_df.drop('smiles', axis=1).values
y = compounds_df['antioxidant_active'].values

4. Train model
predictor = BioactivityPredictor()
predictor.fit(X, y, feature_names)

5. Generate explanations
explainer = ModernBioPathSHAPExplainer(predictor.model, feature_names)
explainer.fit(X)

6. Analyze results
explanations = []
for i in range(10): # First 10 compounds
explanation = explainer.explain_instance(X[i])
explanations.append(explanation)

7. Create report
report = explainer.generate_summary_report(explanations)


## Support and Documentation

### Getting Help
- Review this usage guide and API documentation
- Check example scripts in `examples/`
- Run unit tests to verify installation: `pytest tests/`

### File Structure
biopath-shap-demo/
├── README.md
├── requirements.txt
├── setup.py
├── data/
│ └── sample_compounds.csv
├── src/
│ ├── data_preprocessing/
│ ├── models/
│ ├── explainers/
│ └── visualization/
├── examples/
│ ├── demo_script.py
│ └── demo_notebook.ipynb
├── tests/
└── docs/
├── API.md
└── USAGE.md


## Version Information

- **Current Version**: 2.0.0
- **Python Requirements**: >= 3.8
- **Key Dependencies**: RDKit >= 2023.9.1, SHAP >= 0.42.0, scikit-learn >= 1.3.0

---

*This usage guide covers the complete BioPath SHAP Demo functionality. For additional technical details, see the API documentation and example notebooks.*





