"""
Sample Data Generator for BioPath SHAP Demo

This module generates realistic natural compound datasets for bioactivity prediction
demonstrations, including traditional medicine compounds, known bioactive molecules,
and synthetic derivatives with believable bioactivity patterns.
"""

import os
import logging
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NaturalCompoundGenerator:
    """
    Generator for realistic natural compound datasets with bioactivity labels.
    
    This class creates chemically plausible natural products with realistic
    bioactivity patterns based on known structure-activity relationships.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the natural compound generator.
        
        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Define realistic natural product SMILES templates
        self._init_natural_product_templates()
        
        # Define bioactivity rules based on molecular features
        self._init_bioactivity_rules()
        
        logging.info("NaturalCompoundGenerator initialized")
    
    def _init_natural_product_templates(self):
        """Initialize SMILES templates for different natural product classes."""
        self.templates = {
            'flavonoids': [
                'c1cc(ccc1c2cc(=O)c3c(cc(cc3o2)O)O)O',  # Quercetin-like
                'c1cc(ccc1c2cc(=O)c3c(cc(cc3o2)O)O)O',  # Kaempferol-like
                'COc1cc(cc(c1O)OC)c2cc(=O)c3c(cc(c(c3o2)O)O)O'  # Myricetin-like
            ],
            'alkaloids': [
                'CN1CCc2cc3c(cc2C1)OCO3',  # Berberine-like
                'CCN(CC)CCc1c[nH]c2ccc(cc12)O',  # Serotonin-like
                'CN1c2ccccc2C(=O)N(C1=O)C'  # Caffeine-like
            ],
            'terpenoids': [
                'CC1=C2CC(CC(C2CCC1)(C)C)C(=C)C',  # Limonene-like
                'CC1CCC2(C(C1)C(=C)C(=O)O2)C',  # Parthenolide-like
                'CC1(C2CCC(C(C2)O)(CCC1O)C)C'  # Betulin-like
            ],
            'phenolic_acids': [
                'c1cc(c(cc1CC(C(=O)O)N)O)O',  # DOPA-like
                'c1cc(ccc1C=CC(=O)O)O',  # p-Coumaric acid-like
                'COc1cc(ccc1O)C=CC(=O)O'  # Ferulic acid-like
            ],
            'glycosides': [
                'c1cc(ccc1c2cc(=O)c3c(cc(cc3o2)O)O)OC4C(C(C(C(O4)CO)O)O)O',  # Quercetin-3-glucoside-like
                'CC(C)(C)c1cc(cc(c1O)C(C)(C)C)C(=O)OC2C(C(C(C(O2)CO)O)O)O'  # Arbutin-like
            ],
            'saponins': [
                'CC1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)O)O)O',  # Simple saponin core
                'CC(C)C1CCC2(C(C1)CCC3(C2CC=C(C3)C(=O)O)C)C'  # Triterpene saponin-like
            ]
        }
        
        # Traditional medicine sources for compound attribution
        self.traditional_sources = [
            'Echinacea purpurea', 'Ginkgo biloba', 'Panax ginseng', 'Curcuma longa',
            'Hypericum perforatum', 'Valeriana officinalis', 'Silybum marianum',
            'Camellia sinensis', 'Vitis vinifera', 'Allium sativum', 'Zingiber officinale',
            'Capsicum annuum', 'Rosmarinus officinalis', 'Lavandula angustifolia',
            'Matricaria chamomilla', 'Calendula officinalis', 'Arnica montana',
            'Aloe vera', 'Withania somnifera', 'Bacopa monnieri'
        ]
    
    def _init_bioactivity_rules(self):
        """Initialize rules for realistic bioactivity prediction based on molecular features."""
        self.bioactivity_rules = {
            'antioxidant': {
                'favorable': ['phenol_groups', 'hydroxyl_groups', 'aromatic_rings'],
                'weights': [0.3, 0.25, 0.2],
                'thresholds': {'phenol_groups': 2, 'hydroxyl_groups': 3, 'aromatic_rings': 1}
            },
            'anti_inflammatory': {
                'favorable': ['phenol_groups', 'carbonyl_groups', 'logp'],
                'weights': [0.35, 0.25, 0.15],
                'thresholds': {'phenol_groups': 1, 'carbonyl_groups': 1, 'logp': 2.0}
            },
            'antimicrobial': {
                'favorable': ['aromatic_rings', 'quaternary_n', 'hba_count'],
                'weights': [0.3, 0.4, 0.2],
                'thresholds': {'aromatic_rings': 2, 'quaternary_n': 0, 'hba_count': 3}
            },
            'neuroprotective': {
                'favorable': ['logp', 'molecular_weight', 'rotatable_bonds'],
                'weights': [0.4, -0.2, -0.1],  # Negative weights for MW and rotatable bonds
                'thresholds': {'logp': 1.5, 'molecular_weight': 400, 'rotatable_bonds': 8}
            }
        }
    
    def generate_natural_compound(self, compound_class: str = None) -> Tuple[str, str, str]:
        """
        Generate a single natural compound with modifications.
        
        Args:
            compound_class: Specific class of natural product to generate
            
        Returns:
            Tuple of (SMILES, compound_class, traditional_source)
        """
        if compound_class is None:
            compound_class = random.choice(list(self.templates.keys()))
        
        # Select a template from the chosen class
        base_template = random.choice(self.templates[compound_class])
        
        # Apply realistic modifications
        modified_smiles = self._apply_natural_modifications(base_template)
        
        # Assign a traditional source
        traditional_source = random.choice(self.traditional_sources)
        
        return modified_smiles, compound_class, traditional_source
    
    def _apply_natural_modifications(self, smiles: str) -> str:
        """
        Apply realistic chemical modifications to base natural product templates.
        
        Args:
            smiles: Base SMILES string
            
        Returns:
            Modified SMILES string
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            
            # Apply random modifications with realistic probabilities
            modifications = [
                (self._add_methyl_group, 0.3),
                (self._add_hydroxyl_group, 0.25),
                (self._add_methoxy_group, 0.2),
                (self._remove_hydroxyl, 0.1),
                (self._add_glycoside, 0.15)
            ]
            
            for modification_func, probability in modifications:
                if random.random() < probability:
                    try:
                        mol = modification_func(mol)
                        if mol is None:
                            break
                    except:
                        continue
            
            if mol is not None:
                return Chem.MolToSmiles(mol)
            else:
                return smiles
                
        except Exception as e:
            logging.warning(f"Error modifying SMILES {smiles}: {e}")
            return smiles
    
    def _add_methyl_group(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Add a methyl group to a random carbon."""
        try:
            # Simple methylation simulation
            smiles = Chem.MolToSmiles(mol)
            if 'c' in smiles.lower() and random.random() < 0.5:
                # Add methoxy to aromatic carbons
                smiles = smiles.replace('c1c', f'c1c(C)', 1) if random.random() < 0.5 else smiles
            return Chem.MolFromSmiles(smiles)
        except:
            return mol
    
    def _add_hydroxyl_group(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Add a hydroxyl group to a random carbon."""
        try:
            smiles = Chem.MolToSmiles(mol)
            if 'c' in smiles.lower():
                smiles = smiles.replace('c1c', f'c1c(O)', 1) if random.random() < 0.3 else smiles
            return Chem.MolFromSmiles(smiles)
        except:
            return mol
    
    def _add_methoxy_group(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Add a methoxy group."""
        try:
            smiles = Chem.MolToSmiles(mol)
            if 'c' in smiles.lower():
                smiles = smiles.replace('c1c', f'c1c(OC)', 1) if random.random() < 0.2 else smiles
            return Chem.MolFromSmiles(smiles)
        except:
            return mol
    
    def _remove_hydroxyl(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Remove a hydroxyl group."""
        try:
            smiles = Chem.MolToSmiles(mol)
            if 'O)' in smiles:
                smiles = smiles.replace('O)', ')', 1) if random.random() < 0.3 else smiles
            return Chem.MolFromSmiles(smiles)
        except:
            return mol
    
    def _add_glycoside(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Add a simple glycoside moiety."""
        try:
            smiles = Chem.MolToSmiles(mol)
            if 'O' in smiles and random.random() < 0.1:
                # Simple glucose attachment
                glucose = 'OC1C(C(C(C(O1)CO)O)O)O'
                smiles = smiles.replace('O', glucose, 1)
            return Chem.MolFromSmiles(smiles)
        except:
            return mol
    
    def calculate_bioactivity_probability(self, smiles: str, activity_type: str) -> float:
        """
        Calculate bioactivity probability based on molecular features.
        
        Args:
            smiles: SMILES string of the compound
            activity_type: Type of bioactivity to predict
            
        Returns:
            Probability of bioactivity (0-1)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.5  # Default probability for invalid molecules
            
            # Calculate relevant molecular features
            features = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'hba_count': Descriptors.NumHAcceptors(mol),
                'hbd_count': Descriptors.NumHDonors(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'phenol_groups': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH][c]'))),
                'carbonyl_groups': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]=[#8]'))),
                'hydroxyl_groups': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]'))),
                'quaternary_n': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[N+]')))
            }
            
            # Apply bioactivity rules
            if activity_type not in self.bioactivity_rules:
                return np.random.uniform(0.3, 0.7)  # Random probability for unknown activities
            
            rules = self.bioactivity_rules[activity_type]
            score = 0.5  # Base probability
            
            for feature, weight in zip(rules['favorable'], rules['weights']):
                if feature in features:
                    feature_value = features[feature]
                    threshold = rules['thresholds'].get(feature, 1)
                    
                    if weight > 0:
                        # Positive contribution
                        if feature_value >= threshold:
                            score += weight * min(feature_value / threshold, 2.0) * 0.1
                    else:
                        # Negative contribution (penalize high values)
                        if feature_value > threshold:
                            score += weight * (feature_value / threshold) * 0.1
            
            # Add some noise for realism
            score += np.random.normal(0, 0.05)
            
            # Ensure probability is between 0 and 1
            return np.clip(score, 0.05, 0.95)
            
        except Exception as e:
            logging.warning(f"Error calculating bioactivity for {smiles}: {e}")
            return np.random.uniform(0.3, 0.7)
    
    def generate_dataset(self, 
                        n_compounds: int = 500,
                        activity_types: List[str] = None,
                        test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate a complete dataset with compounds and bioactivity labels.
        
        Args:
            n_compounds: Number of compounds to generate
            activity_types: List of bioactivity types to predict
            test_size: Fraction of data for test set
            
        Returns:
            Tuple of (training_df, test_df)
        """
        if activity_types is None:
            activity_types = ['antioxidant', 'anti_inflammatory', 'antimicrobial', 'neuroprotective']
        
        logging.info(f"Generating dataset with {n_compounds} compounds...")
        
        # Generate compounds
        compounds_data = []
        for i in range(n_compounds):
            smiles, compound_class, traditional_source = self.generate_natural_compound()
            
            # Validate SMILES
            if Chem.MolFromSmiles(smiles) is None:
                continue
            
            compound_info = {
                'compound_id': f'NP_{i+1:04d}',
                'smiles': smiles,
                'compound_class': compound_class,
                'traditional_source': traditional_source
            }
            
            # Calculate bioactivity probabilities and binary labels
            for activity in activity_types:
                prob = self.calculate_bioactivity_probability(smiles, activity)
                compound_info[f'{activity}_probability'] = prob
                compound_info[f'{activity}_active'] = int(prob > 0.5)
            
            compounds_data.append(compound_info)
            
            if (i + 1) % 100 == 0:
                logging.info(f"Generated {i + 1}/{n_compounds} compounds")
        
        # Create DataFrame
        df = pd.DataFrame(compounds_data)
        
        # Split into training and test sets
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=self.random_state)
        
        logging.info(f"Dataset generation complete: {len(train_df)} training, {len(test_df)} test compounds")
        
        return train_df, test_df


def create_sample_datasets():
    """
    Create sample datasets for the BioPath SHAP demo.
    
    This function generates all necessary datasets and saves them to the data directory.
    """
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = NaturalCompoundGenerator(random_state=42)
    
    # Generate main dataset
    logging.info("Creating main bioactivity dataset...")
    train_df, test_df = generator.generate_dataset(
        n_compounds=800,
        activity_types=['antioxidant', 'anti_inflammatory', 'antimicrobial', 'neuroprotective']
    )
    
    # Save datasets
    train_df.to_csv(data_dir / 'natural_compounds_train.csv', index=False)
    test_df.to_csv(data_dir / 'natural_compounds_test.csv', index=False)
    
    # Create a smaller demo dataset
    logging.info("Creating smaller demo dataset...")
    demo_df = train_df.sample(n=100, random_state=42)
    demo_df.to_csv(data_dir / 'sample_compounds.csv', index=False)
    
    # Create traditional knowledge validation dataset
    logging.info("Creating traditional knowledge validation dataset...")
    traditional_df = train_df[train_df['traditional_source'].notna()].copy()
    traditional_df['traditional_use'] = traditional_df['traditional_source'].map(
        lambda x: random.choice(['digestive', 'respiratory', 'immune_support', 'stress_relief', 'anti_inflammatory'])
    )
    traditional_df.to_csv(data_dir / 'traditional_knowledge_compounds.csv', index=False)
    
    # Generate statistics summary
    stats_summary = {
        'total_compounds': len(train_df) + len(test_df),
        'training_compounds': len(train_df),
        'test_compounds': len(test_df),
        'compound_classes': train_df['compound_class'].value_counts().to_dict(),
        'activity_distribution': {
            activity: {
                'active': int(train_df[f'{activity}_active'].sum()),
                'inactive': int((~train_df[f'{activity}_active'].astype(bool)).sum()),
                'mean_probability': float(train_df[f'{activity}_probability'].mean())
            }
            for activity in ['antioxidant', 'anti_inflammatory', 'antimicrobial', 'neuroprotective']
        },
        'traditional_sources': train_df['traditional_source'].value_counts().head(10).to_dict()
    }
    
    # Save statistics
    import json
    with open(data_dir / 'dataset_statistics.json', 'w') as f:
        json.dump(stats_summary, f, indent=2)
    
    logging.info("Sample datasets created successfully!")
    logging.info(f"Files created in {data_dir}:")
    for file_path in data_dir.glob('*.csv'):
        logging.info(f"  - {file_path.name}")
    
    return train_df, test_df, demo_df


if __name__ == "__main__":
    print("🧬 BioPath SHAP Demo - Sample Data Generator")
    print("=" * 50)
    print("Generating realistic natural compound datasets for demonstration...")
    print()
    
    try:
        train_df, test_df, demo_df = create_sample_datasets()
        
        print("✅ Dataset generation completed successfully!")
        print()
        print("📊 Dataset Summary:")
        print(f"  • Training compounds: {len(train_df)}")
        print(f"  • Test compounds: {len(test_df)}")
        print(f"  • Demo compounds: {len(demo_df)}")
        print()
        print("🎯 Bioactivity Distribution (Training Set):")
        for activity in ['antioxidant', 'anti_inflammatory', 'antimicrobial', 'neuroprotective']:
            active_count = train_df[f'{activity}_active'].sum()
            total_count = len(train_df)
            percentage = (active_count / total_count) * 100
            print(f"  • {activity.replace('_', ' ').title()}: {active_count}/{total_count} ({percentage:.1f}% active)")
        print()
        print("🌿 Compound Classes:")
        class_counts = train_df['compound_class'].value_counts()
        for class_name, count in class_counts.items():
            print(f"  • {class_name.replace('_', ' ').title()}: {count} compounds")
        print()
        print("Ready for BioPath SHAP demonstrations! 🚀")
        
    except Exception as e:
        print(f"❌ Error generating datasets: {e}")
        logging.error(f"Dataset generation failed: {e}", exc_info=True)
