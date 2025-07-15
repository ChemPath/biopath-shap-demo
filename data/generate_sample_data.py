#!/usr/bin/env python3
"""
Enhanced Sample Data Generator for BioPath SHAP Demo
Generates realistic natural compound datasets with bioactivity labels
"""

import os
import logging
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Configure logging
logging.basicConfig(level=logging.INFO)

class NaturalCompoundGenerator:
    """Enhanced generator for realistic natural compound datasets"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Comprehensive natural product templates
        self.natural_products = {
            'flavonoids': {
                'templates': [
                    'c1cc(ccc1c2cc(=O)c3c(cc(cc3o2)O)O)O',  # Quercetin
                    'c1cc(ccc1c2cc(=O)c3c(cc(cc3o2)O)O)O',  # Kaempferol
                    'COc1cc(cc(c1O)OC)c2cc(=O)c3c(cc(c(c3o2)O)O)O',  # Myricetin
                    'c1cc(ccc1c2cc(=O)c3c(cc(cc3o2)O)O)O',  # Luteolin
                ],
                'activity_prob': 0.85,
                'sources': ['berries', 'citrus', 'tea', 'onions', 'apples']
            },
            'alkaloids': {
                'templates': [
                    'CN1CCc2cc3c(cc2C1)OCO3',  # Berberine-like
                    'CN1CC[C@H]2c3c(cccc3[C@@H]21)O',  # Morphine-like
                    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
                    'CCN(CC)CC(=O)Nc1c(C)cccc1C',  # Lidocaine-like
                ],
                'activity_prob': 0.80,
                'sources': ['coffee', 'tea', 'poppy', 'cinchona', 'tobacco']
            },
            'terpenoids': {
                'templates': [
                    'CC1=C2CC(CC(C2CCC1)(C)C)C(=C)C',  # Limonene
                    'CC1CCC2(C(C1)C(=C)C(=O)O2)C',  # Parthenolide-like
                    'CC(C)C1CCC2(C(C1)CCC3(C2CC(CC3)O)C)C',  # Cholesterol-like
                    'CC1(C2CCC(C(C2)O)(CCC1O)C)C',  # Betulin-like
                ],
                'activity_prob': 0.70,
                'sources': ['mint', 'eucalyptus', 'pine', 'lavender', 'rosemary']
            },
            'phenolic_acids': {
                'templates': [
                    'c1cc(c(cc1CC(C(=O)O)N)O)O',  # DOPA
                    'c1cc(ccc1C=CC(=O)O)O',  # p-Coumaric acid
                    'COc1cc(ccc1O)C=CC(=O)O',  # Ferulic acid
                    'c1cc(c(c(c1)O)O)C(=O)O',  # Gallic acid
                ],
                'activity_prob': 0.75,
                'sources': ['grapes', 'berries', 'coffee', 'cinnamon', 'cloves']
            },
            'saponins': {
                'templates': [
                    'CC1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)O)O)O',  # Glycoside
                    'CC(C)C1CCC2(C(C1)CCC3(C2CC=C(C3)C(=O)O)C)C',  # Triterpene
                ],
                'activity_prob': 0.65,
                'sources': ['ginseng', 'licorice', 'soapberry', 'yucca']
            },
            'inactive_controls': {
                'templates': [
                    'CCO',  # Ethanol
                    'c1ccccc1',  # Benzene
                    'CC(=O)O',  # Acetic acid
                    'CCCCCCCCCCCCCCCC(=O)O',  # Palmitic acid
                ],
                'activity_prob': 0.15,
                'sources': ['synthetic', 'industrial']
            }
        }
        
        # Traditional sources with cultural context
        self.traditional_sources = {
            'Echinacea purpurea': {'region': 'North America', 'use': 'immune support'},
            'Ginkgo biloba': {'region': 'Asia', 'use': 'cognitive enhancement'},
            'Panax ginseng': {'region': 'Asia', 'use': 'energy, vitality'},
            'Curcuma longa': {'region': 'India', 'use': 'anti-inflammatory'},
            'Hypericum perforatum': {'region': 'Europe', 'use': 'mood support'},
            'Camellia sinensis': {'region': 'Asia', 'use': 'antioxidant'},
            'Vitis vinifera': {'region': 'Mediterranean', 'use': 'cardiovascular'},
            'Allium sativum': {'region': 'Central Asia', 'use': 'antimicrobial'},
            'Zingiber officinale': {'region': 'Asia', 'use': 'digestive'},
            'Capsicum annuum': {'region': 'Americas', 'use': 'topical pain'},
        }
        
        logging.info("Enhanced NaturalCompoundGenerator initialized")
    
    def generate_compound(self, compound_class: str = None) -> Tuple[str, str, str, float]:
        """Generate a natural compound with enhanced metadata"""
        if compound_class is None:
            compound_class = random.choice(list(self.natural_products.keys()))
        
        class_data = self.natural_products[compound_class]
        base_template = random.choice(class_data['templates'])
        
        # Apply natural modifications
        modified_smiles = self._apply_natural_modifications(base_template)
        
        # Assign traditional source
        if compound_class != 'inactive_controls':
            traditional_source = random.choice(list(self.traditional_sources.keys()))
        else:
            traditional_source = 'synthetic'
        
        # Calculate activity probability
        activity_prob = class_data['activity_prob']
        
        return modified_smiles, compound_class, traditional_source, activity_prob
    
    def _apply_natural_modifications(self, smiles: str) -> str:
        """Apply realistic chemical modifications to base templates"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            
            # Apply modifications with natural probabilities
            modifications = [
                (self._add_hydroxyl_group, 0.30),
                (self._add_methoxy_group, 0.25),
                (self._add_methyl_group, 0.20),
                (self._add_glycoside, 0.15),
                (self._remove_hydroxyl, 0.10),
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
    
    def _add_hydroxyl_group(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Add hydroxyl group to aromatic carbon"""
        try:
            smiles = Chem.MolToSmiles(mol)
            if 'c' in smiles.lower() and random.random() < 0.5:
                smiles = smiles.replace('c1c', 'c1c(O)', 1)
            return Chem.MolFromSmiles(smiles)
        except:
            return mol
    
    def _add_methoxy_group(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Add methoxy group"""
        try:
            smiles = Chem.MolToSmiles(mol)
            if 'c' in smiles.lower() and random.random() < 0.3:
                smiles = smiles.replace('c1c', 'c1c(OC)', 1)
            return Chem.MolFromSmiles(smiles)
        except:
            return mol
    
    def _add_methyl_group(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Add methyl group"""
        try:
            smiles = Chem.MolToSmiles(mol)
            if 'c' in smiles.lower() and random.random() < 0.4:
                smiles = smiles.replace('c1c', 'c1c(C)', 1)
            return Chem.MolFromSmiles(smiles)
        except:
            return mol
    
    def _remove_hydroxyl(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Remove hydroxyl group"""
        try:
            smiles = Chem.MolToSmiles(mol)
            if 'O)' in smiles and random.random() < 0.2:
                smiles = smiles.replace('O)', ')', 1)
            return Chem.MolFromSmiles(smiles)
        except:
            return mol
    
    def _add_glycoside(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Add simple glycoside moiety"""
        try:
            smiles = Chem.MolToSmiles(mol)
            if 'O' in smiles and random.random() < 0.1:
                glucose = 'OC1C(C(C(C(O1)CO)O)O)O'
                smiles = smiles.replace('O', glucose, 1)
            return Chem.MolFromSmiles(smiles)
        except:
            return mol
    
    def calculate_enhanced_bioactivity(self, smiles: str, compound_class: str, 
                                     traditional_source: str) -> Dict[str, float]:
        """Calculate enhanced bioactivity probabilities"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {'antioxidant': 0.3, 'anti_inflammatory': 0.3, 'antimicrobial': 0.3}
            
            # Calculate molecular features
            features = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd_count': Descriptors.NumHDonors(mol),
                'hba_count': Descriptors.NumHAcceptors(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'phenol_groups': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH][c]'))),
                'hydroxyl_groups': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]'))),
                'carbonyl_groups': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]=[#8]'))),
            }
            
            # Enhanced bioactivity rules
            bioactivity_rules = {
                'antioxidant': {
                    'phenol_groups': (0.4, 2),
                    'hydroxyl_groups': (0.3, 3),
                    'aromatic_rings': (0.2, 1),
                    'molecular_weight': (-0.1, 500),
                },
                'anti_inflammatory': {
                    'phenol_groups': (0.35, 1),
                    'carbonyl_groups': (0.25, 1),
                    'logp': (0.20, 2.5),
                    'molecular_weight': (-0.05, 400),
                },
                'antimicrobial': {
                    'aromatic_rings': (0.3, 2),
                    'logp': (0.3, 3.0),
                    'hba_count': (0.2, 5),
                    'molecular_weight': (-0.1, 350),
                }
            }
            
            activities = {}
            base_prob = self.natural_products[compound_class]['activity_prob']
            
            for activity, rules in bioactivity_rules.items():
                score = base_prob * 0.7  # Base score
                
                for feature, (weight, threshold) in rules.items():
                    feature_value = features[feature]
                    
                    if weight > 0:
                        if feature_value >= threshold:
                            score += weight * min(feature_value / threshold, 2.0) * 0.15
                    else:
                        if feature_value > threshold:
                            score += weight * (feature_value / threshold) * 0.1
                
                # Add traditional knowledge bonus
                if traditional_source in self.traditional_sources:
                    traditional_use = self.traditional_sources[traditional_source]['use']
                    if activity in traditional_use or 'antioxidant' in traditional_use:
                        score += 0.1
                
                # Add noise and clamp
                score += np.random.normal(0, 0.05)
                activities[activity] = np.clip(score, 0.05, 0.95)
            
            return activities
            
        except Exception as e:
            logging.warning(f"Error calculating bioactivity for {smiles}: {e}")
            return {'antioxidant': 0.3, 'anti_inflammatory': 0.3, 'antimicrobial': 0.3}
    
    def generate_enhanced_dataset(self, n_compounds: int = 200) -> pd.DataFrame:
        """Generate enhanced dataset with comprehensive metadata"""
        logging.info(f"Generating enhanced dataset with {n_compounds} compounds...")
        
        compounds_data = []
        
        # Ensure good distribution of compound classes
        class_distribution = {
            'flavonoids': 0.25,
            'alkaloids': 0.20,
            'terpenoids': 0.20,
            'phenolic_acids': 0.15,
            'saponins': 0.10,
            'inactive_controls': 0.10
        }
        
        for compound_class, fraction in class_distribution.items():
            n_class = int(n_compounds * fraction)
            
            for i in range(n_class):
                smiles, comp_class, traditional_source, base_prob = self.generate_compound(compound_class)
                
                # Validate SMILES
                if Chem.MolFromSmiles(smiles) is None:
                    continue
                
                # Calculate bioactivities
                activities = self.calculate_enhanced_bioactivity(smiles, comp_class, traditional_source)
                
                compound_info = {
                    'compound_id': f'NP_{len(compounds_data)+1:04d}',
                    'smiles': smiles,
                    'compound_class': comp_class,
                    'traditional_source': traditional_source,
                    'traditional_region': self.traditional_sources.get(traditional_source, {}).get('region', 'Unknown'),
                    'traditional_use': self.traditional_sources.get(traditional_source, {}).get('use', 'Unknown'),
                }
                
                # Add activity predictions
                for activity, prob in activities.items():
                    compound_info[f'{activity}_probability'] = prob
                    compound_info[f'{activity}_active'] = int(prob > 0.5)
                
                compounds_data.append(compound_info)
        
        df = pd.DataFrame(compounds_data)
        logging.info(f"Generated {len(df)} valid compounds")
        
        return df

def create_enhanced_sample_datasets():
    """Create enhanced sample datasets for BioPath SHAP demo"""
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = NaturalCompoundGenerator(random_state=42)
    
    # Generate main dataset
    logging.info("Creating enhanced bioactivity dataset...")
    main_df = generator.generate_enhanced_dataset(n_compounds=500)
    
    # Save main dataset
    main_df.to_csv(data_dir / 'enhanced_compounds_full.csv', index=False)
    
    # Create demo subset
    demo_df = main_df.sample(n=100, random_state=42)
    demo_df.to_csv(data_dir / 'sample_compounds.csv', index=False)
    
    # Create activity-specific datasets
    for activity in ['antioxidant', 'anti_inflammatory', 'antimicrobial']:
        activity_df = main_df[main_df[f'{activity}_active'] == 1].copy()
        activity_df.to_csv(data_dir / f'{activity}_compounds.csv', index=False)
    
    # Generate comprehensive statistics
    stats = {
        'total_compounds': len(main_df),
        'compound_classes': main_df['compound_class'].value_counts().to_dict(),
        'traditional_sources': main_df['traditional_source'].value_counts().head(10).to_dict(),
        'activity_distribution': {}
    }
    
    for activity in ['antioxidant', 'anti_inflammatory', 'antimicrobial']:
        active_count = main_df[f'{activity}_active'].sum()
        stats['activity_distribution'][activity] = {
            'active_compounds': int(active_count),
            'percentage': float(active_count / len(main_df) * 100),
            'mean_probability': float(main_df[f'{activity}_probability'].mean())
        }
    
    # Save statistics
    import json
    with open(data_dir / 'enhanced_dataset_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logging.info("Enhanced sample datasets created successfully!")
    return main_df, demo_df

if __name__ == "__main__":
    print("🧬 Enhanced BioPath SHAP Demo - Sample Data Generator")
    print("="*60)
    
    try:
        main_df, demo_df = create_enhanced_sample_datasets()
        
        print("✅ Enhanced dataset generation completed!")
        print(f"📊 Total compounds: {len(main_df)}")
        print(f"📊 Demo compounds: {len(demo_df)}")
        
        print("\n🎯 Activity Distribution:")
        for activity in ['antioxidant', 'anti_inflammatory', 'antimicrobial']:
            active_count = main_df[f'{activity}_active'].sum()
            percentage = active_count / len(main_df) * 100
            print(f"  • {activity.replace('_', ' ').title()}: {active_count}/{len(main_df)} ({percentage:.1f}%)")
        
        print("\n🌿 Compound Classes:")
        for class_name, count in main_df['compound_class'].value_counts().head(5).items():
            print(f"  • {class_name.replace('_', ' ').title()}: {count} compounds")
        
        print("\n🚀 Ready for BioPath SHAP demonstrations!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Dataset generation failed: {e}", exc_info=True)

