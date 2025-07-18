"""
Modern Molecular Feature Engineering for BioPath SHAP Demo
Updated to resolve RDKit deprecation issues and improve performance.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Updated RDKit imports to resolve deprecations
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, AllChem
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator, rdmolops
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.ML.Descriptors import MoleculeDescriptors

# Suppress RDKit warnings
warnings.filterwarnings('ignore', category=UserWarning, module='rdkit')
logging.getLogger('rdkit').setLevel(logging.ERROR)

class ModernMolecularFeatureCalculator:
    """
    Modern molecular feature calculator with updated RDKit API usage.
    Resolves deprecation warnings and improves computational efficiency.
    """
    
    def __init__(self, include_fingerprints: bool = True, fingerprint_radius: int = 2):
        """
        Initialize with modern RDKit feature generators.
        
        Args:
            include_fingerprints: Enable molecular fingerprint calculation
            fingerprint_radius: Morgan fingerprint radius
        """
        self.include_fingerprints = include_fingerprints
        self.fingerprint_radius = fingerprint_radius
        
        # Initialize modern fingerprint generators
        if include_fingerprints:
            self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=fingerprint_radius, 
                fpSize=2048
            )
            self.atom_pair_gen = rdFingerprintGenerator.GetAtomPairGenerator(
                fpSize=2048
            )
            self.topological_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
                fpSize=2048
            )
        
        # Initialize pharmacophore generator
        self.pharm_factory = Gobbi_Pharm2D.factory
        
        logging.info(f"ModernMolecularFeatureCalculator initialized")
    
    def calculate_basic_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate comprehensive molecular descriptors using modern RDKit API."""
        descriptors = {}
        
        try:
            # Basic molecular properties
            descriptors['molecular_weight'] = Descriptors.MolWt(mol)
            descriptors['heavy_atom_count'] = Descriptors.HeavyAtomCount(mol)
            descriptors['ring_count'] = Descriptors.RingCount(mol)
            descriptors['aromatic_rings'] = Descriptors.NumAromaticRings(mol)
            descriptors['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            
            # Hydrogen bonding descriptors
            descriptors['hbd_count'] = Descriptors.NumHDonors(mol)
            descriptors['hba_count'] = Descriptors.NumHAcceptors(mol)
            
            # Physicochemical properties
            descriptors['logp'] = Crippen.MolLogP(mol)
            descriptors['tpsa'] = Descriptors.TPSA(mol)
            descriptors['molar_refractivity'] = Crippen.MolMR(mol)
            
            # Drug-likeness metrics
            descriptors['qed_score'] = QED.qed(mol)
            
            # Updated sp3 fraction calculation for RDKit 2025.03.3
            try:
                from rdkit.Chem.Descriptors import FractionCsp3
                descriptors['sp3_fraction'] = FractionCsp3(mol)
            except ImportError:
                try:
                    from rdkit.Chem.Lipinski import FractionCSP3
                    descriptors['sp3_fraction'] = FractionCSP3(mol)
                except ImportError:
                    # Alternative calculation using hybridization
                    sp3_count = 0
                    total_carbons = 0
                    for atom in mol.GetAtoms():
                        if atom.GetAtomicNum() == 6:  # Carbon
                            total_carbons += 1
                            if atom.GetHybridization() == Chem.HybridizationType.SP3:
                                sp3_count += 1
                    descriptors['sp3_fraction'] = sp3_count / total_carbons if total_carbons > 0 else 0.0
            
            # Complexity descriptors
            descriptors['bertz_complexity'] = Descriptors.BertzCT(mol)
            descriptors['balaban_index'] = Descriptors.BalabanJ(mol)
            
            # Updated formal charge calculation for RDKit 2025.03.3
            try:
                from rdkit.Chem.Descriptors import FormalCharge
                descriptors['formal_charge'] = FormalCharge(mol)
            except ImportError:
                try:
                    from rdkit.Chem import rdmolops
                    descriptors['formal_charge'] = rdmolops.GetFormalCharge(mol)
                except ImportError:
                    # Manual calculation fallback
                    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
                    descriptors['formal_charge'] = formal_charge
            
            # Radical electrons
            descriptors['radical_electrons'] = Descriptors.NumRadicalElectrons(mol)
            
        except Exception as e:
            logging.warning(f"Error calculating basic descriptors: {e}")
            # Return default values for failed calculations
            descriptors = {
                'molecular_weight': 0.0,
                'heavy_atom_count': 0,
                'ring_count': 0,
                'aromatic_rings': 0,
                'rotatable_bonds': 0,
                'hbd_count': 0,
                'hba_count': 0,
                'logp': 0.0,
                'tpsa': 0.0,
                'molar_refractivity': 0.0,
                'qed_score': 0.0,
                'sp3_fraction': 0.0,
                'bertz_complexity': 0.0,
                'balaban_index': 0.0,
                'formal_charge': 0,
                'radical_electrons': 0
            }
        
        return descriptors
    
    def calculate_fingerprint_features(self, mol: Chem.Mol) -> Dict[str, int]:
        """Calculate molecular fingerprints using modern RDKit generators."""
        features = {}
        
        if not self.include_fingerprints:
            return features
        
        try:
            # Morgan fingerprints (updated API)
            morgan_fp = self.morgan_gen.GetFingerprint(mol)
            morgan_bits = morgan_fp.ToBitString()
            features.update({
                f'morgan_bit_{i}': int(bit) 
                for i, bit in enumerate(morgan_bits[:100])  # First 100 bits
            })
            
            # Atom pair fingerprints
            ap_fp = self.atom_pair_gen.GetFingerprint(mol)
            ap_bits = ap_fp.ToBitString()
            features.update({
                f'atompair_bit_{i}': int(bit) 
                for i, bit in enumerate(ap_bits[:50])  # First 50 bits
            })
            
            # Topological torsion fingerprints
            tt_fp = self.topological_gen.GetFingerprint(mol)
            tt_bits = tt_fp.ToBitString()
            features.update({
                f'torsion_bit_{i}': int(bit) 
                for i, bit in enumerate(tt_bits[:50])  # First 50 bits
            })
            
        except Exception as e:
            logging.warning(f"Error calculating fingerprint features: {e}")
        
        return features
    
    def calculate_ethnobotanical_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate features relevant to traditional medicine compounds."""
        features = {}
        
        try:
            # Natural product indicators
            features['chiral_centers'] = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            features['stereocenters'] = Descriptors.NumAliphaticCarbocycles(mol)
            
            # Functional group patterns with null checking
            patterns = {
                'phenol_groups': '[OH][c]',
                'hydroxyl_groups': '[OH]',
                'carbonyl_groups': '[#6]=[#8]',
                'ether_groups': '[#6]-[#8]-[#6]',
                'nitrogen_containing': '[N]',
                'basic_nitrogen': '[N;!$(N=*);!$(N-*=!#6)]',
                'sugar_like': '[CH2]([OH])[CH]([OH])'
            }
            
            for feature_name, pattern in patterns.items():
                try:
                    smarts_mol = Chem.MolFromSmarts(pattern)
                    if smarts_mol is not None:
                        features[feature_name] = len(mol.GetSubstructMatches(smarts_mol))
                    else:
                        features[feature_name] = 0
                except Exception as e:
                    logging.warning(f"Error calculating {feature_name}: {e}")
                    features[feature_name] = 0
            
        except Exception as e:
            logging.warning(f"Error calculating ethnobotanical features: {e}")
            features = {key: 0.0 for key in features.keys()}
        
        return features
    
    def calculate_all_features(self, smiles: str) -> Optional[Dict[str, Union[float, int]]]:
        """
        Calculate comprehensive molecular features for a SMILES string.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Dictionary of calculated features or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logging.warning(f"Invalid SMILES: {smiles}")
                return None
            
            # Add hydrogens for accurate calculations
            mol = Chem.AddHs(mol)
            
            # Calculate all feature categories
            features = {}
            features.update(self.calculate_basic_descriptors(mol))
            features.update(self.calculate_ethnobotanical_features(mol))
            features.update(self.calculate_fingerprint_features(mol))
            
            # Add SMILES for reference
            features['smiles'] = smiles
            
            return features
            
        except Exception as e:
            logging.error(f"Error processing SMILES {smiles}: {e}")
            return None
    
    def process_batch(self, smiles_list: List[str], show_progress: bool = True) -> pd.DataFrame:
        """
        Process batch of SMILES with progress tracking.
        
        Args:
            smiles_list: List of SMILES strings to process
            show_progress: Whether to show progress updates
            
        Returns:
            DataFrame with calculated features
        """
        results = []
        failed_count = 0
        
        for i, smiles in enumerate(smiles_list):
            if show_progress and (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(smiles_list)} molecules...")
            
            features = self.calculate_all_features(smiles)
            if features is not None:
                results.append(features)
            else:
                failed_count += 1
        
        if failed_count > 0:
            logging.warning(f"Failed to process {failed_count} molecules")
        
        # Convert to DataFrame and handle missing values
        df = pd.DataFrame(results)
        if not df.empty:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(0)
        
        logging.info(f"Successfully processed {len(df)} molecules")
        return df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get feature groups for SHAP analysis organization."""
        return {
            'Basic Properties': [
                'molecular_weight', 'heavy_atom_count', 'ring_count', 
                'aromatic_rings', 'rotatable_bonds'
            ],
            'Drug-likeness': [
                'logp', 'tpsa', 'hbd_count', 'hba_count', 'qed_score'
            ],
            'Structural Complexity': [
                'bertz_complexity', 'balaban_index', 'sp3_fraction'
            ],
            'Natural Product Features': [
                'chiral_centers', 'stereocenters', 'phenol_groups', 
                'hydroxyl_groups', 'sugar_like'
            ],
            'Functional Groups': [
                'carbonyl_groups', 'ether_groups', 'nitrogen_containing', 
                'basic_nitrogen'
            ],
            'Fingerprint Features': [
                col for col in self.get_all_feature_names() 
                if any(fp in col for fp in ['morgan', 'atompair', 'torsion'])
            ]
        }
    
    def get_all_feature_names(self) -> List[str]:
        """Get all possible feature names."""
        sample_features = self.calculate_all_features('CCO')  # Ethanol as test
        return list(sample_features.keys()) if sample_features else []
