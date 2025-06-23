"""
Molecular Feature Engineering for BioPath SHAP Demo

This module provides comprehensive molecular descriptor calculation for natural compounds,
focusing on features relevant to bioactivity prediction and traditional medicine validation.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.ML.Descriptors import MoleculeDescriptors
from scipy.stats import skew, kurtosis

# Suppress RDKit warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='rdkit')
logging.getLogger('rdkit').setLevel(logging.ERROR)


class MolecularFeatureCalculator:
    """
    Advanced molecular feature calculator for natural compound bioactivity prediction.
    
    This class implements a comprehensive suite of molecular descriptors optimized
    for natural products and traditional medicine compounds.
    """
    
    def __init__(self, include_fingerprints: bool = True, fingerprint_radius: int = 2):
        """
        Initialize the molecular feature calculator.
        
        Args:
            include_fingerprints: Whether to include molecular fingerprints
            fingerprint_radius: Radius for Morgan fingerprints (default: 2)
        """
        self.include_fingerprints = include_fingerprints
        self.fingerprint_radius = fingerprint_radius
        self.descriptor_names = self._get_descriptor_names()
        
        # Initialize pharmacophore generator for bioactivity features
        self.pharm_factory = Gobbi_Pharm2D.factory
        
        logging.info(f"MolecularFeatureCalculator initialized with {len(self.descriptor_names)} descriptors")
    
    def _get_descriptor_names(self) -> List[str]:
        """Get all available RDKit descriptor names."""
        descriptor_list = [desc[0] for desc in Descriptors._descList]
        return descriptor_list
    
    def calculate_basic_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate basic molecular descriptors.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of descriptor names and values
        """
        descriptors = {}
        
        try:
            # Basic molecular properties
            descriptors['molecular_weight'] = Descriptors.MolWt(mol)
            descriptors['heavy_atom_count'] = Descriptors.HeavyAtomCount(mol)
            descriptors['ring_count'] = Descriptors.RingCount(mol)
            descriptors['aromatic_rings'] = Descriptors.NumAromaticRings(mol)
            descriptors['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            descriptors['hbd_count'] = Descriptors.NumHDonors(mol)
            descriptors['hba_count'] = Descriptors.NumHAcceptors(mol)
            
            # Lipinski descriptors for drug-likeness
            descriptors['logp'] = Crippen.MolLogP(mol)
            descriptors['tpsa'] = Descriptors.TPSA(mol)
            descriptors['molar_refractivity'] = Crippen.MolMR(mol)
            
            # QED (Quantitative Estimate of Drug-likeness)
            descriptors['qed_score'] = QED.qed(mol)
            
        except Exception as e:
            logging.warning(f"Error calculating basic descriptors: {e}")
            # Return zeros for failed calculations
            descriptors = {key: 0.0 for key in descriptors.keys()}
        
        return descriptors
    
    def calculate_ethnobotanical_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate features specifically relevant to traditional medicine compounds.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of ethnobotanical feature names and values
        """
        features = {}
        
        try:
            # Natural product-likeness indicators
            features['sp3_fraction'] = Descriptors.FractionCsp3(mol)
            features['stereocenters'] = Descriptors.NumAliphaticCarbocycles(mol)
            features['chiral_centers'] = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            
            # Complexity measures relevant to natural products
            features['bertz_complexity'] = Descriptors.BertzCT(mol)
            features['balaban_index'] = Descriptors.BalabanJ(mol)
            features['wiener_index'] = Descriptors.WienerIndex(mol)
            
            # Functional group counts (common in medicinal plants)
            features['phenol_groups'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH][c]')))
            features['carbonyl_groups'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]=[#8]')))
            features['ether_groups'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]-[#8]-[#6]')))
            features['hydroxyl_groups'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]')))
            
            # Glycoside and terpene-like features
            features['sugar_like'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[CH2]([OH])[CH]([OH])')))
            features['isoprene_units'] = self._count_isoprene_units(mol)
            
        except Exception as e:
            logging.warning(f"Error calculating ethnobotanical features: {e}")
            features = {key: 0.0 for key in features.keys()}
        
        return features
    
    def _count_isoprene_units(self, mol: Chem.Mol) -> int:
        """Count potential isoprene units (C5 building blocks) in molecule."""
        try:
            # Simple approximation: count C5 patterns
            isoprene_pattern = Chem.MolFromSmarts('CCCCC')
            if isoprene_pattern:
                return len(mol.GetSubstructMatches(isoprene_pattern))
        except:
            pass
        return 0
    
    def calculate_pharmacophore_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate pharmacophore-based features for bioactivity prediction.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of pharmacophore features
        """
        features = {}
        
        try:
            # Generate 2D pharmacophore fingerprint
            pharm_fp = Generate.Gen2DFingerprint(mol, self.pharm_factory)
            
            # Extract key pharmacophore features
            features['hydrophobic_regions'] = sum(1 for i in range(len(pharm_fp)) if pharm_fp[i] and i < 10)
            features['hbd_regions'] = sum(1 for i in range(10, 20) if i < len(pharm_fp) and pharm_fp[i])
            features['hba_regions'] = sum(1 for i in range(20, 30) if i < len(pharm_fp) and pharm_fp[i])
            features['aromatic_regions'] = sum(1 for i in range(30, 40) if i < len(pharm_fp) and pharm_fp[i])
            
            # Spatial distribution of pharmacophores
            features['pharmacophore_density'] = sum(pharm_fp) / len(pharm_fp) if len(pharm_fp) > 0 else 0
            
        except Exception as e:
            logging.warning(f"Error calculating pharmacophore features: {e}")
            features = {key: 0.0 for key in features.keys()}
        
        return features
    
    def calculate_fingerprint_features(self, mol: Chem.Mol) -> Dict[str, int]:
        """
        Calculate molecular fingerprint features.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of fingerprint features
        """
        features = {}
        
        if not self.include_fingerprints:
            return features
        
        try:
            # Morgan fingerprints (circular fingerprints)
            morgan_fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=self.fingerprint_radius, nBits=1024
            )
            
            # Convert to dictionary with bit positions
            morgan_bits = morgan_fp.ToBitString()
            features.update({f'morgan_bit_{i}': int(bit) for i, bit in enumerate(morgan_bits[:50])})  # First 50 bits for demo
            
            # Atom pair fingerprints
            atom_pairs = Pairs.GetAtomPairFingerprintAsBitVect(mol)
            ap_bits = atom_pairs.ToBitString()
            features.update({f'atompair_bit_{i}': int(bit) for i, bit in enumerate(ap_bits[:25])})  # First 25 bits
            
        except Exception as e:
            logging.warning(f"Error calculating fingerprint features: {e}")
        
        return features
    
    def calculate_all_features(self, smiles: str) -> Optional[Dict[str, Union[float, int]]]:
        """
        Calculate all molecular features for a given SMILES string.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Dictionary containing all calculated features, or None if invalid SMILES
        """
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logging.warning(f"Invalid SMILES: {smiles}")
                return None
            
            # Add hydrogens for accurate descriptor calculation
            mol = Chem.AddHs(mol)
            
            # Calculate all feature categories
            features = {}
            features.update(self.calculate_basic_descriptors(mol))
            features.update(self.calculate_ethnobotanical_features(mol))
            features.update(self.calculate_pharmacophore_features(mol))
            features.update(self.calculate_fingerprint_features(mol))
            
            # Add SMILES for reference
            features['smiles'] = smiles
            
            return features
            
        except Exception as e:
            logging.error(f"Error processing SMILES {smiles}: {e}")
            return None
    
    def process_batch(self, smiles_list: List[str], show_progress: bool = True) -> pd.DataFrame:
        """
        Process a batch of SMILES strings and return a feature DataFrame.
        
        Args:
            smiles_list: List of SMILES strings
            show_progress: Whether to show progress updates
            
        Returns:
            DataFrame with calculated features for each valid molecule
        """
        results = []
        failed_count = 0
        
        for i, smiles in enumerate(smiles_list):
            if show_progress and (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(smiles_list)} molecules...")
            
            features = self.calculate_all_features(smiles)
            if features is not None:
                results.append(features)
            else:
                failed_count += 1
        
        if failed_count > 0:
            logging.warning(f"Failed to process {failed_count} molecules")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Fill any NaN values with zeros
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        logging.info(f"Successfully processed {len(df)} molecules with {len(df.columns)} features")
        
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Group features by chemical relevance for SHAP analysis.
        
        Returns:
            Dictionary mapping group names to feature lists
        """
        groups = {
            'Basic Properties': ['molecular_weight', 'heavy_atom_count', 'ring_count'],
            'Drug-likeness': ['logp', 'tpsa', 'hbd_count', 'hba_count', 'qed_score'],
            'Structural Complexity': ['bertz_complexity', 'balaban_index', 'wiener_index'],
            'Natural Product Features': ['sp3_fraction', 'chiral_centers', 'stereocenters'],
            'Functional Groups': ['phenol_groups', 'carbonyl_groups', 'hydroxyl_groups'],
            'Pharmacophores': ['hydrophobic_regions', 'hbd_regions', 'hba_regions', 'aromatic_regions'],
            'Fingerprint Features': [col for col in self.get_all_feature_names() if 'bit_' in col]
        }
        
        return groups
    
    def get_all_feature_names(self) -> List[str]:
        """Get all possible feature names that could be calculated."""
        # This would return all feature names - simplified for demo
        sample_features = self.calculate_all_features('CCO')  # Simple molecule for testing
        return list(sample_features.keys()) if sample_features else []


# Utility functions for external use
def validate_smiles(smiles: str) -> bool:
    """
    Validate if a SMILES string is chemically valid.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def normalize_features(df: pd.DataFrame, exclude_columns: List[str] = None) -> pd.DataFrame:
    """
    Normalize feature values for machine learning.
    
    Args:
        df: DataFrame with molecular features
        exclude_columns: Columns to exclude from normalization
        
    Returns:
        DataFrame with normalized features
    """
    exclude_columns = exclude_columns or ['smiles']
    
    # Select numeric columns for normalization
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
    
    df_normalized = df.copy()
    
    # Z-score normalization
    for col in numeric_columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            df_normalized[col] = (df[col] - mean_val) / std_val
        else:
            df_normalized[col] = 0
    
    return df_normalized
