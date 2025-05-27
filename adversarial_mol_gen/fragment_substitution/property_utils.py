"""
Functions for calculating and comparing molecular properties.
"""

from typing import Dict
import math

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


def calculate_properties(mol: Chem.Mol) -> Dict[str, float]:
    """
    Calculate key molecular properties.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        Dictionary of properties
    """
    props = {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
        "HeavyAtoms": Descriptors.HeavyAtomCount(mol),
        "Rings": Descriptors.RingCount(mol),
        "AromaticRings": Descriptors.NumAromaticRings(mol)
    }
    return props


def property_similarity(props1: Dict[str, float], props2: Dict[str, float]) -> float:
    """
    Calculate similarity between two sets of molecular properties.
    
    Args:
        props1: First molecule's properties
        props2: Second molecule's properties
        
    Returns:
        Property similarity score (0-1)
    """
    # Define weights for different properties
    weights = {
        "MW": 0.15,
        "LogP": 0.2,
        "TPSA": 0.2,
        "HBD": 0.1,
        "HBA": 0.1,
        "RotBonds": 0.05,
        "HeavyAtoms": 0.1,
        "Rings": 0.05,
        "AromaticRings": 0.05
    }
    
    # Calculate normalized differences
    total_diff = 0.0
    total_weight = 0.0
    
    for prop, weight in weights.items():
        if prop in props1 and prop in props2:
            # Special handling for different properties
            if prop in ["HBD", "HBA", "Rings", "AromaticRings"]:
                # For integer properties, use exact difference
                diff = abs(props1[prop] - props2[prop])
                # Normalize to 0-1 range (allow max difference of 2)
                norm_diff = min(diff / 2.0, 1.0)
            elif prop in ["MW"]:
                # For molecular weight, normalize by percentage
                diff = abs(props1[prop] - props2[prop]) / max(props1[prop], 1.0)
                norm_diff = min(diff, 1.0)
            elif prop in ["LogP"]:
                # For LogP, allow difference of 1 unit
                diff = abs(props1[prop] - props2[prop])
                norm_diff = min(diff / 1.0, 1.0)
            elif prop in ["TPSA"]:
                # For TPSA, allow difference of 10 units
                diff = abs(props1[prop] - props2[prop])
                norm_diff = min(diff / 10.0, 1.0)
            else:
                # Default normalization
                diff = abs(props1[prop] - props2[prop]) / max(props1[prop], 1.0)
                norm_diff = min(diff, 1.0)
                
            total_diff += norm_diff * weight
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
        
    # Convert to similarity (1 - normalized_difference)
    return 1.0 - (total_diff / total_weight)


def has_similar_properties(mol1: Chem.Mol, mol2: Chem.Mol, min_similarity: float = 0.8) -> bool:
    """
    Check if two molecules have similar properties.
    
    Args:
        mol1: First molecule
        mol2: Second molecule
        min_similarity: Minimum property similarity threshold
        
    Returns:
        True if properties are similar, False otherwise
    """
    props1 = calculate_properties(mol1)
    props2 = calculate_properties(mol2)
    
    sim = property_similarity(props1, props2)
    return sim >= min_similarity


def modify_functional_group(
    mol: Chem.Mol, 
    smarts: str, 
    replacements: list,
    preserve_scaffold: bool = True,
    original_properties: Dict[str, float] = None,
    min_property_similarity: float = 0.8
) -> list:
    """
    Modify a functional group in a molecule while preserving its structure and properties.
    
    Args:
        mol: Original molecule
        smarts: SMARTS pattern for the functional group
        replacements: List of SMARTS patterns to replace with
        preserve_scaffold: Whether to enforce scaffold preservation
        original_properties: Properties of the original molecule to preserve
        min_property_similarity: Minimum property similarity threshold
        
    Returns:
        List of modified molecules
    """
    from .similarity import get_molecule_scaffold, calculate_similarity
    
    results = []
    pattern = Chem.MolFromSmarts(smarts)
    
    if not pattern:
        return results
        
    # Find all matches of the pattern
    matches = mol.GetSubstructMatches(pattern)
    
    if not matches:
        return results
        
    # Get the scaffold of the original molecule
    scaffold = get_molecule_scaffold(mol) if preserve_scaffold else None
    
    # Try each replacement
    for repl_smarts in replacements:
        repl = Chem.MolFromSmarts(repl_smarts)
        if not repl:
            continue
            
        try:
            # Replace the substructure
            new_mols = Chem.ReplaceSubstructs(mol, pattern, repl)
            
            for new_mol in new_mols:
                try:
                    # Convert to canonical form
                    new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
                    
                    if new_mol:
                        # Validate molecule
                        Chem.SanitizeMol(new_mol)
                        
                        # Check if scaffold is preserved (if required)
                        if preserve_scaffold and scaffold:
                            new_scaffold = get_molecule_scaffold(new_mol)
                            if not new_scaffold or not calculate_similarity(scaffold, new_scaffold, "morgan") > 0.7:
                                continue
                        
                        # Check property similarity if original properties provided
                        if original_properties:
                            new_props = calculate_properties(new_mol)
                            if property_similarity(original_properties, new_props) < min_property_similarity:
                                continue
                                
                        # Add to results if it's a valid molecule
                        results.append(new_mol)
                except Exception:
                    continue
        except Exception:
            continue
            
    return results