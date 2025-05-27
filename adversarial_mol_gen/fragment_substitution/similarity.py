"""
Functions for calculating molecular similarity and analyzing scaffolds.
"""

from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_molecule_scaffold(mol: Chem.Mol) -> Chem.Mol:
    """Extract the Murcko scaffold from a molecule."""
    return MurckoScaffold.GetScaffoldForMol(mol)


def calculate_similarity(mol1: Chem.Mol, mol2: Chem.Mol, method: str = "morgan") -> float:
    """
    Calculate similarity between two molecules.

    Args:
        mol1: First molecule
        mol2: Second molecule
        method: Fingerprint method ('morgan', 'maccs', 'topological')

    Returns:
        Tanimoto similarity score (0-1)
    """
    if method == "morgan":
        # Use MorganGenerator for fingerprint generation
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp1 = morgan_gen.GetFingerprint(mol1)
        fp2 = morgan_gen.GetFingerprint(mol2)
    elif method == "maccs":
        # Use MACCS keys fingerprint
        from rdkit.Chem import MACCSkeys
        fp1 = MACCSkeys.GenMACCSKeys(mol1)
        fp2 = MACCSkeys.GenMACCSKeys(mol2)
    else:  # default to topological
        # Use RDKit's topological fingerprint
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def has_similar_size(mol1: Chem.Mol, mol2: Chem.Mol, max_diff_ratio: float = 0.3) -> bool:
    """Check if two molecules have similar sizes (within max_diff_ratio)."""
    num_atoms1 = mol1.GetNumAtoms()
    num_atoms2 = mol2.GetNumAtoms()
    
    if num_atoms1 == 0 or num_atoms2 == 0:
        return False
        
    size_ratio = abs(num_atoms1 - num_atoms2) / max(num_atoms1, num_atoms2)
    return size_ratio <= max_diff_ratio


def has_same_scaffold(mol1: Chem.Mol, mol2: Chem.Mol) -> bool:
    """Check if two molecules have the same Murcko scaffold."""
    try:
        scaffold1 = MurckoScaffold.MurckoScaffoldSmiles(Chem.MolToSmiles(mol1))
        scaffold2 = MurckoScaffold.MurckoScaffoldSmiles(Chem.MolToSmiles(mol2))
        return scaffold1 == scaffold2
    except:
        return False