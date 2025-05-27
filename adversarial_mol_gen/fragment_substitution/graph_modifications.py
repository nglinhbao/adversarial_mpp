"""
Functions for modifying molecular graphs (bonds, atoms, etc.)
"""

from typing import List, Tuple, Set, Optional

from rdkit import Chem


def substitute_bond(mol: Chem.Mol, bond_idx: int, new_bond_type: Chem.BondType) -> Optional[Chem.Mol]:
    """
    Substitute a bond with a new bond type.
    
    Args:
        mol: Original molecule
        bond_idx: Index of the bond to modify
        new_bond_type: New bond type (Chem.BondType.SINGLE, DOUBLE, TRIPLE, etc.)
        
    Returns:
        Modified molecule or None if invalid
    """
    if bond_idx >= mol.GetNumBonds():
        return None
    
    # Create a modifiable molecule
    rw_mol = Chem.RWMol(mol)
    bond = mol.GetBondWithIdx(bond_idx)
    
    # Get the atoms connected by this bond
    begin_atom_idx = bond.GetBeginAtomIdx()
    end_atom_idx = bond.GetEndAtomIdx()
    
    # Remove the old bond and add the new one
    rw_mol.RemoveBond(begin_atom_idx, end_atom_idx)
    rw_mol.AddBond(begin_atom_idx, end_atom_idx, new_bond_type)
    
    try:
        # Convert to molecule and validate
        new_mol = rw_mol.GetMol()
        Chem.SanitizeMol(new_mol)
        return new_mol
    except Exception:
        return None


def get_modifiable_bonds(mol: Chem.Mol, scaffold: Optional[Chem.Mol] = None) -> List[Tuple[int, Set[Chem.BondType]]]:
    """
    Get indices of bonds that can be modified and their allowed new bond types.
    
    Args:
        mol: Molecule to analyze
        scaffold: Optional scaffold to protect
        
    Returns:
        List of (bond_idx, set of allowed bond types)
    """
    modifiable_bonds = []
    
    # Find bonds in scaffold (if provided)
    scaffold_bonds = set()
    if scaffold:
        scaffold_matches = mol.GetSubstructMatches(scaffold)
        for match in scaffold_matches:
            for bond in scaffold.GetBonds():
                begin_idx = match[bond.GetBeginAtomIdx()]
                end_idx = match[bond.GetEndAtomIdx()]
                # Find this bond in the original molecule
                bond_idx = mol.GetBondBetweenAtoms(begin_idx, end_idx).GetIdx()
                scaffold_bonds.add(bond_idx)
    
    # Analyze each bond
    for bond_idx in range(mol.GetNumBonds()):
        # Skip scaffold bonds if protecting scaffold
        if bond_idx in scaffold_bonds:
            continue
            
        bond = mol.GetBondWithIdx(bond_idx)
        begin_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
        end_atom = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
        
        # Determine what bond types are chemically feasible
        allowed_types = set()
        current_type = bond.GetBondType()
        
        # Add allowed bond types based on connected atoms
        if current_type == Chem.BondType.SINGLE:
            # Single to double is often possible
            allowed_types.add(Chem.BondType.DOUBLE)
            
            # Check if atoms can support triple bond (C≡C, C≡N, N≡N)
            begin_sym = begin_atom.GetSymbol()
            end_sym = end_atom.GetSymbol()
            if (begin_sym in ['C', 'N'] and end_sym in ['C', 'N']):
                allowed_types.add(Chem.BondType.TRIPLE)
                
        elif current_type == Chem.BondType.DOUBLE:
            # Double to single is usually possible
            allowed_types.add(Chem.BondType.SINGLE)
            
            # Double to triple for specific atom types
            begin_sym = begin_atom.GetSymbol()
            end_sym = end_atom.GetSymbol()
            if (begin_sym in ['C', 'N'] and end_sym in ['C', 'N']):
                allowed_types.add(Chem.BondType.TRIPLE)
                
        elif current_type == Chem.BondType.TRIPLE:
            # Triple to single or double
            allowed_types.add(Chem.BondType.SINGLE)
            allowed_types.add(Chem.BondType.DOUBLE)
            
        elif current_type == Chem.BondType.AROMATIC:
            # Aromatic bonds have limited modification options
            # Can sometimes be converted to single/double
            allowed_types.add(Chem.BondType.SINGLE)
            allowed_types.add(Chem.BondType.DOUBLE)
        
        # Remove current type from allowed types
        if current_type in allowed_types:
            allowed_types.remove(current_type)
            
        if allowed_types:
            modifiable_bonds.append((bond_idx, allowed_types))
            
    return modifiable_bonds


def add_atom_to_molecule(mol: Chem.Mol, atom_idx: int, new_atom_symbol: str, 
                        bond_type: Chem.BondType = Chem.BondType.SINGLE) -> Optional[Chem.Mol]:
    """
    Add a new atom to the molecule connected to an existing atom.
    
    Args:
        mol: Original molecule
        atom_idx: Index of the atom to connect to
        new_atom_symbol: Symbol of the new atom to add
        bond_type: Type of bond to create
        
    Returns:
        Modified molecule or None if invalid
    """
    if atom_idx >= mol.GetNumAtoms():
        return None
        
    # Create a modifiable molecule
    rw_mol = Chem.RWMol(mol)
    
    # Get the atom to connect to
    atom = rw_mol.GetAtomWithIdx(atom_idx)
    
    # Check if the atom can accept another bond
    if atom.GetExplicitValence() + bond_type.real >= 5:
        return None  # Atom would exceed typical valence
    
    # Add the new atom
    new_atom_idx = rw_mol.AddAtom(Chem.Atom(new_atom_symbol))
    
    # Add the bond
    rw_mol.AddBond(atom_idx, new_atom_idx, bond_type)
    
    try:
        # Convert to molecule and validate
        new_mol = rw_mol.GetMol()
        Chem.SanitizeMol(new_mol)
        return new_mol
    except Exception:
        return None


def get_atom_addition_candidates(mol: Chem.Mol, scaffold: Optional[Chem.Mol] = None) -> List[Tuple[int, List[str]]]:
    """
    Find atoms in the molecule where new atoms can be added.
    
    Args:
        mol: Molecule to analyze
        scaffold: Optional scaffold to protect
        
    Returns:
        List of (atom_idx, list of possible atoms to add)
    """
    addition_candidates = []
    
    # Determine scaffold atoms (if any)
    scaffold_atoms = set()
    if scaffold:
        scaffold_matches = mol.GetSubstructMatches(scaffold)
        for match in scaffold_matches:
            for idx in match:
                scaffold_atoms.add(idx)
    
    # Common atoms to add
    common_atoms = ["C", "N", "O", "F", "Cl", "Br", "S"]
    
    # Check each atom
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        
        # Skip atoms in the scaffold if we're preserving it
        if atom_idx in scaffold_atoms and scaffold:
            continue
            
        # Check if this atom can accept more bonds
        explicit_valence = atom.GetExplicitValence()
        implicit_hs = atom.GetNumImplicitHs()
        
        if implicit_hs > 0 or explicit_valence < 3:
            # This atom can likely accept a new connection
            # The atoms we can add depend on the current atom
            symbol = atom.GetSymbol()
            
            if symbol in ["C", "O", "N", "S"]:
                # These can connect to most common atoms
                addition_candidates.append((atom_idx, common_atoms))
            elif symbol in ["F", "Cl", "Br", "I"]:
                # Halogens typically don't accept additional bonds
                pass
            else:
                # For other atoms, just try carbon
                addition_candidates.append((atom_idx, ["C"]))
    
    return addition_candidates


def add_bond_between_atoms(mol: Chem.Mol, atom1_idx: int, atom2_idx: int, 
                          bond_type: Chem.BondType = Chem.BondType.SINGLE) -> Optional[Chem.Mol]:
    """
    Add a new bond between two existing atoms in the molecule.
    
    Args:
        mol: Original molecule
        atom1_idx: Index of first atom
        atom2_idx: Index of second atom
        bond_type: Type of bond to create
        
    Returns:
        Modified molecule or None if invalid
    """
    if atom1_idx >= mol.GetNumAtoms() or atom2_idx >= mol.GetNumAtoms():
        return None
    
    # Check if bond already exists
    if mol.GetBondBetweenAtoms(atom1_idx, atom2_idx):
        return None
    
    # Create a modifiable molecule
    rw_mol = Chem.RWMol(mol)
    
    # Add the bond
    rw_mol.AddBond(atom1_idx, atom2_idx, bond_type)
    
    try:
        # Convert to molecule and validate
        new_mol = rw_mol.GetMol()
        Chem.SanitizeMol(new_mol)
        return new_mol
    except Exception:
        return None


def find_potential_bond_additions(mol: Chem.Mol, scaffold: Optional[Chem.Mol] = None, 
                                 max_distance: int = 4) -> List[Tuple[int, int]]:
    """
    Find pairs of atoms that could be connected with a new bond.
    
    Args:
        mol: Molecule to analyze
        scaffold: Optional scaffold to protect
        max_distance: Maximum topological distance between atoms to consider
        
    Returns:
        List of (atom1_idx, atom2_idx) pairs that could be connected
    """
    # Get the distance matrix
    distance_matrix = Chem.GetDistanceMatrix(mol)
    
    # Determine scaffold atoms (if any)
    scaffold_atoms = set()
    if scaffold:
        scaffold_matches = mol.GetSubstructMatches(scaffold)
        for match in scaffold_matches:
            scaffold_atoms.update(match)
    
    # Find candidate pairs
    candidates = []
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            # Check if both atoms are in scaffold
            if i in scaffold_atoms and j in scaffold_atoms and scaffold:
                continue
            
            # Check if atoms are already bonded
            if mol.GetBondBetweenAtoms(i, j):
                continue
                
            # Check if atoms are close enough topologically
            distance = distance_matrix[i][j]
            if distance >= 2 and distance <= max_distance:
                # Check if adding a bond would create a valid structure
                atom_i = mol.GetAtomWithIdx(i)
                atom_j = mol.GetAtomWithIdx(j)
                
                # Skip if either atom is a halogen or has high valence
                if atom_i.GetSymbol() in ["F", "Cl", "Br", "I"] or atom_j.GetSymbol() in ["F", "Cl", "Br", "I"]:
                    continue
                
                # Skip if either atom would exceed typical valence
                if atom_i.GetExplicitValence() >= 3 or atom_j.GetExplicitValence() >= 3:
                    continue
                    
                candidates.append((i, j))
    
    return candidates