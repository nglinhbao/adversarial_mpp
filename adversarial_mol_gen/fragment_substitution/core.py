"""
Core FragmentSubstitution class for generating adversarial molecules.
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Iterable, Optional, Tuple, Set, Any
import random
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import BRICS

# Import from our own modules
from .fragment_utils import build_runtime_tables
from .similarity import calculate_similarity, get_molecule_scaffold, has_similar_size, has_same_scaffold
from .property_utils import calculate_properties, property_similarity, modify_functional_group
from .graph_modifications import (
    substitute_bond, get_modifiable_bonds, 
    add_atom_to_molecule, get_atom_addition_candidates,
    add_bond_between_atoms, find_potential_bond_additions
)


# Generic replacements to use as fallback
_GENERIC_REPL = ["C", "N", "O", "S", "F", "Cl"]

class FragmentSubstitution:
    """
    Enhanced adversarial analogue generator with property preservation.
    
    Updated: 2025-05-27 by nglinhbao
    """

    def __init__(
        self,
        fragment_library: Optional[Dict[str, Sequence[str]]] = None,
        functional_group_rules: Optional[Dict[str, Sequence[str]]] = None,
        fragment_type_map: Optional[Dict[str, Sequence[str]]] = None,
        custom_rules: Optional[Dict[str, Dict[str, str]]] = None,
        max_brics_repl_per_type: int = 3,
        min_similarity: float = 0.4,
        preserve_scaffold: bool = True,
        max_size_diff_ratio: float = 0.3,
        # Graph modification options
        enable_bond_substitution: bool = True,
        enable_atom_addition: bool = True,
        enable_bond_addition: bool = True,
        max_modifications_per_molecule: int = 2,
        # Property preservation options
        min_property_similarity: float = 0.8,
        preserve_hba_hbd: bool = True,
        preserve_logp: bool = True,
        max_logp_diff: float = 5.0,
        preserve_tpsa: bool = True,
        max_tpsa_diff: float = 15.0
    ):
        """
        Initialize the fragment substitution with similarity and property constraints.
        
        Args:
            fragment_library: Dictionary mapping fragment types to lists of SMILES
            functional_group_rules: Dictionary mapping patterns to replacements
            fragment_type_map: Dictionary mapping patterns to fragment types
            custom_rules: Custom substitution rules
            max_brics_repl_per_type: Maximum number of BRICS replacements per type
            min_similarity: Minimum similarity threshold (0-1)
            preserve_scaffold: Whether to preserve the molecular scaffold
            max_size_diff_ratio: Maximum allowed size difference ratio (0-1)
            enable_bond_substitution: Whether to enable bond substitution
            enable_atom_addition: Whether to enable adding new atoms
            enable_bond_addition: Whether to enable adding new bonds
            max_modifications_per_molecule: Maximum number of graph modifications per molecule
            min_property_similarity: Minimum property similarity threshold
            preserve_hba_hbd: Strictly preserve H-bond donor and acceptor counts
            preserve_logp: Strictly preserve LogP within specified range
            max_logp_diff: Maximum allowed LogP difference
            preserve_tpsa: Strictly preserve TPSA within specified range
            max_tpsa_diff: Maximum allowed TPSA difference
        """
        if fragment_library is None or functional_group_rules is None or fragment_type_map is None:
            lib, fg_rules, ftype_map = build_runtime_tables()
            fragment_library = fragment_library or lib
            functional_group_rules = functional_group_rules or fg_rules
            fragment_type_map = fragment_type_map or ftype_map

        self.fragment_library = fragment_library
        self.functional_group_rules = functional_group_rules
        self.fragment_type_map = fragment_type_map
        self.custom_rules = custom_rules or {}
        self.max_brics_repl_per_type = max_brics_repl_per_type
        
        # Similarity constraints
        self.min_similarity = min_similarity
        self.preserve_scaffold = preserve_scaffold
        self.max_size_diff_ratio = max_size_diff_ratio
        
        # Graph modification options
        self.enable_bond_substitution = enable_bond_substitution
        self.enable_atom_addition = enable_atom_addition
        self.enable_bond_addition = enable_bond_addition
        self.max_modifications_per_molecule = max_modifications_per_molecule
        
        # Property preservation options
        self.min_property_similarity = min_property_similarity
        self.preserve_hba_hbd = preserve_hba_hbd
        self.preserve_logp = preserve_logp
        self.max_logp_diff = max_logp_diff
        self.preserve_tpsa = preserve_tpsa
        self.max_tpsa_diff = max_tpsa_diff
        
        print(f"[INFO] Initialized FragmentSubstitution with:")
        print(f"  - min_similarity: {min_similarity}")
        print(f"  - preserve_scaffold: {preserve_scaffold}")
        print(f"  - min_property_similarity: {min_property_similarity}")
        print(f"  - preserve_hba_hbd: {preserve_hba_hbd}")
        print(f"  - preserve_logp: {preserve_logp} (max diff: {max_logp_diff})")
        print(f"  - preserve_tpsa: {preserve_tpsa} (max diff: {max_tpsa_diff})")

    # ───────────────────────── public API ──────────────────────────
    def generate_substitutions(
        self,
        mol: Chem.Mol,
        important_fragments: Iterable[str],
        n_candidates: int = 10,
    ) -> List[Chem.Mol]:
        """
        Generate structurally similar adversarial molecules with property preservation.
        
        Args:
            mol: Original molecule
            important_fragments: List of important fragments to target
            n_candidates: Maximum number of candidates to generate
            
        Returns:
            List of adversarial molecules that maintain structural similarity and properties
        """
        important_fragments = list(important_fragments or [])
        mol_smiles = Chem.MolToSmiles(mol)
        print(f"[INFO] Original: {mol_smiles}")
        
        # Calculate original molecule properties
        original_props = calculate_properties(mol)
        print(f"[INFO] Original properties: MW={original_props['MW']:.2f}, LogP={original_props['LogP']:.2f}, "
              f"TPSA={original_props['TPSA']:.2f}, HBD={original_props['HBD']}, HBA={original_props['HBA']}")
        
        # Extract scaffold information
        try:
            scaffold = get_molecule_scaffold(mol) if self.preserve_scaffold else None
            scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else "N/A"
            print(f"[INFO] Scaffold: {scaffold_smiles}")
        except Exception as e:
            print(f"[WARN] Couldn't extract scaffold: {str(e)}")
            scaffold = None

        # Track all candidates and their similarity scores
        cands: List[Tuple[Chem.Mol, float, float]] = []  # (molecule, struct_similarity, prop_similarity)

        # ------------------ Method 1: targeted fragment swap ----------------
        print("[INFO] Method 1: Targeted fragment swap with property constraints")
        for frag in important_fragments:
            if not frag or frag not in mol_smiles:
                continue
                
            print(f"  - Processing fragment: {frag}")
            repls = next(
                (r for k, r in self.fragment_type_map.items() if k in frag),
                _GENERIC_REPL,
            )
            
            for r in repls:
                try:
                    smi = mol_smiles.replace(frag, r, 1)
                    m = Chem.MolFromSmiles(smi)
                    
                    if m and Chem.MolToSmiles(m) != mol_smiles:
                        try:
                            Chem.SanitizeMol(m)
                            
                            # Check size similarity
                            if not has_similar_size(mol, m, self.max_size_diff_ratio):
                                continue
                                
                            # Calculate structural similarity
                            struct_sim = calculate_similarity(mol, m, "morgan")
                            
                            # Check scaffold preservation if required
                            if self.preserve_scaffold and scaffold:
                                if not has_same_scaffold(mol, m):
                                    continue
                            
                            # Check property similarity
                            new_props = calculate_properties(m)
                            prop_sim = property_similarity(original_props, new_props)
                            
                            # Apply property constraints
                            if self.preserve_hba_hbd:
                                # Allow at most 1 difference in donors/acceptors
                                if abs(original_props["HBD"] - new_props["HBD"]) > 1 or \
                                   abs(original_props["HBA"] - new_props["HBA"]) > 1:
                                    continue
                                    
                            if self.preserve_logp:
                                if abs(original_props["LogP"] - new_props["LogP"]) > self.max_logp_diff:
                                    continue
                                    
                            if self.preserve_tpsa:
                                if abs(original_props["TPSA"] - new_props["TPSA"]) > self.max_tpsa_diff:
                                    continue
                            
                            # Add if passes all thresholds
                            if struct_sim >= self.min_similarity and prop_sim >= self.min_property_similarity:
                                print(f"    Created valid molecule (struct_sim={struct_sim:.2f}, prop_sim={prop_sim:.2f}): {frag} → {r}")
                                print(f"    New properties: MW={new_props['MW']:.2f}, LogP={new_props['LogP']:.2f}, "
                                      f"TPSA={new_props['TPSA']:.2f}, HBD={new_props['HBD']}, HBA={new_props['HBA']}")
                                cands.append((m, struct_sim, prop_sim))
                            
                        except Exception as e:
                            print(f"    Error: {str(e)}")
                except Exception:
                    pass

        # --------------- Method 2: functional-group replacement ------------
        print("[INFO] Method 2: Functional group replacement with property preservation")
        for pat, repls in self.functional_group_rules.items():
            if pat not in mol_smiles:
                continue
                
            for r in repls:
                try:
                    # Try to replace with more controlled approach
                    pattern = Chem.MolFromSmarts(pat)
                    if pattern and mol.HasSubstructMatch(pattern):
                        modified_mols = modify_functional_group(
                            mol, pat, [r], 
                            preserve_scaffold=self.preserve_scaffold,
                            original_properties=original_props,
                            min_property_similarity=self.min_property_similarity
                        )
                        
                        for m in modified_mols:
                            if m and Chem.MolToSmiles(m) != mol_smiles:
                                # Calculate structural similarity
                                struct_sim = calculate_similarity(mol, m, "morgan")
                                
                                # Calculate property similarity
                                new_props = calculate_properties(m)
                                prop_sim = property_similarity(original_props, new_props)
                                
                                # Apply additional property constraints
                                if self.preserve_hba_hbd:
                                    if abs(original_props["HBD"] - new_props["HBD"]) > 1 or \
                                       abs(original_props["HBA"] - new_props["HBA"]) > 1:
                                        continue
                                        
                                if self.preserve_logp:
                                    if abs(original_props["LogP"] - new_props["LogP"]) > self.max_logp_diff:
                                        continue
                                        
                                if self.preserve_tpsa:
                                    if abs(original_props["TPSA"] - new_props["TPSA"]) > self.max_tpsa_diff:
                                        continue
                                
                                # Add if passes all thresholds
                                if struct_sim >= self.min_similarity and prop_sim >= self.min_property_similarity:
                                    print(f"    Created valid molecule (struct_sim={struct_sim:.2f}, prop_sim={prop_sim:.2f}): {pat} → {r}")
                                    cands.append((m, struct_sim, prop_sim))
                    else:
                        # Fallback to simple replacement
                        smi = mol_smiles.replace(pat, r, 1)
                        m = Chem.MolFromSmiles(smi)
                        
                        if m and Chem.MolToSmiles(m) != mol_smiles:
                            Chem.SanitizeMol(m)
                            
                            # Calculate structural similarity
                            struct_sim = calculate_similarity(mol, m, "morgan")
                            
                            # Check scaffold preservation if required
                            if self.preserve_scaffold and scaffold:
                                if not has_same_scaffold(mol, m):
                                    continue
                            
                            # Calculate property similarity
                            new_props = calculate_properties(m)
                            prop_sim = property_similarity(original_props, new_props)
                            
                            # Apply additional property constraints
                            if self.preserve_hba_hbd:
                                if abs(original_props["HBD"] - new_props["HBD"]) > 1 or \
                                   abs(original_props["HBA"] - new_props["HBA"]) > 1:
                                    continue
                                    
                            if self.preserve_logp:
                                if abs(original_props["LogP"] - new_props["LogP"]) > self.max_logp_diff:
                                    continue
                                    
                            if self.preserve_tpsa:
                                if abs(original_props["TPSA"] - new_props["TPSA"]) > self.max_tpsa_diff:
                                    continue
                            
                            # Add if passes all thresholds
                            if struct_sim >= self.min_similarity and prop_sim >= self.min_property_similarity:
                                print(f"    Created valid molecule (struct_sim={struct_sim:.2f}, prop_sim={prop_sim:.2f}): {pat} → {r}")
                                cands.append((m, struct_sim, prop_sim))
                except Exception:
                    pass

        # ----------------- Method 3: BRICS recombination -------------------
        print("[INFO] Method 3: BRICS recombination with property preservation")
        try:
            frags = list(BRICS.BRICSDecompose(mol, keepNonLeafNodes=True))
        except Exception:
            frags = []
            
        if len(frags) > 1:
            # Find BRICS fragments that contain important fragments
            prio = [
                i for i, f in enumerate(frags)
                if any(imp and imp in f for imp in important_fragments)
            ] or list(range(min(3, len(frags))))
            
            # Map to identify fragments that are part of the scaffold
            scaffold_frags = set()
            if scaffold:
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                for i, frag in enumerate(frags):
                    if frag in scaffold_smiles:
                        scaffold_frags.add(i)
            
            for idx in prio:
                # Skip modification of scaffold fragments if preserving scaffold
                if self.preserve_scaffold and idx in scaffold_frags:
                    continue
                    
                for lib in self.fragment_library.values():
                    for repl in lib[: self.max_brics_repl_per_type]:
                        new_frags = frags.copy()
                        new_frags[idx] = repl
                        parts = [Chem.MolFromSmiles(x) for x in new_frags if x]
                        if len(parts) < 2:
                            continue
                        try:
                            m = BRICS.BRICSBuild(parts)
                            if m and Chem.MolToSmiles(m) != mol_smiles:
                                Chem.SanitizeMol(m)
                                
                                # Calculate structural similarity
                                struct_sim = calculate_similarity(mol, m, "morgan")
                                
                                # Check size similarity
                                if not has_similar_size(mol, m, self.max_size_diff_ratio):
                                    continue
                                    
                                # Check scaffold preservation if required
                                if self.preserve_scaffold and scaffold:
                                    m_scaffold = get_molecule_scaffold(m)
                                    if not m_scaffold:
                                        continue
                                    scaffold_sim = calculate_similarity(scaffold, m_scaffold, "morgan")
                                    if scaffold_sim < 0.7:
                                        continue
                                
                                # Calculate property similarity
                                new_props = calculate_properties(m)
                                prop_sim = property_similarity(original_props, new_props)
                                
                                # Apply additional property constraints
                                if self.preserve_hba_hbd:
                                    if abs(original_props["HBD"] - new_props["HBD"]) > 1 or \
                                       abs(original_props["HBA"] - new_props["HBA"]) > 1:
                                        continue
                                        
                                if self.preserve_logp:
                                    if abs(original_props["LogP"] - new_props["LogP"]) > self.max_logp_diff:
                                        continue
                                        
                                if self.preserve_tpsa:
                                    if abs(original_props["TPSA"] - new_props["TPSA"]) > self.max_tpsa_diff:
                                        continue
                                
                                # Add if passes all thresholds
                                if struct_sim >= self.min_similarity and prop_sim >= self.min_property_similarity:
                                    print(f"    Created valid molecule via BRICS (struct_sim={struct_sim:.2f}, prop_sim={prop_sim:.2f})")
                                    cands.append((m, struct_sim, prop_sim))
                        except Exception:
                            pass

        # ------------------- Method 4: atom-level mutation -----------------
        print("[INFO] Method 4: Atom-level mutation with property preservation")
        atom_mutation_table = {
            "O": ["N", "S", "F"],
            "N": ["O", "S", "P"],
            "C": ["N", "O", "S"],
            "S": ["O", "N", "P"],
            "F": ["Cl", "Br"],
            "Cl": ["F", "Br"],
            "Br": ["Cl", "I"],
            "I": ["Br", "Cl"],
        }
        
        # Find atoms that should not be modified if preserving scaffold
        protected_atoms = set()
        if self.preserve_scaffold and scaffold:
            scaffold_matches = mol.GetSubstructMatches(scaffold)
            for match in scaffold_matches:
                for atom_idx in match:
                    # Allow terminal atoms of scaffold to be modified
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetDegree() > 1:  # Not a terminal atom
                        protected_atoms.add(atom_idx)
        
        priority_atoms = self._priority_atoms(mol, important_fragments)
        pt = Chem.GetPeriodicTable()

        for idx in priority_atoms:
            # Skip protected atoms when preserving scaffold
            if self.preserve_scaffold and idx in protected_atoms:
                continue
                
            orig = mol.GetAtomWithIdx(idx).GetSymbol()
            for new in atom_mutation_table.get(orig, []):
                rw = Chem.RWMol(mol)
                rw.GetAtomWithIdx(idx).SetAtomicNum(pt.GetAtomicNumber(new))
                try:
                    m = rw.GetMol()
                    Chem.SanitizeMol(m)
                    
                    if Chem.MolToSmiles(m) != mol_smiles:
                        # Calculate structural similarity
                        struct_sim = calculate_similarity(mol, m, "morgan")
                        
                        # Calculate property similarity
                        new_props = calculate_properties(m)
                        prop_sim = property_similarity(original_props, new_props)
                        
                        # Apply additional property constraints
                        if self.preserve_hba_hbd:
                            if abs(original_props["HBD"] - new_props["HBD"]) > 1 or \
                               abs(original_props["HBA"] - new_props["HBA"]) > 1:
                                continue
                                
                        if self.preserve_logp:
                            if abs(original_props["LogP"] - new_props["LogP"]) > self.max_logp_diff:
                                continue
                                
                        if self.preserve_tpsa:
                            if abs(original_props["TPSA"] - new_props["TPSA"]) > self.max_tpsa_diff:
                                continue
                        
                        # Add if passes all thresholds
                        if struct_sim >= self.min_similarity and prop_sim >= self.min_property_similarity:
                            print(f"    Created valid molecule by changing atom {idx} from {orig} to {new} "
                                  f"(struct_sim={struct_sim:.2f}, prop_sim={prop_sim:.2f})")
                            cands.append((m, struct_sim, prop_sim))
                except Exception:
                    pass

        # ------------------- Method 5: Bond substitution -----------------
        if self.enable_bond_substitution:
            print("[INFO] Method 5: Bond substitution with property preservation")
            # Get modifiable bonds
            modifiable_bonds = get_modifiable_bonds(mol, scaffold if self.preserve_scaffold else None)
            
            for bond_idx, new_types in modifiable_bonds:
                bond = mol.GetBondWithIdx(bond_idx)
                begin_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
                end_atom = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
                
                for new_type in new_types:
                    modified_mol = substitute_bond(mol, bond_idx, new_type)
                    
                    if modified_mol and Chem.MolToSmiles(modified_mol) != mol_smiles:
                        # Calculate structural similarity
                        struct_sim = calculate_similarity(mol, modified_mol, "morgan")
                        
                        # Calculate property similarity
                        new_props = calculate_properties(modified_mol)
                        prop_sim = property_similarity(original_props, new_props)
                        
                        # Apply additional property constraints
                        if self.preserve_hba_hbd:
                            if abs(original_props["HBD"] - new_props["HBD"]) > 1 or \
                               abs(original_props["HBA"] - new_props["HBA"]) > 1:
                                continue
                                
                        if self.preserve_logp:
                            if abs(original_props["LogP"] - new_props["LogP"]) > self.max_logp_diff:
                                continue
                                
                        if self.preserve_tpsa:
                            if abs(original_props["TPSA"] - new_props["TPSA"]) > self.max_tpsa_diff:
                                continue
                        
                        # Add if passes all thresholds
                        if struct_sim >= self.min_similarity and prop_sim >= self.min_property_similarity:
                            print(f"    Created valid molecule by changing bond {bond_idx} from {bond.GetBondType()} to {new_type} "
                                  f"(struct_sim={struct_sim:.2f}, prop_sim={prop_sim:.2f})")
                            cands.append((modified_mol, struct_sim, prop_sim))

        # ------------------- Method 6: Atom addition -----------------
        if self.enable_atom_addition:
            print("[INFO] Method 6: Atom addition with property preservation")
            # Get atom addition candidates
            addition_candidates = get_atom_addition_candidates(mol, scaffold if self.preserve_scaffold else None)
            
            for atom_idx, new_atoms in addition_candidates:
                for new_atom in new_atoms:
                    # Try both single and double bonds
                    for bond_type in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]:
                        modified_mol = add_atom_to_molecule(mol, atom_idx, new_atom, bond_type)
                        
                        if modified_mol and Chem.MolToSmiles(modified_mol) != mol_smiles:
                            # Check size similarity
                            if not has_similar_size(mol, modified_mol, self.max_size_diff_ratio):
                                continue
                                
                            # Calculate structural similarity
                            struct_sim = calculate_similarity(mol, modified_mol, "morgan")
                            
                            # Check scaffold preservation
                            if self.preserve_scaffold and scaffold:
                                if not has_same_scaffold(mol, modified_mol):
                                    continue
                            
                            # Calculate property similarity
                            new_props = calculate_properties(modified_mol)
                            prop_sim = property_similarity(original_props, new_props)
                            
                            # Apply additional property constraints
                            if self.preserve_hba_hbd:
                                if abs(original_props["HBD"] - new_props["HBD"]) > 1 or \
                                   abs(original_props["HBA"] - new_props["HBA"]) > 1:
                                    continue
                                    
                            if self.preserve_logp:
                                if abs(original_props["LogP"] - new_props["LogP"]) > self.max_logp_diff:
                                    continue
                                    
                            if self.preserve_tpsa:
                                if abs(original_props["TPSA"] - new_props["TPSA"]) > self.max_tpsa_diff:
                                    continue
                            
                            # Add if passes all thresholds
                            if struct_sim >= self.min_similarity and prop_sim >= self.min_property_similarity:
                                print(f"    Created valid molecule by adding {new_atom} to atom {atom_idx} with {bond_type} bond "
                                      f"(struct_sim={struct_sim:.2f}, prop_sim={prop_sim:.2f})")
                                cands.append((modified_mol, struct_sim, prop_sim))

        # ------------------- Method 7: Bond addition -----------------
        if self.enable_bond_addition:
            print("[INFO] Method 7: Bond addition with property preservation")
            # Find potential bond additions
            bond_additions = find_potential_bond_additions(mol, scaffold if self.preserve_scaffold else None)
            
            for atom1_idx, atom2_idx in bond_additions:
                # Try adding a single bond
                modified_mol = add_bond_between_atoms(mol, atom1_idx, atom2_idx, Chem.BondType.SINGLE)
                
                if modified_mol and Chem.MolToSmiles(modified_mol) != mol_smiles:
                    # Calculate structural similarity
                    struct_sim = calculate_similarity(mol, modified_mol, "morgan")
                    
                    # Check scaffold preservation - bond addition could affect the scaffold
                    if self.preserve_scaffold and scaffold:
                        modified_scaffold = get_molecule_scaffold(modified_mol)
                        if modified_scaffold:
                            scaffold_sim = calculate_similarity(scaffold, modified_scaffold, "morgan")
                            if scaffold_sim < 0.7:
                                continue
                    
                    # Calculate property similarity
                    new_props = calculate_properties(modified_mol)
                    prop_sim = property_similarity(original_props, new_props)
                    
                    # Apply additional property constraints
                    if self.preserve_hba_hbd:
                        if abs(original_props["HBD"] - new_props["HBD"]) > 1 or \
                           abs(original_props["HBA"] - new_props["HBA"]) > 1:
                            continue
                            
                    if self.preserve_logp:
                        if abs(original_props["LogP"] - new_props["LogP"]) > self.max_logp_diff:
                            continue
                            
                    if self.preserve_tpsa:
                        if abs(original_props["TPSA"] - new_props["TPSA"]) > self.max_tpsa_diff:
                            continue
                    
                    # Add if passes all thresholds
                    if struct_sim >= self.min_similarity and prop_sim >= self.min_property_similarity:
                        print(f"    Created valid molecule by adding bond between atoms {atom1_idx} and {atom2_idx} "
                              f"(struct_sim={struct_sim:.2f}, prop_sim={prop_sim:.2f})")
                        cands.append((modified_mol, struct_sim, prop_sim))

        # --------------------- final dedup / sort by similarity ----------------------
        # Sort by uniqueness (to remove duplicates) and combined similarity score
        uniq_mols = self._dedup_valid([m for m, _, _ in cands], mol_smiles)
        
        # Re-calculate similarities for the unique molecules
        scored_mols = []
        for m in uniq_mols:
            struct_sim = calculate_similarity(mol, m, "morgan")
            new_props = calculate_properties(m)
            prop_sim = property_similarity(original_props, new_props)
            
            # Combined score with higher weight for property similarity
            combined_score = 0.3 * struct_sim + 0.7 * prop_sim
            scored_mols.append((m, combined_score, struct_sim, prop_sim))
            
        # Sort by combined score (descending)
        scored_mols.sort(key=lambda x: x[1], reverse=True)
        
        # Get molecules only
        final_mols = [m for m, _, _, _ in scored_mols[:n_candidates]]
        
        print(f"[INFO] Generated {len(final_mols)} unique valid candidates with property similarity ≥ {self.min_property_similarity}")
        
        if final_mols:
            print("[INFO] Top candidates:")
            for i, (m, combined, struct_sim, prop_sim) in enumerate(scored_mols[:min(5, len(scored_mols))]):
                props = calculate_properties(m)
                print(f"  {i+1}. {Chem.MolToSmiles(m)}")
                print(f"     Combined score: {combined:.2f} (struct_sim: {struct_sim:.2f}, prop_sim: {prop_sim:.2f})")
                print(f"     Properties: MW={props['MW']:.2f}, LogP={props['LogP']:.2f}, TPSA={props['TPSA']:.2f}, HBD={props['HBD']}, HBA={props['HBA']}")
                
        return final_mols

    # ────────────────────── small utilities ──────────────────────
    @staticmethod
    def _priority_atoms(mol: Chem.Mol, imp_frags: List[str]) -> List[int]:
        prio = set()
        for s in imp_frags:
            q = Chem.MolFromSmiles(s) if s else None
            if q:
                for match in mol.GetSubstructMatches(q):
                    prio.update(match)
        atoms = [(i, 1 if i in prio else 0) for i in range(mol.GetNumAtoms())]
        atoms.sort(key=lambda t: t[1], reverse=True)
        return [i for i, _ in atoms]

    @staticmethod
    def _dedup_valid(mols: Iterable[Chem.Mol], orig: str) -> List[Chem.Mol]:
        seen, out = set(), []
        for m in mols:
            if not m:
                continue
            try:
                Chem.SanitizeMol(m)
                smi = Chem.MolToSmiles(m)
                if smi != orig and smi not in seen:
                    out.append(m)
                    seen.add(smi)
            except Exception:
                pass
        return out