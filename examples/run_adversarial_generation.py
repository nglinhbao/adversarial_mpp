import os
import sys
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
import argparse
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adversarial_mol_gen.main import AdversarialMoleculeGenerator
from adversarial_mol_gen.utils import mol_to_smiles, smiles_to_mol

def parse_args():
    parser = argparse.ArgumentParser(description='Generate adversarial molecules (advanced)')
    parser.add_argument('--smiles', type=str, default="CC(=O)Oc1ccccc1C(=O)O",
                        help='SMILES string of the molecule to attack (default: aspirin)')
    parser.add_argument('--n_candidates', type=int, default=10000,
                        help='Number of candidates to generate')
    parser.add_argument('--max_selection', type=int, default=100,
                        help='Maximum number of adversarial examples to return')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--example', type=str, default='aspirin',
                        help='Predefined example (aspirin, paracetamol, ibuprofen)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold for adversarial selection (default: None)')
    
    return parser.parse_args()

def get_example_smiles(example_name):
    examples = {
        'aspirin': "CC(=O)Oc1ccccc1C(=O)O",
        'paracetamol': "CC(=O)Nc1ccc(O)cc1",
        'ibuprofen': "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        'caffeine': "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        'morphine': "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
        'cocaine': "COC(=O)C1C(CC2CCC1N2C)OC(=O)c3ccccc3"
    }
    return examples.get(example_name.lower(), examples['aspirin'])

def print_mol_properties(mol):
    """Print key properties of a molecule"""
    if mol is None:
        print("Invalid molecule")
        return
        
    print(f"SMILES: {Chem.MolToSmiles(mol)}")
    print(f"Molecular Weight: {Descriptors.MolWt(mol):.2f}")
    print(f"LogP: {Descriptors.MolLogP(mol):.2f}")
    print(f"TPSA: {Descriptors.TPSA(mol):.2f}")
    print(f"# Atoms: {mol.GetNumAtoms()}")
    print(f"# Bonds: {mol.GetNumBonds()}")
    print(f"# Rings: {Descriptors.RingCount(mol)}")
    print(f"# H-Donors: {Descriptors.NumHDonors(mol)}")
    print(f"# H-Acceptors: {Descriptors.NumHAcceptors(mol)}")
    print(f"# Rotatable Bonds: {Descriptors.NumRotatableBonds(mol)}")

def main():
    args = parse_args()
    
    # Use example if specified
    if args.example and args.example != 'aspirin':
        args.smiles = get_example_smiles(args.example)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    generator = AdversarialMoleculeGenerator(threshold=args.threshold)
    
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        print(f"Error: Invalid SMILES string: {args.smiles}")
        return
    
    print(f"Generating adversarial molecules for: {args.smiles}")
    print("\nOriginal Molecule Properties:")
    print_mol_properties(mol)
    
    # Generate adversarial molecules
    adversarial_mols = generator.generate(
        mol,
        n_candidates=args.n_candidates,
        max_selection=args.max_selection
    )
    
    # Print results
    print(f"\nResults:")
    print(f"Original molecule: {mol_to_smiles(mol)}")
    for i, adv_mol in enumerate(adversarial_mols):
        print(f"\nAdversarial molecule {i+1}: {mol_to_smiles(adv_mol)}")
        print(f"  Prediction: {generator.black_box_model.predict(adv_mol):.4f}")
        print(f"  Change: {generator.black_box_model.predict(adv_mol) - generator.black_box_model.predict(mol):.4f}")
        print("\nProperties:")
        print_mol_properties(adv_mol)
    
    # Save results
    output_path = generator.save_results(mol, adversarial_mols, args.output_dir)
    print(f"\nResults saved to: {output_path}")
    
    # Display visualization
    if adversarial_mols:
        img = generator.visualize_results(mol, adversarial_mols)
        img.save(os.path.join(args.output_dir, 'visualization.png'))
        print(f"Visualization saved to: {os.path.join(args.output_dir, 'visualization.png')}")
    
    print(f"\nDone!")

if __name__ == '__main__':
    main()