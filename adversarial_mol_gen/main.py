from rdkit import Chem
import numpy as np
import os
import pickle

from .molecule_encoder import MoleculeEncoder
from .latent_perturbation import LatentPerturbation
from .fragment_analysis import FragmentAnalysis
from .fragment_substitution import FragmentSubstitution
from .model_query import BlackBoxModel, ModelQuery
from .utils import mol_to_smiles, smiles_to_mol, calculate_similarity, visualize_molecules

class AdversarialMoleculeGenerator:
    """
    Main class for generating adversarial molecules.
    """
    def __init__(self, threshold=None, encoder_model_path=None, black_box_model_path=None, 
                fragment_library_path=None):
        """
        Initialize the adversarial molecule generator.
        
        Args:
            encoder_model_path: Path to encoder model (optional)
            black_box_model_path: Path to black box model (optional)
            fragment_library_path: Path to fragment library (optional)
        """
        # Initialize components
        self.black_box_model = BlackBoxModel(black_box_model_path)
        self.encoder = MoleculeEncoder(model_path=encoder_model_path)
        self.perturbation = LatentPerturbation(self.encoder, self.black_box_model)
        self.fragment_analysis = FragmentAnalysis(self.black_box_model)
        self.fragment_substitution = FragmentSubstitution(fragment_library_path)
        self.model_query = ModelQuery(self.black_box_model)
        self.threshold = threshold

    def generate(self, mol, target_label=None, n_candidates=20, max_selection=5):
        """
        Generate adversarial molecules.
        
        Args:
            mol: Original molecule (RDKit mol or SMILES)
            target_label: Target label (None for untargeted attack)
            n_candidates: Number of candidates to generate
            max_selection: Maximum number of examples to return
            
        Returns:
            List of adversarial molecules
        """
        # Convert SMILES to RDKit mol if needed
        if isinstance(mol, str):
            mol = smiles_to_mol(mol)
        
        if mol is None:
            raise ValueError("Invalid molecule")
        
        # Step 1: Encode the molecule to latent space
        print("Step 1: Encoding molecule to latent space")
        z_orig = self.encoder.encode(mol)
        
        # Step 2: Perturb the latent vector
        print("Step 2: Perturbing latent vector")
        z_adv = self.perturbation.perturb(mol, target_label)
        
        # Step 3: Analyze fragments
        print("Step 3: Analyzing important fragments")
        important_fragments = self.fragment_analysis.identify_important_fragments(mol)
        
        # Step 4: Generate candidate molecules with fragment substitutions
        print("Step 4: Generating candidates with fragment substitutions")
        candidates = self.fragment_substitution.generate_substitutions(
            mol, important_fragments, n_candidates=n_candidates
        )
        
        print(f"Generated {len(candidates)} candidate molecules")
        
        # Step 5: Query black-box model
        print("Step 5: Querying black-box model")
        results = self.model_query.query_candidates(mol, candidates)
        
        # Step 6: Select adversarial examples
        print("Step 6: Selecting adversarial examples")
        adversarial_mols = self.model_query.select_adversarial(
            results, threshold=self.threshold, max_selection=max_selection
        )
        
        print(f"Generated {len(adversarial_mols)} adversarial molecules.")
        return adversarial_mols
    
    def visualize_results(self, mol_orig, adversarial_mols, filename=None):
        """
        Visualize the original molecule and adversarial examples.
        
        Args:
            mol_orig: Original molecule
            adversarial_mols: List of adversarial molecules
            filename: Output filename (optional)
            
        Returns:
            Visualization image
        """
        all_mols = [mol_orig] + adversarial_mols
        labels = ['Original'] + [f'Adversarial {i+1}' for i in range(len(adversarial_mols))]
        
        return visualize_molecules(all_mols, labels, filename)
    
    def save_results(self, mol_orig, adversarial_mols, output_dir='results'):
        """
        Save the results to disk.
        
        Args:
            mol_orig: Original molecule
            adversarial_mols: List of adversarial molecules
            output_dir: Output directory
            
        Returns:
            Path to saved results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save SMILES
        with open(os.path.join(output_dir, 'smiles.txt'), 'w') as f:
            f.write(f"Original: {mol_to_smiles(mol_orig)}\n")
            for i, mol in enumerate(adversarial_mols):
                f.write(f"Adversarial {i+1}: {mol_to_smiles(mol)}\n")
        
        # Save visualization
        if adversarial_mols:
            img_path = os.path.join(output_dir, 'visualization.png')
            self.visualize_results(mol_orig, adversarial_mols, img_path)
        
        # Save predictions
        predictions = {}
        predictions['original'] = self.black_box_model.predict(mol_orig)
        for i, mol in enumerate(adversarial_mols):
            predictions[f'adversarial_{i+1}'] = self.black_box_model.predict(mol)
        
        with open(os.path.join(output_dir, 'predictions.pkl'), 'wb') as f:
            pickle.dump(predictions, f)
        
        return output_dir