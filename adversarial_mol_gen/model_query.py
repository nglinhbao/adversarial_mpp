import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pickle
import os
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs


class BlackBoxModel:
    """
    Interface for querying black-box models.
    This is a placeholder - in practice, you would connect to a real model.
    """
    def __init__(self, model_path=None):
        """
        Initialize the black-box model.
        
        Args:
            model_path: Path to model file (optional)
        """
        self.model = None
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
    
    def predict(self, mol):
        """
        Predict property for a molecule.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Prediction score
        """
        if self.model is not None:
            # Use loaded model for prediction
            return self._model_predict(mol)
        else:
            # Use a simple rule-based model as placeholder
            return self._simple_predict(mol)

    def _model_predict(self, mol):
        """
        Make prediction using loaded model.
        """
        # Use MorganGenerator (replacement for deprecated fingerprint API)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = generator.GetFingerprint(mol)

        # Convert to numpy array
        fp_array = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        
        # Make prediction
        return self.model.predict_proba(fp_array.reshape(1, -1))[0][1]

    
    def _simple_predict(self, mol):
        """
        Simple rule-based prediction for demonstration.
        This model is deterministic and based on molecular properties
        that can be easily manipulated by structural changes.
        """
        if mol is None:
            return 0.5
            
        # Calculate some molecular properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        num_rings = Descriptors.RingCount(mol)
        num_hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 6)
        
        # Simple formula for demonstration
        # This ensures small changes to the molecule can affect the prediction
        score = 0.5 + 0.01 * (mw/100) + 0.05 * logp - 0.02 * (tpsa/50) + 0.1 * (num_rings/3) + 0.1 * (num_hetero/5)
        
        # Ensure score is between 0 and 1
        return max(0, min(score, 1))

class ModelQuery:
    """
    Class for querying black-box models and selecting adversarial examples.
    """
    def __init__(self, black_box_model):
        """
        Initialize the model query.
        
        Args:
            black_box_model: Black-box model to query
        """
        self.black_box_model = black_box_model
    
    def query_candidates(self, mol_orig, candidates):
        """
        Query the black-box model with candidate molecules.
        
        Args:
            mol_orig: Original molecule
            candidates: List of candidate molecules
            
        Returns:
            List of (molecule, prediction, score) tuples
        """
        orig_pred = self.black_box_model.predict(mol_orig)
        print(f"Original molecule prediction: {orig_pred:.4f}")
        
        results = []
        for i, mol in enumerate(candidates):
            if mol is None:
                continue
                
            pred = self.black_box_model.predict(mol)
            
            # Calculate adversarial score (difference from original prediction)
            score = abs(pred - orig_pred)
            
            print(f"Candidate {i+1}: prediction={pred:.4f}, score={score:.4f}")
            results.append((mol, pred, score))
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results
    
    def select_adversarial(self, results, threshold=None, max_selection=5):
        """
        Select adversarial examples from results.
        
        Args:
            results: List of (molecule, prediction, score) tuples
            threshold: Minimum score threshold (lowered from 0.2 to 0.05)
            max_selection: Maximum number of examples to select
            
        Returns:
            List of selected adversarial molecules
        """
        selected = []
        
        for mol, pred, score in results:
            if (threshold and score >= threshold) or threshold is None:
                selected.append(mol)
            
            if len(selected) >= max_selection:
                break
        
        print(f"Selected {len(selected)} adversarial molecules (threshold={threshold})")
        return selected