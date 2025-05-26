import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

class LatentPerturbation:
    """
    Class for performing latent space perturbation to find adversarial examples.
    """
    def __init__(self, encoder, black_box_model, epsilon=0.5, max_iter=100):
        """
        Initialize the latent perturbation.
        
        Args:
            encoder: The molecule encoder
            black_box_model: The black-box model to attack
            epsilon: Maximum perturbation magnitude
            max_iter: Maximum number of iterations
        """
        self.encoder = encoder
        self.black_box_model = black_box_model
        self.epsilon = epsilon
        self.max_iter = max_iter
    
    def perturb(self, mol_orig, target_label=None, method='gradient'):
        """
        Perturb the latent representation to find adversarial examples.
        
        Args:
            mol_orig: Original molecule
            target_label: Target label (None for untargeted attack)
            method: 'gradient' or 'optimization'
            
        Returns:
            Perturbed latent vector
        """
        z_orig = self.encoder.encode(mol_orig)
        
        if method == 'gradient':
            return self._gradient_based_perturbation(z_orig, mol_orig, target_label)
        elif method == 'optimization':
            return self._optimization_based_perturbation(z_orig, mol_orig, target_label)
        else:
            raise ValueError(f"Method {method} not supported.")
    
    def _gradient_based_perturbation(self, z_orig, mol_orig, target_label=None):
        """
        Gradient-based perturbation in latent space.
        In a real-world scenario with a black-box model, we would use
        zeroth-order optimization techniques to estimate gradients.
        """
        # This is a simplified version using random search as a placeholder
        # In practice, you would use more sophisticated techniques
        
        orig_pred = self.black_box_model.predict(mol_orig)
        
        best_z = None
        best_loss = float('-inf')
        
        for _ in range(self.max_iter):
            # Generate random perturbation
            delta = np.random.normal(0, 1, z_orig.shape)
            delta = delta / np.linalg.norm(delta) * self.epsilon
            
            # Apply perturbation
            z_perturbed = z_orig + delta
            
            # Decode perturbed latent vector (in practice, you would use a proper decoder)
            mol_perturbed = self._simulate_decoding(z_perturbed, mol_orig)
            
            if mol_perturbed is None:
                continue
            
            # Calculate loss
            if target_label is None:
                # Untargeted attack: maximize prediction difference
                perturbed_pred = self.black_box_model.predict(mol_perturbed)
                loss = abs(perturbed_pred - orig_pred)
            else:
                # Targeted attack: minimize distance to target
                perturbed_pred = self.black_box_model.predict(mol_perturbed)
                loss = -abs(perturbed_pred - target_label)
            
            # Update best perturbation
            if loss > best_loss:
                best_loss = loss
                best_z = z_perturbed
        
        return best_z
    
    def _optimization_based_perturbation(self, z_orig, mol_orig, target_label=None):
        """
        Optimization-based perturbation in latent space.
        Uses scipy's minimize to find optimal perturbation.
        """
        orig_pred = self.black_box_model.predict(mol_orig)
        
        # Define objective function
        def objective(delta):
            z_perturbed = z_orig + delta
            mol_perturbed = self._simulate_decoding(z_perturbed, mol_orig)
            
            if mol_perturbed is None:
                return 0
            
            perturbed_pred = self.black_box_model.predict(mol_perturbed)
            
            if target_label is None:
                # Untargeted attack: maximize prediction difference
                return -abs(perturbed_pred - orig_pred)
            else:
                # Targeted attack: minimize distance to target
                return abs(perturbed_pred - target_label)
        
        # Constraint: ||delta|| <= epsilon
        constraint = {'type': 'ineq', 'fun': lambda x: self.epsilon - np.linalg.norm(x)}
        
        # Initial perturbation
        delta0 = np.random.normal(0, 0.01, z_orig.shape)
        
        # Optimize
        result = minimize(
            objective, 
            delta0, 
            method='SLSQP',
            constraints=[constraint],
            options={'maxiter': self.max_iter}
        )
        
        # Apply optimal perturbation
        z_perturbed = z_orig + result.x
        
        return z_perturbed
    
    def _simulate_decoding(self, z_perturbed, mol_orig):
        """
        Simulate decoding for demonstration purposes.
        In a real implementation, you would use the actual decoder.
        """
        # In a real implementation, this would use the decoder
        # Here we're just returning the original molecule as a placeholder
        from rdkit import Chem
        return mol_orig