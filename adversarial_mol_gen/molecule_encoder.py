import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem
import pickle
import os
from rdkit.Chem import rdFingerprintGenerator

class MoleculeVAE(nn.Module):
    """
    Variational Autoencoder for molecular encoding/decoding.
    This is a simplified version - in practice, you might want to use
    a pre-trained model like JTVAE, MolGPT, or other state-of-the-art models.
    """
    def __init__(self, input_dim=2048, hidden_dim=512, latent_dim=128):
        super(MoleculeVAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mean = self.fc2_mean(h)
        logvar = self.fc2_logvar(h)
        return mean, logvar
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)
        return x_recon, mean, logvar

class MoleculeEncoder:
    """
    Class for encoding molecules into latent space using VAE or flow models.
    """
    def __init__(self, model_type='vae', model_path=None, latent_dim=128):
        self.model_type = model_type
        self.latent_dim = latent_dim
        
        if model_type == 'vae':
            self.model = MoleculeVAE(input_dim=2048, latent_dim=latent_dim)
            if model_path and os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        else:
            raise ValueError(f"Model type {model_type} not supported.")
    
    def mol_to_fingerprint(self, mol):
        """Convert molecule to Morgan fingerprint."""
        if mol is None:
            return None
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return generator.GetFingerprint(mol)
    
    def encode(self, mol):
        """Encode molecule to latent space."""
        fp = self.mol_to_fingerprint(mol)
        if fp is None:
            return None
        
        fp_tensor = torch.FloatTensor(fp).unsqueeze(0)
        with torch.no_grad():
            mean, _ = self.model.encode(fp_tensor)
        return mean.numpy()[0]
    
    def decode(self, z, num_candidates=10):
        """
        Decode latent vector to molecule.
        In a real implementation, this would use more sophisticated methods
        to convert the output of the decoder back to a valid molecule.
        """
        z_tensor = torch.FloatTensor(z).unsqueeze(0)
        with torch.no_grad():
            fp_recon = self.model.decode(z_tensor).numpy()[0]
        
        # This is a placeholder - in reality, you would use a trained
        # model that can properly decode from fingerprint/latent space to molecule
        # Here we're just returning None to indicate this is a placeholder
        return None
    
    def save_model(self, path):
        """Save model to disk."""
        torch.save(self.model.state_dict(), path)
    
    def train_model(self, molecules, epochs=50, batch_size=32, lr=1e-3):
        """
        Train the VAE model on a dataset of molecules.
        """
        # Convert molecules to fingerprints
        fingerprints = [self.mol_to_fingerprint(mol) for mol in molecules]
        fingerprints = np.array([fp for fp in fingerprints if fp is not None])
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(fingerprints))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                
                # Forward pass
                x_recon, mean, logvar = self.model(x)
                
                # Compute loss
                recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader.dataset):.4f}")
        
        self.model.eval()