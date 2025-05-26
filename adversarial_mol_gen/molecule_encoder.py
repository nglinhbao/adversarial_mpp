import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem
import pickle
import os
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import cDataStructs as DataStructs

class MoleculeVAE(nn.Module):
    def __init__(self):
        super(MoleculeVAE, self).__init__()

        self.conv_1 = nn.Conv1d(120, 9, kernel_size=9)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)
        self.linear_0 = nn.Linear(70, 435)
        self.linear_1 = nn.Linear(435, 292)
        self.linear_2 = nn.Linear(435, 292)

        self.linear_3 = nn.Linear(292, 292)
        self.gru = nn.GRU(292, 501, 3, batch_first=True)
        self.linear_4 = nn.Linear(501, 33)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def encode(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        return self.linear_1(x), self.linear_2(x)

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_4(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar

class MoleculeEncoder:
    """
    Class for encoding molecules into latent space using VAE or flow models.
    """
    def __init__(self, model_type='vae', model_path=None, latent_dim=128):
        self.model_type = model_type
        self.latent_dim = latent_dim
        
        if model_type == 'vae':
            self.model = MoleculeVAE()
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
        fp = generator.GetFingerprint(mol)
        
        # Convert to numpy array
        fp_array = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        
        # Calculate required input sequence length
        # After 3 convolutions with kernels 9,9,11, we need to end up with 70 features
        # Working backwards: 10 channels × 7 = 70 features after flatten
        # So sequence_length needs to be 7+11-1=17 before 3rd conv
        # Then 17+9-1=25 before 2nd conv
        # Then 25+9-1=33 before 1st conv
        seq_length = 33
        
        # Total length required for 120 channels with sequence length 33
        target_length = 120 * seq_length
        
        # Pad the array with zeros
        padded_fp = np.zeros((target_length,), dtype=np.float32)
        padded_fp[:2048] = fp_array
        
        # Reshape to 120 channels
        fp_reshaped = padded_fp.reshape(120, seq_length)
        
        return fp_reshaped

    def encode(self, mol):
        """Encode molecule to latent space."""
        fp = self.mol_to_fingerprint(mol)
        if fp is None:
            return None
        
        # Add batch dimension
        fp_tensor = torch.FloatTensor(fp).unsqueeze(0)  # Shape: [1, 120, seq_length]
        
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