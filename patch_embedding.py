import torch
import torch.nn as nn
import math

class PatchingEmbedding(nn.Module):
    def __init__(self, input_patch_len, model_dim, max_len=5000):
        super(PatchingEmbedding, self).__init__()
        self.input_patch_len = input_patch_len
        self.model_dim = model_dim
        self.max_len = max_len

        # Input Layers (Residual Block)
        self.input_proj = nn.Linear(input_patch_len, model_dim)
        self.residual_block = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )
        self.layer_norm = nn.LayerNorm(model_dim)

        # Positional Encoding
        self.register_buffer('positional_encoding', self._create_positional_encoding())

    def _create_positional_encoding(self):
        position = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.model_dim, 2) * -(math.log(10000.0) / self.model_dim))
        pe = torch.zeros(self.max_len, self.model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, mask=None):
        # x shape: (batch_size, sequence_length)
        batch_size, seq_len = x.size()

        # 1. Patching
        num_patches = math.ceil(seq_len / self.input_patch_len)
        pad_len = num_patches * self.input_patch_len - seq_len
        x = torch.nn.functional.pad(x, (0, pad_len))
        x = x.view(batch_size, num_patches, self.input_patch_len)

        # Create mask if not provided
        if mask is None:
            mask = torch.zeros_like(x)
        else:
            mask = mask.view(batch_size, num_patches, self.input_patch_len)

        # 2. Input Layers (Residual Block)
        x = x * (1 - mask)  # Apply mask
        x = self.input_proj(x)  # Project to model_dim
        x = x + self.residual_block(x)  # Residual connection
        x = self.layer_norm(x)  # Layer norm

        # 3. Positional Encoding
        pe = self.positional_encoding[:, :num_patches, :]

        # 4. Embedding
        x = x + pe

        return x

# Test function
if __name__ == "__main__":
    input_patch_len = 32
    model_dim = 256
    max_len = 5000

    patching_embedding = PatchingEmbedding(input_patch_len=32, model_dim=256, max_len=5000)

    # Example input
    batch_size = 5
    sequence_length = 128
    x = torch.zeros(batch_size, sequence_length)

    # Forward pass
    embedded_patches = patching_embedding(x)
    print(f"Input shape: {x.shape}")
    # Expected: (batch_size, sequence_length) = (5, 128)
    print(f"Embedded patches shape: {embedded_patches.shape}")
    # Expected: (batch_size, num_patches, model_dim) = (5, 4, 256)
    # num_patches = ceil(sequence_length / input_patch_len) = ceil(128 / 32) = 4