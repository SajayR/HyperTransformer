import torch
import torch.nn as nn
from patch_embedding import PatchingEmbedding
from hyper_transformer_block import HyperTransformerBlock
from weight_generator import WeightGenerator
from lstm import LSTMWithGeneratedWeights

class HyperTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 patch_size, d_model, num_heads, d_ff, dropout, max_seq_length, lstm_input_len, pred_len):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.lstm_input_len = lstm_input_len
        self.pred_len = pred_len
        self.patch_embed = PatchingEmbedding(patch_size, d_model, max_seq_length)

        # Separate Transformer for each LSTM layer
        self.transformers = nn.ModuleList([
            nn.Sequential(*[
                HyperTransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(3)  # Using 3 blocks per transformer
            ])
            for _ in range(num_layers)
        ])

        self.weight_generator = WeightGenerator(d_model, hidden_size, num_layers, input_size, output_size)
        self.lstm = LSTMWithGeneratedWeights(input_size, hidden_size, output_size, num_layers, pred_len)

    def forward(self, x):
        # Split input into input and target sections
        x_input = x[:, :self.lstm_input_len]
        x_target = x[:, self.lstm_input_len:]

        # Patch and embed the input
        embedded = self.patch_embed(x_input)

        # Pass through separate transformers for each layer
        transformer_outputs = []
        for transformer in self.transformers:
            layer_output = transformer(embedded)
            transformer_outputs.append(layer_output[:, -1, :])  # Use the last token's embedding
            embedded = layer_output  # Use output as input for the next layer

        # Generate LSTM weights
        generated_weights = self.weight_generator(transformer_outputs)

        # Pass through LSTM
        lstm_output = self.lstm(x_input.unsqueeze(-1), generated_weights)

        return lstm_output

# Test function
def test_hypertransformer():
    # Hyperparameters
    batch_size = 32
    input_size = 1
    hidden_size = 128 
    num_layers = 2
    output_size = 1
    patch_size = 32
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    max_seq_length = 5000
    lstm_input_len = 100
    pred_len = 28

    # Create model
    model = HyperTransformer(
        input_size, hidden_size, num_layers, output_size,
        patch_size, d_model, num_heads, d_ff, dropout, max_seq_length, lstm_input_len, pred_len
    )

    # Test input
    x = torch.randn(batch_size, lstm_input_len + pred_len)

    # Forward pass
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("HyperTransformer test completed successfully!")

if __name__ == "__main__":
    test_hypertransformer()