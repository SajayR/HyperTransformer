import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input, hidden, weights):
        batch_size = input.size(0)
        hx, cx = hidden

        weight_ih = weights['weight_ih'].view(batch_size, 4 * self.hidden_size, -1)
        weight_hh = weights['weight_hh'].view(batch_size, 4 * self.hidden_size, -1)
        bias_ih = weights['bias_ih'].view(batch_size, 4 * self.hidden_size)
        bias_hh = weights['bias_hh'].view(batch_size, 4 * self.hidden_size)

        gates = torch.bmm(weight_ih, input.unsqueeze(2)).squeeze(2) + \
                torch.bmm(weight_hh, hx.unsqueeze(2)).squeeze(2) + \
                bias_ih + bias_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        #ingate = torch.sigmoid(ingate)
        #forgetgate = torch.sigmoid(forgetgate)
        #cellgate = torch.tanh(cellgate)
        #outgate = torch.sigmoid(outgate)
        ingate = torch.sigmoid(self.layer_norm(ingate))
        forgetgate = torch.sigmoid(self.layer_norm(forgetgate))
        cellgate = torch.tanh(self.layer_norm(cellgate))
        outgate = torch.sigmoid(self.layer_norm(outgate))

        #cy = (forgetgate * cx) + (ingate * cellgate)
        #hy = outgate * torch.tanh(cy)
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(self.layer_norm(cy))

        return hy, cy

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([CustomLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

    def forward(self, input, generated_weights):
        batch_size, seq_len, _ = input.size()
        hidden = [torch.zeros(batch_size, self.hidden_size).to(input.device) for _ in range(2)]
        outputs = []

        for t in range(seq_len):
            x = input[:, t, :]
            for l in range(self.num_layers):
                layer_weights = {
                    'weight_ih': generated_weights[f'weight_ih_l{l}'],
                    'weight_hh': generated_weights[f'weight_hh_l{l}'],
                    'bias_ih': generated_weights[f'bias_ih_l{l}'],
                    'bias_hh': generated_weights[f'bias_hh_l{l}']
                }
                x, hidden[1] = self.cells[l](x, (hidden[0], hidden[1]), layer_weights)
                hidden[0] = x
            outputs.append(x)

        return torch.stack(outputs, dim=1)

class LSTMWithGeneratedWeights(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, pred_len):
        super(LSTMWithGeneratedWeights, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.lstm = CustomLSTM(input_size, hidden_size, num_layers)

    def forward(self, x, generated_weights):
        lstm_output = self.lstm(x, generated_weights)
        
        fc_weight = generated_weights['fc_weight']
        fc_bias = generated_weights['fc_bias']
        
        output = torch.bmm(lstm_output, fc_weight) + fc_bias.unsqueeze(1)

        return output[:, -self.pred_len:, :]


if __name__ == "__main__":
    input_size = 1
    hidden_size = 128
    output_size = 1
    num_layers = 2
    seq_len = 100
    pred_len = 28
    batch_size = 32

    model = LSTMWithGeneratedWeights(input_size, hidden_size, output_size, num_layers, pred_len)
    x_input = torch.randn(batch_size, seq_len, input_size)

    generated_weights = {
        "weight_ih_l0": torch.randn(batch_size, 4 * hidden_size, input_size),
        "weight_hh_l0": torch.randn(batch_size, 4 * hidden_size, hidden_size),
        "bias_ih_l0": torch.randn(batch_size, 4 * hidden_size),
        "bias_hh_l0": torch.randn(batch_size, 4 * hidden_size),
        "weight_ih_l1": torch.randn(batch_size, 4 * hidden_size, hidden_size),
        "weight_hh_l1": torch.randn(batch_size, 4 * hidden_size, hidden_size),
        "bias_ih_l1": torch.randn(batch_size, 4 * hidden_size),
        "bias_hh_l1": torch.randn(batch_size, 4 * hidden_size),
        "fc_weight": torch.randn(batch_size, hidden_size, output_size),
        "fc_bias": torch.randn(batch_size, output_size)
    }

    output = model(x_input, generated_weights)
    print("Output shape:", output.shape)
    print("Generated weights shapes:")
    for name, weight in generated_weights.items():
        print(f"{name}: {weight.shape}")