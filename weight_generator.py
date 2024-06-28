import torch
import torch.nn as nn

class WeightGeneratorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        return torch.matmul(input, weight.t()) + bias

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad_output.t(), input)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class WeightGenerator(nn.Module):
    def __init__(self, transformer_output_dim, hidden_size, num_lstm_layers, input_size, output_size):
        super().__init__()
        self.transformer_output_dim = transformer_output_dim
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.input_size = input_size
        self.output_size = output_size
        self.weight_generators = nn.ModuleDict()

        for layer in range(num_lstm_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            layer_generators = nn.ModuleDict({
                f"weight_ih_l{layer}": nn.Linear(transformer_output_dim, 4 * hidden_size * layer_input_size),
                f"weight_hh_l{layer}": nn.Linear(transformer_output_dim, 4 * hidden_size * hidden_size),
                f"bias_ih_l{layer}": nn.Linear(transformer_output_dim, 4 * hidden_size),
                f"bias_hh_l{layer}": nn.Linear(transformer_output_dim, 4 * hidden_size)
            })
            self.weight_generators[f"layer_{layer}"] = layer_generators

        self.weight_generators["fc"] = nn.ModuleDict({
            "fc_weight": nn.Linear(transformer_output_dim, hidden_size * output_size),
            "fc_bias": nn.Linear(transformer_output_dim, output_size)
        })

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=0.1)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, transformer_outputs):
        batch_size = transformer_outputs[0].shape[0]
        generated_weights = {}

        for layer, transformer_output in enumerate(transformer_outputs):
            layer_input_size = self.input_size if layer == 0 else self.hidden_size
            for name, linear in self.weight_generators[f"layer_{layer}"].items():
                weights = WeightGeneratorFunction.apply(transformer_output, linear.weight, linear.bias)
                if 'weight_ih' in name or 'weight_hh' in name:
                    shape = (4 * self.hidden_size, layer_input_size if 'ih' in name else self.hidden_size)
                else:  # bias
                    shape = (4 * self.hidden_size,)
                generated_weights[name] = torch.tanh(weights.view(batch_size, *shape)) * 0.1

        fc_linear_weight = self.weight_generators["fc"]["fc_weight"]
        fc_linear_bias = self.weight_generators["fc"]["fc_bias"]
        fc_weights = WeightGeneratorFunction.apply(transformer_outputs[-1], fc_linear_weight.weight, fc_linear_weight.bias)
        fc_bias = WeightGeneratorFunction.apply(transformer_outputs[-1], fc_linear_bias.weight, fc_linear_bias.bias)
        generated_weights["fc_weight"] = torch.tanh(fc_weights.view(batch_size, self.hidden_size, self.output_size)) * 0.01
        generated_weights["fc_bias"] = torch.tanh(fc_bias.view(batch_size, self.output_size)) * 0.01

        return generated_weights

# Test function
if __name__ == "__main__":
    transformer_output_dim = 512
    num_lstm_layers = 2
    hidden_size = 128
    input_size = 1
    output_size = 1
    batch_size = 32

    weight_generator = WeightGenerator(transformer_output_dim, hidden_size, num_lstm_layers, input_size, output_size)

    # Simulate transformer outputs for each layer
    transformer_outputs = [torch.randn(batch_size, transformer_output_dim) for _ in range(num_lstm_layers)]

    # Generate weights
    generated_weights = weight_generator(transformer_outputs)

    # Print shapes of generated weights
    print("Generated weights shapes:")
    for name, weight in generated_weights.items():
        print(f"{name}: {weight.shape}")