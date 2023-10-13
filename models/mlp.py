from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_p=0.1):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            if i == num_layers - 1:  # Last Layer
                layers.append(nn.Linear(prev_dim, output_dim))
                continue
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(True),
                nn.Dropout(dropout_p)
            ])
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
