from torch import nn
from mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, h_dim, num_layers=2, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_forward = MLP(d_model, d_model, h_dim, num_layers, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, X):
        # Attention
        q_new, _ = self.attention(X, X, X)
        q_new = X + q_new
        q_new = self.layer_norm1(q_new)

        # Feed forward network
        f_out = self.dropout_1(self.feed_forward(q_new))
        f_out = self.layer_norm2(f_out + q_new)
        return f_out
