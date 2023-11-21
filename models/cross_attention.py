from torch import nn
from mlp import MLP


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, h_dim, num_layers=2, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.feed_forward = MLP(d_model, d_model, h_dim, num_layers, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # Layer normalization
        q_new = self.layer_norm_1(q)
        k_new = self.layer_norm_1(k)
        v_new = self.layer_norm_1(v)

        # Attention
        q_new, _ = self.attention(q_new, k_new, v_new)
        q_new = self.dropout_1(q_new)

        # Add attended query to the original query
        q_new = q + q_new
        # Normalize
        q_new = self.layer_norm_2(q_new)
        # Pass through a feed forward layer
        ffn_out = self.dropout_2(self.feed_forward(q_new))
        ffn_out = self.layer_norm_3(ffn_out + q_new)
        return ffn_out
