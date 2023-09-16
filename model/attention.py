import torch.nn as nn
import math
import torch
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, query_dim, value_dim, hidden_dim=1000):
        super(Attention, self).__init__()
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(value_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.scaling = math.sqrt(hidden_dim)

    def forward(self, query, value):
        query = self.query_proj(query)
        key = self.key_proj(value)
        value = self.value_proj(value)

        scores = torch.matmul(query.transpose(-2, -1), key) / self.scaling
        weights = F.softmax(scores, dim=-1)

        weights = self.layer_norm(weights)
        context = torch.matmul(value, weights)
        attn_map = weights.detach().cpu().numpy()

        return context, attn_map
