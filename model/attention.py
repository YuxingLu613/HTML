import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, query_dim, value_dim, hidden_dim=1000):
        """Initialize the Attention module.
        
        Args:
            query_dim (int): Dimensionality of the query.
            value_dim (int): Dimensionality of the value.
            hidden_dim (int, optional): Hidden dimensionality. Defaults to 1000.
        """
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
        """Forward pass of the Attention mechanism.
        
        Args:
            query (Tensor): Query tensor.
            value (Tensor): Value tensor.
            
        Returns:
            Tensor: Context vectors.
            Tensor: Attention weights.
        """
        query = self.query_proj(query)
        key = self.key_proj(value)
        value = self.value_proj(value)

        scores = torch.matmul(query.transpose(-2, -1), key) / self.scaling
        weights = F.softmax(scores, dim=-1)
        weights = self.layer_norm(weights)
        context = torch.matmul(value, weights)
        
        return context, weights
