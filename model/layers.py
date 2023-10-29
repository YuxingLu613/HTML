import torch.nn as nn

def xavier_init(m):
    """Initialize weights using Xavier normal initialization.
    
    Args:
        m (nn.Module): PyTorch module to initialize.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0) if m.bias is not None else None

class LinearLayer(nn.Module):
    """A simple linear layer with Xavier initialization.
    
    Attributes:
        clf (nn.Sequential): Sequential container for linear layer.
    """
    
    def __init__(self, in_dim, out_dim):
        super(LinearLayer, self).__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        """Forward pass through the linear layer.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Processed tensor.
        """
        return self.clf(x)
