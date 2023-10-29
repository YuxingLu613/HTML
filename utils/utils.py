import os
import random
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

def seed_it(seed):
    """Set random seeds for various libraries to ensure reproducibility.
    
    Args:
        seed (int): The seed value.
    """
    random.seed(seed) 
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)

def one_hot_tensor(y, num_dim):
    """Convert label tensor to one-hot tensor.
    
    Args:
        y (Tensor): The label tensor.
        num_dim (int): The number of dimensions for the one-hot tensor.
        
    Returns:
        Tensor: One-hot encoded tensor.
    """
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    return y_onehot

def save_checkpoint(model, checkpoint_path, filename="checkpoint.pt"):
    """Save model checkpoint to a file.
    
    Args:
        model (nn.Module): The PyTorch model.
        checkpoint_path (str): Directory path to save the checkpoint.
        filename (str, optional): Name of the checkpoint file. Defaults to "checkpoint.pt".
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(model, filename)

def load_checkpoint(model, path):
    """Load model from a checkpoint file.
    
    Args:
        model (nn.Module): The PyTorch model structure.
        path (str): Path to the checkpoint file.
        
    Returns:
        nn.Module: The loaded model.
    """
    return torch.load(path)
