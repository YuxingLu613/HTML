import os
import random
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def seed_it(seed):
    # Set random seeds for reproducibility
    random.seed(seed) 
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
    
    
def one_hot_tensor(y, num_dim):
    # Convert label tensor to one-hot tensor
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    return y_onehot


def save_checkpoint(model, checkpoint_path, filename="checkpoint.pt"):
    # Save model checkpoint to file
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(model, filename)
    

def load_checkpoint(model, path):
    # Load model checkpoint from file
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint)
    
    
def computeAUROC(dataGT, dataPRED, classCount=5):
    # Compute area under ROC curve for each class
    outAUROC = []
        
    datanpGT = dataGT
    datanpPRED = dataPRED
    dataIndex = torch.argmax(dataGT, dim=1)

    for i in range(classCount):
        if i in dataIndex:
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
    return outAUROC


def computeAUPRC(dataGT, dataPRED, classCount=5):
    # Compute area under precision-recall curve for each class
    outAUPRC = []

    datanpGT = dataGT
    datanpPRED = dataPRED
    dataIndex = torch.argmax(dataGT, dim=1)

    for i in range(classCount):
        if i in dataIndex:
            outAUPRC.append(average_precision_score(datanpGT[:, i], datanpPRED[:, i]))
            
    return outAUPRC