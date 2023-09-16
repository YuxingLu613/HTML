import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from collections import Counter
from model.html_model import HTMLModel
from preprocess.preprocessing import prepare_data_kfold
from utils.utils import seed_it, save_checkpoint, load_checkpoint, computeAUROC, computeAUPRC
from tqdm import trange

cuda = torch.cuda.is_available()

seed_it(40)

def train_epoch(data_list, label, model, optimizer):
    model.train()
    optimizer.zero_grad()
    loss, _, uncertainty = model(data_list, label)
    l1_loss = 0.
    lambda_ = 0.0001
    for param in model.parameters():
        l1_loss += torch.abs(param).sum()
    loss = torch.mean(loss) + lambda_ * l1_loss
    loss.backward()
    optimizer.step()

def test_epoch(data_list, model):
    model.eval()
    with torch.no_grad():
        logit, uncertainty = model.infer(data_list)
        prob = F.softmax(logit, dim=1).data.cpu().numpy()
    return prob, uncertainty
    

from preprocess.preprocessing import prepare_data_kfold
from model import HTMLModel
from utils.utils import computeAUROC, computeAUPRC, save_checkpoint, load_checkpoint

def train_kfold(data_folder, modelpath, testonly):
    
    test_inverval = 1
    hidden_dim = [1000]
    num_epoch = 2000
    lr = 1e-4

    kfold_data = prepare_data_kfold(data_folder, k=5, modalities=[1, 2, 3])
    best_result = {"acc": [], "f1-macro": [], "f1-weighted": [], "auc": [], "prc": [], "uncertainty": []}
    
    for k_iter in range(5):
        data_tr_list, data_test_list, train_labels, test_labels = kfold_data
        
        data_tr_list = data_tr_list[k_iter]
        data_test_list = data_test_list[k_iter]
        train_labels = torch.tensor(train_labels[k_iter])
        test_labels = test_labels[k_iter]
        
        print(Counter(test_labels))
        
        num_class = max(train_labels).item() + 1
        
        dim_list = [x.shape[1] for x in data_tr_list]
        model = HTMLModel(dim_list, hidden_dim, num_class, dropout=0.5)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
        
        labels = torch.tensor([])
        for i in test_labels:
            _ = [0] * (max(test_labels) + 1)
            _[i] = 1
            labels = torch.cat([labels, torch.tensor([_])], dim=0)
        
        if testonly:
            load_checkpoint(model, os.path.join(modelpath, data_folder[8:-1], "checkpoint.pt"))
            te_prob, uncertainty = test_epoch(data_test_list, model)
            if num_class == 2:
                print("Test ACC: {:.5f}".format(accuracy_score(test_labels, te_prob.argmax(1)) * 100))
                print("Test F1: {:.5f}".format(f1_score(test_labels, te_prob.argmax(1)) * 100))
                print("Test AUC: {:.5f}".format(roc_auc_score(test_labels, te_prob[:, 1]) * 100))
                print("Test PRC: {:.5f}".format(average_precision_score(test_labels, te_prob[:, 1]) * 100))
                print("Test Uncertainty:{:.5f}".format(np.mean(uncertainty)))
            else:
                print("Test ACC: {:.5f}".format(accuracy_score(test_labels, te_prob.argmax(1)) * 100))
                print("Test F1: {:.5f}".format(f1_score(test_labels, te_prob.argmax(1), average='macro') * 100))
                print("Test average AUC: {:.5f}".format(np.mean(computeAUROC(labels, te_prob, num_class)) * 100))
                print("Test average PRC: {:.5f}".format(np.mean(computeAUPRC(labels, te_prob, num_class)) * 100))
                print("Test Uncertainty:{:.5f}".format(np.mean(uncertainty)))
        else:
            print("\nTraining...")
            best_acc = 0
            best_f1_macro = 0
            best_f1_weighted = 0
            best_auc = 0
            best_uncertainty = 100
            for epoch in trange(num_epoch + 1):
                train_epoch(data_tr_list, train_labels, model, optimizer)
                scheduler.step()
                if epoch % test_inverval == 0:
                    te_prob, uncertainty = test_epoch(data_test_list, model)
                    # print("\nTrain: Epoch {:d}".format(epoch))
                    if num_class == 2:
                        pass
                        # print("Train ACC: {:.5f}".format(accuracy_score(test_labels, te_prob.argmax(1)) * 100))
                        # print("Train F1-macro: {:.5f}".format(f1_score(test_labels, te_prob.argmax(1), average='macro') * 100))
                        # print("Train F1-weighted: {:.5f}".format(f1_score(test_labels, te_prob.argmax(1), average='weighted') * 100))
                        # print("Train AUC: {:.5f}".format(roc_auc_score(test_labels, te_prob[:, 1]) * 100))
                        # print("Train PRC: {:.5f}".format(average_precision_score(test_labels, te_prob[:, 1]) * 100))
                        # print("Train Uncertainty:{:.5f}".format(np.mean(uncertainty) * 100))
                    else:
                        pass
                        # print("Train ACC: {:.5f}".format(accuracy_score(test_labels, te_prob.argmax(1)) * 100))
                        # print("Train F1-macro: {:.5f}".format(f1_score(test_labels, te_prob.argmax(1), average='macro') * 100))
                        # print("Train F1-weighted: {:.5f}".format(f1_score(test_labels, te_prob.argmax(1), average='weighted') * 100))
                        # print("Train average AUC: {:.5f}".format(np.mean(computeAUROC(labels, te_prob, num_class)) * 100))
                        # print("Train average PRC: {:.5f}".format(np.mean(computeAUPRC(labels, te_prob, num_class)) * 100))
                        # print("Train Uncertainty:{:.5f}".format(np.mean(uncertainty) * 100))
                    if accuracy_score(test_labels, te_prob.argmax(1)) * 100 >= best_acc:
                        best_acc = accuracy_score(test_labels, te_prob.argmax(1)) * 100
                        best_f1_macro = f1_score(test_labels, te_prob.argmax(1), average='macro') * 100
                        best_f1_weighted = f1_score(test_labels, te_prob.argmax(1), average='weighted') * 100
                        best_auc = np.mean(computeAUROC(labels, te_prob, num_class)) * 100
                        best_prc = np.mean(computeAUPRC(labels, te_prob, num_class)) * 100
                        best_uncertainty = np.mean(uncertainty) * 100
                        save_checkpoint(model.state_dict(), os.path.join(modelpath, data_folder[8:]))
                        
            print(f"k_iter:{k_iter}")
            print(f"acc:{best_acc}")
            print(f"f1_macro:{best_f1_macro}")
            print(f"f1_weighted:{best_f1_weighted}")
            print(f"auc:{best_auc}")
            print(f"prc:{best_prc}")
            print(f"uncertainty:{best_uncertainty}")
            
            best_result["acc"].append(best_acc)
            best_result["f1-macro"].append(best_f1_macro)
            best_result["f1-weighted"].append(best_f1_weighted)
            best_result["prc"].append(best_prc)
            best_result["auc"].append(best_auc)
            best_result["uncertainty"].append(best_uncertainty)
    
    return best_result