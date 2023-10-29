from .dirichlet import DS_Combin,ce_loss
from .layers import LinearLayer
from .contrastive_loss import TripleContrastiveLoss
from .attention import Attention
import torch.nn as nn
import torch


class HTMLModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout_rate = dropout

        self.FeatureInforEncoder = nn.ModuleList([LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.ModalityConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.ModalityClassifierLayer = nn.ModuleList([nn.Sequential(LinearLayer(hidden_dim[0],self.classes)) for _ in range(self.views)])
        
        self.Triple_Contrastive_loss=TripleContrastiveLoss()

        self.AttentionBlock =nn.ModuleList(Attention(in_dim[0],in_dim[view],hidden_dim=in_dim[view]) for view in range(1,self.views))
        
        self.criterion=nn.CrossEntropyLoss()
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh=nn.Tanh()

    def forward(self, data_list, label=None, infer=False):

        FeatureInfo, feature, Prediction, MCC ,DNA_Guided_view,attn_map= dict(), dict(), dict(), dict(),dict(),dict()
        for view in range(self.views):
            data_list[view]=data_list[view].squeeze(0)
            FeatureInfo[view] = torch.sigmoid(self.dropout(
                self.FeatureInforEncoder[view](data_list[view])))
            feature[view] = data_list[view] * FeatureInfo[view]


        for view in range(self.views):
            if view==0:
                DNA_Guided_view[view]=data_list[view]
            else:
                DNA_Guided_view[view],attn_map[view]=self.AttentionBlock[view-1](data_list[0],data_list[view])
                DNA_Guided_view[view]=torch.sigmoid(DNA_Guided_view[view])
            
        for view in range(self.views):
            feature[view]=torch.mean(torch.stack([feature[view],DNA_Guided_view[view]]), dim=0)
            
            feature[view] = self.dropout(feature[view])
        
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = self.dropout(feature[view])
        
        if not infer:
            MMLoss = self.Triple_Contrastive_loss(feature)


        # Classifier
        for view in range(self.views):
            Prediction[view] = self.dropout(self.ModalityClassifierLayer[view](feature[view]).squeeze(0))
                
            # Modality Confidence Calculation
            MCC[view] = torch.sigmoid(self.ModalityConfidenceLayer[view](feature[view]))
    
        if self.views==3:
            depth_evidence, rgb_evidence, pseudo_evidence = MCC[0]*torch.sigmoid(Prediction[0]), MCC[1]*torch.sigmoid(Prediction[1]), MCC[2]*torch.sigmoid(Prediction[2])
            depth_alpha, rgb_alpha, pseudo_alpha = depth_evidence+1, rgb_evidence+1, pseudo_evidence+1
            MMlogit,uncertainty= DS_Combin([depth_alpha, rgb_alpha,pseudo_alpha],classes=self.classes)
        
        if self.views==2:
            depth_evidence, rgb_evidence = MCC[0]*torch.sigmoid(Prediction[0]), MCC[1]*torch.sigmoid(Prediction[1])
            depth_alpha, rgb_alpha = depth_evidence+1, rgb_evidence+1
            MMlogit,uncertainty= DS_Combin([depth_alpha, rgb_alpha],classes=self.classes)
        
        if self.views==1:
            MMlogit=torch.sigmoid(Prediction[0])
            uncertainty=torch.tensor(0)
        
        if infer:
            return MMlogit, uncertainty.cpu().detach().numpy()
        
        MMLoss = MMLoss + torch.mean(self.criterion(Prediction[0],label))
        if self.views>1:
            MMLoss = MMLoss + torch.mean(self.criterion(Prediction[1],label))
        if self.views>2:
            MMLoss = MMLoss + torch.mean(self.criterion(Prediction[2],label))
        MMLoss = MMLoss + torch.mean(ce_loss(label,MMlogit,self.classes,0,1))
        return MMLoss, MMlogit, uncertainty.cpu().detach().numpy()

    
    def infer(self, data_list):
        MMlogit,uncertainty = self.forward(data_list, infer=True)
        return MMlogit,uncertainty