import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, tau=1.0):
        """Initialize the InfoNCE Loss module.
        
        Args:
            tau (float, optional): Temperature parameter. Defaults to 1.0.
        """
        super().__init__()
        self.tau = tau

    def forward(self, concatenated, pos_indices, neg_indices):
        """Forward pass of the InfoNCE Loss.
        
        Args:
            concatenated (Tensor): Concatenated tensor.
            pos_indices (Tensor): Indices for positive samples.
            neg_indices (Tensor): Indices for negative samples.
            
        Returns:
            Tensor: Computed loss.
        """
        softmax = F.softmax(concatenated / self.tau, dim=1)
        pos_softmax = softmax.index_select(dim=0, index=pos_indices)
        neg_softmax = softmax.index_select(dim=0, index=neg_indices)
        dot_product = torch.mm(pos_softmax, neg_softmax.transpose(0, 1))
        loss = -torch.mean(torch.log(dot_product + 1e-8))
        return loss


class TripleContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature):
        sample_num = feature[0].shape[0]
        sample_index = list(range(sample_num))
        total_loss = 0
        loss_fn = InfoNCELoss()

        for i in sample_index:
            dna_embedding = feature[0][i]
            mrna_embedding = feature[1][i]
            mirna_embedding = feature[2][i]

            # Negative samples
            negative_sample1 = random.choice(
                [j for j in sample_index if j != i])
            dna_neg_embedding1 = feature[0][negative_sample1]
            mrna_neg_embedding1 = feature[1][negative_sample1]
            mirna_neg_embedding1 = feature[2][negative_sample1]

            negative_sample2 = random.choice(
                [j for j in sample_index if j not in (i, negative_sample1)])
            dna_neg_embedding2 = feature[0][negative_sample2]
            mrna_neg_embedding2 = feature[1][negative_sample2]
            mirna_neg_embedding2 = feature[2][negative_sample2]

            # Concatenate all embeddings
            concatenated = torch.stack((dna_embedding, mrna_embedding, mirna_embedding,
                                        dna_neg_embedding1, mrna_neg_embedding1, mirna_neg_embedding1,
                                        dna_neg_embedding2, mrna_neg_embedding2, mirna_neg_embedding2), dim=0)
            # Generate indices for the positive and negative embeddings
            pos_indices = torch.tensor([0, 1, 2], dtype=torch.long)
            neg_indices = torch.tensor(
                [3, 4, 5, 6, 7, 8], dtype=torch.long)

            # Apply InfoNCE loss
            total_loss += loss_fn(concatenated, pos_indices, neg_indices)

        return total_loss / (len(sample_index) * 9)
