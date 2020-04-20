import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Matrix Factorization With Biases
class LFM(torch.nn.Module):
    def __init__(self, num_user, num_item, hidden, mu):
        super(LFM, self).__init__()
        self.mu = mu
        self.user_embed = nn.Embedding(num_user, hidden)
        self.item_embed = nn.Embedding(num_item, hidden)
        self.bias_u = nn.Embedding(num_user, 1)
        self.bias_i = nn.Embedding(num_item, 1)
        self.init_params()

    def init_params(self):
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
        nn.init.constant_(self.bias_u.weight, 0.0)
        nn.init.constant_(self.bias_i.weight, 0.0)

    def forward(self, user_indexs, item_indexs):
        P = self.user_embed(user_indexs)  # [batch_num,hidden]
        Q = self.item_embed(item_indexs)  # [batch_num,hidden]
        interaction = torch.mul(P, Q).sum(dim=1).unsqueeze(dim=1)  # element-wise product [batch_num,1]
        return self.mu + self.bias_u(user_indexs) + self.bias_i(item_indexs) + interaction  # [batch_num,1]
