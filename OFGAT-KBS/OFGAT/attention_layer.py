import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Optimized Multi-Head Graph Attention Layer with improved space complexity.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super(GraphAttentionLayer, self).__init__()

        self.W1 = nn.Parameter(torch.FloatTensor(hidden_dim, input_dim))
        self.W2 = nn.Parameter(torch.FloatTensor(hidden_dim, input_dim))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W1.size(1))
        self.W1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.W2.size(1))
        self.W2.data.uniform_(-stdv, stdv)



    def forward(self, x):
        x1 = self.W1 @ x.T
        x2 = self.W2 @ x.T
        att = x1.T @ x2
        output = F.softmax(att)
        # output = output - torch.diag_embed(torch.diag(output))
        return output
