import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadGraphAttentionLayer(nn.Module):
    """
    Optimized Multi-Head Graph Attention Layer with improved space complexity.
    """

    def __init__(self, in_features, hid_features, n_heads, dropout, alpha, concat=False):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.hid_features = hid_features // n_heads  # 每个head的输出特征数
        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # 多头注意力的每个头的独立权重矩阵 W 和注意力参数 a
        self.W = nn.ParameterList([nn.Parameter(torch.empty(size=(in_features, self.hid_features))) for _ in range(n_heads)])
        self.a_src = nn.ParameterList([nn.Parameter(torch.empty(size=(self.hid_features, 1))) for _ in range(n_heads)])
        self.a_dst = nn.ParameterList([nn.Parameter(torch.empty(size=(self.hid_features, 1))) for _ in range(n_heads)])

        for w, a_s, a_d in zip(self.W, self.a_src, self.a_dst):
            nn.init.xavier_uniform_(w.data, gain=1.414)
            nn.init.xavier_uniform_(a_s.data, gain=1.414)
            nn.init.xavier_uniform_(a_d.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        # 对每个头进行注意力计算并收集结果
        attentions = [self._calculate_attention(h, self.W[i], self.a_src[i], self.a_dst[i]) for i in range(self.n_heads)]

        # 拼接或求平均
        if self.concat:
            mid = torch.cat(attentions, dim=-1)
            return mid - torch.diag(torch.diag_part(mid))  # 将所有头的输出拼接
        else:
            mid = torch.mean(torch.stack(attentions), dim=0)
            return mid  # 对所有头的输出求平均


    def _calculate_attention(self, h, W, a_src, a_dst):
        # 直接通过矩阵乘法避免显式构建 (N, N, in_features) 的拼接矩阵
        Wh = torch.matmul(h, W)  # (N, out_features)

        # 计算 e_ij = LeakyReLU(a^T [Wh_i || Wh_j])，通过拆分来减少空间开销
        e_src = torch.matmul(Wh, a_src)  # (N, 1) 对每个节点 i
        e_dst = torch.matmul(Wh, a_dst)  # (N, 1) 对每个节点 j
        e = e_src + e_dst.T  # (N, N) 广播加法
        # 计算 softmax 注意力矩阵
        attention = F.softmax(self.leakyrelu(e), dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)  # Dropout

        return attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features * self.n_heads) + ')'
