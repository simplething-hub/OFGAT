import os

import numpy as np
import scipy.io as scio
from sklearn.neighbors import kneighbors_graph
import torch
import scipy.sparse as sp
import random

### Data normalization
def normalization(data):
    maxVal = torch.max(data)
    minVal = torch.min(data)
    data = (data - minVal) // (maxVal - minVal)
    return data


### Data standardization
def standardization(data):
    rowSum = torch.sqrt(torch.sum(data ** 2, 1))
    repMat = rowSum.repeat((data.shape[1], 1)) + 1e-10
    data = torch.div(data, repMat.t())
    return data


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def construct_laplacian(adj):
    """
        construct the Laplacian matrix
    :param adj: original Laplacian matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj_ = sp.coo_matrix(adj)
    # adj_ = sp.eye(adj.shape[0]) + adj
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    # print("mean_degree:", rowsum.mean())
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_wave = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    lp = sp.eye(adj.shape[0]) - adj_wave
    # lp = adj_wave
    return lp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def LoadMatData(datasets, k, path):
    print("loading {} data...".format(datasets))
    data = scio.loadmat("{}{}.mat".format(path, datasets))
    x = data["X"]
    adj = []
    features = []
    nfeats = []
    for i in range(x.shape[1]):
        if datasets == '100leaves':
            x[0, i] = np.transpose(x[0, i])
        features.append(x[0, i].astype('float32'))
        nfeats.append(len(features[i][1]))
        temp = kneighbors_graph(features[i], k)
        temp = sp.coo_matrix(temp)
        temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
        temp = normalize(temp + sp.eye(temp.shape[0]))
        temp = sparse_mx_to_torch_sparse_tensor(temp)
        adj.append(temp)
    labels = data["Y"].reshape(-1, ).astype('int64')
    labels = labels - min(set(labels))
    num_class = len(set(np.array(labels)))
    labels = torch.from_numpy(labels)
    return adj, features, labels, nfeats, len(nfeats), num_class
    ''' # 每个类中取一个
    label_idx_list = [ [] for la in range(num_class)]
    for idx, label in enumerate(labels):
        label_idx_list[label].append(idx)
    label_idx_list = [label_set[0] for label_set in label_idx_list]
    all_views = [ [] for v in range(len(features))]
    for v, view in enumerate(features):
        for la in label_idx_list:
            all_views[v].append(view[la])
    for v in range(len(all_views)):
        all_views[v] = torch.tensor(all_views[v])
    label_ten = [0,1,2,3,4,5,6,7,8,9]
    #label_ten = np.array(label_ten)
    label_ten = torch.tensor(label_ten)
    #两个类中取5个
    all_views = [[] for v in range(len(features))]
    for v, view in enumerate(features):
        for la in [0,200]:
            for a in [0,1,2,3,4]:
                all_views[v].append(view[la+a])
    for v in range(len(all_views)):
        all_views[v] = torch.tensor(all_views[v])
    label_ten = [0, 1]
    # label_ten = np.array(label_ten)
    label_ten = torch.tensor(label_ten)
    num_class = 2
    #重新取类别的knn
    for i in range(len(all_views)):
        temp = kneighbors_graph(all_views[i], k)
        temp = sp.coo_matrix(temp)
        temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
        temp = normalize(temp + sp.eye(temp.shape[0]))
        temp = sparse_mx_to_torch_sparse_tensor(temp)
        adj.append(temp)
    return adj, all_views, label_ten, nfeats, len(nfeats), num_class'''


def feature_normalization(features, normalization_type='normalize'):
    for idx, fea in enumerate(features[0]):
        if normalization_type == 'normalize':
            features[0][idx] = normalize(fea)
        else:
            print("Please enter a correct normalization type!")
    return features



