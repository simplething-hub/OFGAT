import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="scene15", help='Dataset to use.')
    parser.add_argument('--rep_num', type=int, default=5, help='Number of rep.')
    parser.add_argument('--epoch', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--k', type=int, default=5, help='k of kneighbors_graph.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nhid', type=int, default=256, help='the dimension of hidden layer')
    parser.add_argument('--alpha', type=float, default=0.2, help='for LeakyRelu')
    parser.add_argument('--nheads', type=int, default=1, help='num of attention head')
    parser.add_argument('--w1', type=float, default=1, help='sr loss weight')
    parser.add_argument('--w2', type=float, default=1, help='knn loss weight')
    parser.add_argument('--w3', type=float, default=0.1, help='l1 norm weight')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--res_path', type=str, default="./results/", help='Dataset to use.')
    parser.add_argument('--device', type=str, default="0", help='gpu')

    args = parser.parse_args()

    return args
