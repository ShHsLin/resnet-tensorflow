import argparse
import sys


def parse_args():
    """
    parse arguments
    """
    parser = argparse.ArgumentParser(description='Parse for hyperparamters for training')
    parser.add_argument('--lr', dest='lr',
                        help='learning rate. Default: 1e-2',
                        default=1e-2, type=float)
    parser.add_argument('--net', dest='which_resnet',
                        help='Name of the Neural Network. Default: resnet_v1_29',
                        default='v1_29', type=str)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='Batchsize for training. Default: 128',
                        default=128, type=int)
    parser.add_argument('--ckpt_dir', dest='ckpt_dir',
                        help='Directory for checkpoint files and log, Default:Model/CIFAR10/Default/',
                        default='Model/CIFAR10/Default/', type=str)
    parser.add_argument('--opt', dest='which_opt',
                        help='optimizer for the neural network',
                        default='Mom', type=str)
    parser.add_argument('--regu', dest='regu',
                        help='coefficient for regularization',
                        default=0.0001, type=float)
    parser.add_argument('--init_step', dest='global_step',
                        help='global step in training',
                        default=0, type=int)
    parser.add_argument('--iter', dest='num_iter',
                        help='number of iteration to train',
                        default=50000, type=int)
    parser.add_argument('--bond_dim', dest='bond_dim',
                        help='bond dimension for tt-rank',
                        default=30, type=int)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args
